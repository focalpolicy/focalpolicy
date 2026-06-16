import sys
sys.path.append('FocalPolicy/focal_policy_3d')
from typing import Dict
import torch
from termcolor import cprint
import torch_dct as dct
from focal_policy_3d.model.common.normalizer import LinearNormalizer
from focal_policy_3d.policy.base_policy import BasePolicy
from focal_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D
from focal_policy_3d.model.flow.mask_generator import LowdimMaskGenerator
from focal_policy_3d.models.time_sampler import sample_two_timesteps
from focal_policy_3d.common.pytorch_util import dict_apply
from focal_policy_3d.common.model_util import print_params
from focal_policy_3d.model.vision.pointnet_extractor import FocalPolicyEncoder
import warnings
warnings.filterwarnings("ignore")

class FocalPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="mlp",
            pointcloud_encoder_cfg=None,
            freq_weight=1e-4,  
            sample_cfg=None,              
            eta=0.01,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: 
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        # point cloud encoder
        obs_encoder = FocalPolicyEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        #obs_as_global_cond=true
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[FocalPolicyEncoder] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[FocalPolicyEncoder] pointnet_type: {self.pointnet_type}", "yellow")
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.eta = eta
        self.eps = 1e-2
        self.sample_cfg = sample_cfg
        self.freq_weight = freq_weight
        cprint(f"[FocalPolicy] freq_weight: {self.freq_weight}", "yellow")
        cprint(f"[FocalPolicy] Horizon: {self.horizon}; n_action_steps: {self.n_action_steps}", "yellow")
        print_params(self)
        
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # Normalize observations and prepare tensors
        normalized_obs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            normalized_obs['point_cloud'] = normalized_obs['point_cloud'][..., :3]
        # shapes and dims
        batch_example = next(iter(normalized_obs.values()))
        batch_size, _ = batch_example.shape[:2]
        horizon = self.horizon
        action_dim = self.action_dim
        obs_feat_dim = self.obs_feature_dim
        n_obs_steps = self.n_obs_steps

        device = self.device
        dtype = self.dtype
        # prepare conditioning (local_cond unused here)
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # encode the initial observations as a global conditioning vector
            obs_for_encoding = dict_apply(normalized_obs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            encoded_obs = self.obs_encoder(obs_for_encoding)
            if "cross_attention" in self.condition_type:
                # keep temporal dimension for cross-attention
                global_cond = encoded_obs.reshape(batch_size, n_obs_steps, -1)
            else:
                # flatten temporal dimension into a single global vector
                global_cond = encoded_obs.reshape(batch_size, -1)
            cond_tensor = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_tensor, dtype=torch.bool)
        else:
            # use inpainting-style conditioning: append encoded obs to the early timesteps
            obs_for_encoding = dict_apply(normalized_obs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            encoded_obs = self.obs_encoder(obs_for_encoding)
            encoded_obs = encoded_obs.reshape(batch_size, n_obs_steps, -1)
            cond_tensor = torch.zeros(size=(batch_size, horizon, action_dim + obs_feat_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_tensor, dtype=torch.bool)
            cond_tensor[:, :n_obs_steps, action_dim:] = encoded_obs
            cond_mask[:, :n_obs_steps, action_dim:] = True
        # initialize latent with noise (same shape as conditioning tensor)
        noise = torch.randn(size=cond_tensor.shape, dtype=cond_tensor.dtype, device=cond_tensor.device)
        noise = noise.detach().clone()
        # single deterministic Euler step (matching original behaviour)
        dt = 1
        eps = self.eps
        t = torch.ones(noise.shape[0], device=noise.device) * eps
        pred_v = self.model(noise, t * 99, local_cond=local_cond, global_cond=global_cond)
        pred_action = noise + pred_v * dt
        # restore conditioned portions and convert back to action space
        pred_action[cond_mask] = cond_tensor[cond_mask]
        naction_pred = pred_action[..., :action_dim]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # select first action steps to return
        start_idx = n_obs_steps - 1
        end_idx = start_idx + self.n_action_steps
        action = action_pred[:, start_idx:end_idx]
        return {
            'action': action,
            'action_pred': action_pred,
        }
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    def compute_loss(self, batch):
        eps = self.eps
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # gt & noise
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps # 1=sde.T
        #### sampling r.
        # In this preliminary experiment, for simplicity, we directly adopt MeanFlow's time sampling strategy, which anchors r to 1.
        # Theoretically, any sampling scheme that leverages logit-normal sampling to concentrate the time anchor of r at 1 
        # is consistent with the intended effect of LAS, and is not restricted to the specific implementation.
        _, r = sample_two_timesteps(self.sample_cfg, num_samples=target.shape[0], device=target.device)
        r = 1 - r
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.-t_expand) * noise
        xr = r_expand * target + (1.-r_expand) * noise
        #apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]
        time_ends_expand = torch.ones_like(t_expand)
        vt = self.model(xt, t*99, local_cond=local_cond, global_cond=global_cond)
        with torch.no_grad():
            vr = self.model(xr, r*99, local_cond=local_cond, global_cond=global_cond)
        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]
        vr = torch.nan_to_num(vr)
        ft = self._f_euler(t_expand, time_ends_expand, xt, vt)
        fr = self._f_euler(r_expand, time_ends_expand, xr, vr)
        ################################## Spatial-domain consistency flow loss ##########################################
        # In the spatial domain, compute the consistency flow loss only over the short-term horizon.
        s_ft = ft[:, :self.n_action_steps, :]
        s_fr = fr[:, :self.n_action_steps, :]
        losses_f = torch.square(s_ft - s_fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
        ################################# Action frequency-domain loss ######################################
        # Capture the action-change distribution over the full horizon in the frequency domain.
        naction_pred = ft[...,:self.action_dim]  # B,T,Da
        naction_target = target  # B,T,Da
        # Convert to the frequency domain by applying DCT along the temporal dimension.
        freq_pred = dct.dct(naction_pred.transpose(1,2), norm='ortho').transpose(1,2)  # (B, T, Da)
        freq_target = dct.dct(naction_target.transpose(1,2), norm='ortho').transpose(1,2)  # (B, T, Da)
        # Compute the frequency-domain loss.
        loss_freq = self._frequency_loss(freq_target, freq_pred)
        ################################### Total loss ########################################
        loss_freq = torch.mean(loss_freq)
        loss_space = torch.mean(losses_f)
        loss = loss_space + self.freq_weight * loss_freq
        loss_dict = {
            'space_loss': loss_space.item(),
            'freq_loss': loss_freq.item(),
            'total_loss': loss.item(),
        }
        return loss, loss_dict

    def _f_euler(self, t_expand, time_ends_expand, xt, vt):
        """
        Formula:
            f_euler(t, segment_ends, x_t, v_t) = x_t + (segment_ends - t) * v_t
        Args:
            t_expand: Expanded timestep tensor (B, T, Da)
            time_ends_expand: Expanded segment-end tensor (B, T, Da)
            xt: State tensor at timestep t (B, T, Da)
            vt: Velocity tensor at timestep t (B, T, Da)
        Returns:
            Computed result tensor (B, T, Da)
        """
        return xt + (time_ends_expand - t_expand) * vt
    
    
    def _masked_losses_v(self, vt, vr, threshold, segment_ends, t, t_expand, reduce_op, trajectory, delta):
        """
        Compute the masked velocity loss.
        Args:
            vt: Velocity tensor v_t (B, T, Da)
            vr: Velocity tensor v_r (B, T, Da)
            threshold: Threshold value (scalar)
            segment_ends: Segment-end tensor (B,)
            t: Timestep tensor (B,)
            t_expand: Expanded timestep tensor (B, T, Da)
            reduce_op: Reduction operation for tensors, such as torch.mean
            trajectory: Trajectory tensor (B, T, Da)
            delta: Timestep size (scalar)
        Returns:
            Computed result tensor (B,)
        """
        if (threshold, int) and threshold == 0:
            return 0

        less_than_threshold = t_expand < threshold
    
        far_from_segment_ends = (segment_ends - t) > 1.01 * delta
        far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
    
        losses_v = torch.square(vt - vr)
        losses_v = less_than_threshold * far_from_segment_ends * losses_v
        losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
    
        return losses_v
    
    def _frequency_loss(self, tar_freq_actions, pre_freq_actions):
        """
        Frequency-domain loss function.
        
        Formula:
        L_freq = mean(Σ_t (F_target(b,t,d) - F_pred(b,t,d))²)
        
        Args:
            tar_freq_actions: Frequency-domain representation of target actions (B, T, Da)
            pre_freq_actions: Frequency-domain representation of predicted actions (B, T, Da)
            
        Returns:
            loss: Scalar loss value
        """
        loss = torch.sum((tar_freq_actions - pre_freq_actions) ** 2, dim=(1,2))
        return loss
    
    def _adaptive_frequency_loss(self, tar_freq_actions, pre_freq_actions):
        """
        Adaptive frequency-domain loss function optimized with softmax.
        
        Formula:
        1. D²(b,t,d) = (F_target(b,t,d) - F_pred(b,t,d))²
        2. w(b,t,d) = softmax(D²(b,t,d)) along time dimension
        3. L_freq = mean(Σ_t D²(b,t,d) * w(b,t,d))
        
        Args:
            tar_freq_actions: Frequency-domain representation of target actions (B, T, Da)
            pre_freq_actions: Frequency-domain representation of predicted actions (B, T, Da)
            
        Returns:
            loss: Scalar loss value
        """
        # Compute the squared difference D²(b,t,d).
        diff_square = torch.square(tar_freq_actions - pre_freq_actions)  # (B, T, Da)
        
        # Compute adaptive weights with softmax along the temporal dimension.
        # Softmax handles numerical stability automatically, so no manual epsilon is needed.
        weights = torch.softmax(diff_square, dim=1)  # (B, T, Da)
        weights = weights.detach()  # Stop gradient propagation through the weights.
        # Compute the weighted loss, then sum directly and average.
        weighted_loss = torch.sum(diff_square * weights, dim=1)  # (B, Da)
        loss = torch.mean(weighted_loss)  # Scalar
        
        return loss
