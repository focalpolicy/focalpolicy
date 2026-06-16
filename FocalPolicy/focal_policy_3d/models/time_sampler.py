import torch


def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time


def sample_two_timesteps(cfg, num_samples: int, device: torch.device):
    if cfg["tr_sampler"] == "v0":
        t, r = sample_two_timesteps_t_r_v0(cfg, num_samples, device=device)
        return t, r
    elif cfg["tr_sampler"] == "v1":
        t, r = sample_two_timesteps_t_r_v1(cfg, num_samples, device=device)
        return t, r
    else:
        raise ValueError(f"Unknown joint time sampler: {cfg['tr_sampler']}")

def sample_two_timesteps_t_r_v0(cfg, num_samples: int, device: torch.device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 0: used in paper.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(cfg["P_mean_t"], cfg["P_std_t"], num_samples, device=device)
    r = logit_normal_timestep_sample(cfg["P_mean_r"], cfg["P_std_r"], num_samples, device=device)

    # step 2: ensure t >= r
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # step 3: make t and r different with a probability of cfg["ratio"]
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - cfg["ratio"]
    r = torch.where(mask, t, r)

    return t, r


def sample_two_timesteps_t_r_v1(cfg, num_samples: int, device: torch.device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 1: different post-processing to ensure t >= r.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(cfg["P_mean_t"], cfg["P_std_t"], num_samples, device=device)
    r = logit_normal_timestep_sample(cfg["P_mean_r"], cfg["P_std_r"], num_samples, device=device)

    # step 2: make t and r different with a probability of cfg["ratio"]
    prob = torch.rand(num_samples, device=device)
    mask = prob < 1 - cfg["ratio"]
    r = torch.where(mask, t, r)

    # step 3: ensure t >= r
    r = torch.minimum(t, r)

    return t, r  

if __name__ == "__main__":
    # test
    cfg = {
        "tr_sampler": "v0",
        "P_mean_t": -0.6,
        "P_std_t": 1.6,
        "P_mean_r": -4.0,
        "P_std_r": 1.6,
        "ratio": 0.75
    }
    t, r = sample_two_timesteps(cfg, num_samples=128, device=torch.device("cpu"))
    # r = torch.distributions.Beta(torch.tensor(8.0), torch.tensor(1.0)).sample([r.shape[0]])
    r = 1 - r
    print(t)
    print(r)
    # print((t >= r.float()).float().mean())


