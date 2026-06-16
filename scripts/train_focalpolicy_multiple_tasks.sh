#!/bin/bash
################### Metaworld tasks ######################
### medium tasks 
tasks=("metaworld_basketball" "metaworld_bin-picking" "metaworld_box-close" "metaworld_coffee-pull" \
       "metaworld_coffee-push" "metaworld_hammer" "metaworld_peg-insert-side" "metaworld_push-wall" \
       "metaworld_soccer" "metaworld_sweep" "metaworld_sweep-into")
### hard tasks 
# tasks=("metaworld_assembly" "metaworld_hand-insert" "metaworld_pick-out-of-hole"\
#        "metaworld_pick-place" "metaworld_push")
### very hard tasks 
# tasks=("metaworld_shelf-place" "metaworld_disassemble" "metaworld_stick-pull" \
#        "metaworld_stick-push" "metaworld_pick-place-wall")
#################### Adroit ######################
# tasks=("adroit_door" "adroit_hammer" "adroit_pen" )
#######################################################################################
algorithm="focalpolicy"
extra_params="metaworld"  
random_seeds=("0" "1" "2")  
gpu_id="0"  
for task in "${tasks[@]}"; do
  echo "Starting training for task: $task"
  for seed in "${random_seeds[@]}"; do
    echo "Training task: $task with seed: $seed"
    bash scripts/train_focalpolicy.sh $algorithm $task $extra_params $seed $gpu_id
    echo "$task with seed $seed training completed."
  done
  echo "All seeds for task $task completed. Moving to the next task."
  echo "----------------------------------------"
done

echo "All tasks have been trained with all seeds."
