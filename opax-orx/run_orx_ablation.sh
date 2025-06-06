#!/bin/bash

# ORX Ablation Study
# Compares ORX, OpAx, and Random strategies across different noise levels

# Set WandB configuration
export WANDB_ENTITY=bbullinger
export WANDB_PROJECT=ORX-Ablation-Long-Single-Seed

# Parameters optimized for ~2-hour runtime with 20 data points per curve
DEFAULT_TOTAL_TRAIN_STEPS=3000       # Total environment interactions
DEFAULT_MAX_TRAIN_STEPS=250           # Max gradient updates per agent.train_step() call
DEFAULT_ROLLOUT_STEPS=50             # Env. interactions per learning step
DEFAULT_TRAIN_STEPS=250               # Target updates within one agent.train_step()
DEFAULT_TIME_LIMIT=200                # Episode length during training rollouts
DEFAULT_TIME_LIMIT_EVAL=200           # Episode length during evaluation
DEFAULT_BATCH_SIZE=128                # Batch size for training
DEFAULT_NUM_ENSEMBLES=3              # Fewer ensembles
DEFAULT_EVAL_FREQ=1                  # Evaluate every step (maximum data points)
DEFAULT_EVAL_EPISODES=1              # Single episode per evaluation
DEFAULT_TRAIN_FREQ=1                 # Train every step
DEFAULT_NUM_EPOCHS=-1                # Use train_steps instead of epochs
DEFAULT_NORMALIZE=true               # Normalize observations
DEFAULT_ACTION_NORMALIZE=true        # Normalize actions  
DEFAULT_VALIDATE=true                # Enable validation
DEFAULT_BUFFER_SIZE=1000000          # Replay buffer size
DEFAULT_EXPLORATION_STEPS=0          # No initial random exploration
DEFAULT_VALIDATION_BUFFER_SIZE=100000
DEFAULT_VALIDATION_BATCH_SIZE=4096
DEFAULT_ACTION_COST=0.0              # No action penalty

NOISE_LEVELS=(1.0)
SEEDS=(834 835 836)

echo "========================================================"
echo "ORX Full Ablation Study"
echo "Comparing ORX, OpAx, and Random strategies (noise_level=${NOISE_LEVELS[0]})"
echo "Seeds: ${SEEDS[@]}"
echo "========================================================"

# Function to run experiment
run_experiment() {
    local strategy=$1
    local noise_level=$2
    local seed=$3
    local run_name="${strategy}-noise${noise_level}-seed${seed}"
    
    echo "Running: $strategy with noise_level=$noise_level, seed=$seed"
    
    # Set WandB run name via environment variable
    export WANDB_NAME="$run_name"
    
    python experiments/pendulum_exp/active_exploration_exp_pendulum.py \
        --exp_name "$WANDB_PROJECT" \
        --use_wandb \
        --exploration_strategy "$strategy" \
        --use_log --use_al \
        --total_train_steps $DEFAULT_TOTAL_TRAIN_STEPS \
        --max_train_steps $DEFAULT_MAX_TRAIN_STEPS \
        --rollout_steps $DEFAULT_ROLLOUT_STEPS \
        --train_steps $DEFAULT_TRAIN_STEPS \
        --time_limit $DEFAULT_TIME_LIMIT \
        --time_limit_eval $DEFAULT_TIME_LIMIT_EVAL \
        --batch_size $DEFAULT_BATCH_SIZE \
        --num_ensembles $DEFAULT_NUM_ENSEMBLES \
        --eval_freq $DEFAULT_EVAL_FREQ \
        --eval_episodes $DEFAULT_EVAL_EPISODES \
        --train_freq $DEFAULT_TRAIN_FREQ \
        --num_epochs $DEFAULT_NUM_EPOCHS \
        --normalize --action_normalize \
        --validate \
        --buffer_size $DEFAULT_BUFFER_SIZE \
        --exploration_steps $DEFAULT_EXPLORATION_STEPS \
        --validation_buffer_size $DEFAULT_VALIDATION_BUFFER_SIZE \
        --validation_batch_size $DEFAULT_VALIDATION_BATCH_SIZE \
        --action_cost $DEFAULT_ACTION_COST \
        --seed "$seed" \
        --noise_level "$noise_level"
    
    echo "Completed: $strategy with noise_level=$noise_level, seed=$seed"
    echo ""
}

echo "ORX: Optimistic Random eXploration (Optimized Policy, Random Eta)"
echo "--------------------------------------------------------"
for noise_level in "${NOISE_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "ORX" "$noise_level" "$seed"
    done
done

echo "OPAX: Intelligent Optimization (Optimized Policy, Optimized Eta)"
echo "--------------------------------------------------------"
for noise_level in "${NOISE_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "Optimistic" "$noise_level" "$seed"
    done
done

echo "RANDOM: Pure Random Actions (No Planning)"
echo "--------------------------------------------------------"
for noise_level in "${NOISE_LEVELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_experiment "Uniform" "$noise_level" "$seed"
    done
done

echo ""
echo "========================================================"
echo "ORX Full Ablation Study Completed!"
echo "View results at: https://wandb.ai/bbullinger/ORX-Ablation"
echo "========================================================" 