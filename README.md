# Deep Reinforcement Learning Heuristic (DRLH) for PDPTW

This project implements a Deep Reinforcement Learning approach to solve the Pickup and Delivery Problem with Time Windows (PDPTW) using Proximal Policy Optimization (PPO) with specialized heuristic operators.

## Overview

The system combines classical PDPTW heuristics with deep reinforcement learning to intelligently select and sequence optimization operators. The RL agent learns to choose the most effective heuristic operations at each step of the optimization process.

### Key Components

- **RL Agent**: PPO-based agent with Actor-Critic architecture
- **Environment**: PDPTW problem formulated as a Gym environment  
- **Heuristics**: Specialized remove and insert operators for PDPTW constraints
- **State Representation**: Multiple encoding schemes for solution states
- **Reward Functions**: Various reward strategies to guide learning
- **Time Windows**: Full support for pickup and delivery time constraints

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd drlh_cvrp

# Create conda environment (recommended)
conda create -n drlh python=3.11
conda activate drlh

# Install dependencies
pip install torch numpy gym tensorboard
pip install "numpy<1.25"  # For compatibility

# Optional: Install CUDA-enabled PyTorch for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Train a New Agent

```bash
# Basic training on PDPTW dataset (requires dataset path)
python main.py --mode train --dataset data/pdptw/pdptw-training-96

# Training with time limit (useful for development)
python main.py --mode train --dataset data/pdptw/pdptw-training-96 --time-limit 30

# Full training with custom parameters
python main.py --mode train \
               --dataset data/pdptw/pdptw-training-192 \
               --episodes 1000 \
               --learning-rate 0.0003 \
               --logdir logs/pdptw_experiment
```

### 2. Test a Trained Agent

```bash
# Test on PDPTW test dataset
python main.py --mode test \
               --model-path logs/pdptw_experiment/models/final_model.pt \
               --dataset data/pdptw/pdptw-test-6

# Test with detailed logging
python main.py --mode test \
               --model-path logs/pdptw_experiment/models/final_model.pt \
               --dataset data/pdptw/pdptw-test-6 \
               --verbose \
               --logdir logs/test_results
```

### 3. Solve Specific Instances

```bash
# Solve a limited number of instances
python main.py --mode solve \
               --dataset data/pdptw/pdptw-test-6 \
               --instances 5

# Quick solve with trained model
python main.py --mode solve \
               --dataset data/pdptw/pdptw-test-6 \
               --instances 10 \
               --verbose
```

## Command-Line Arguments

### Core Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--mode` | choice | ✅ | Mode: `train`, `test`, or `solve` |
| `--dataset` | string | ✅ | Path to directory containing PDPTW instance files (.txt) |

### Training Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 100 | Number of training episodes |
| `--instances` | int | None | Maximum number of instances to use (default: all) |
| `--time-limit` | float | None | Time limit for training in minutes (default: no limit) |

### Model Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | string | - | Path to saved model (required for test mode) |
| `--learning-rate` | float | 0.0003 | Learning rate for optimizer |
| `--gamma` | float | 0.99 | Discount factor for rewards |
| `--batch-size` | int | 64 | Batch size for training |
| `--epochs` | int | 10 | Number of epochs per training step |

### PDPTW Specific Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--T-f` | float | 0.05 | Final temperature for simulated annealing |
| `--state-rep` | string | (see below) | State representation encoding scheme |
| `--reward-func` | string | "5310" | Reward function configuration |

### System Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--logdir` | string | auto | Directory for logs and model outputs |
| `--verbose` | flag | False | Enable detailed output during execution |
| `--no-cuda` | flag | False | Disable CUDA even if available |
| `--save-model` | flag | True | Save model checkpoints during training |
| `--save-interval` | int | 100 | Episodes between model saves |

### Complete Usage Examples

```bash
# Development training with time limit
python main.py --mode train \
               --dataset data/pdptw/pdptw-training-96 \
               --time-limit 15 \
               --verbose \
               --logdir logs/dev_experiment

# Production training
python main.py --mode train \
               --dataset data/pdptw/pdptw-training-192 \
               --episodes 2000 \
               --learning-rate 0.0001 \
               --batch-size 128 \
               --save-interval 50 \
               --logdir logs/production_run

# Model testing and evaluation
python main.py --mode test \
               --model-path logs/production_run/models/checkpoint_1000.pt \
               --dataset data/pdptw/pdptw-test-6 \
               --verbose \
               --logdir logs/evaluation

# Quick problem solving
python main.py --mode solve \
               --dataset data/pdptw/pdptw-test-6 \
               --instances 20 \
               --verbose
```

## Configuration Options

### State Representations

The default state representation is:
```
"reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen"
```

Available components:
- **reduced_dist**: Normalized distances
- **dist_from_min**: Distance from minimum found
- **no_improvement**: Steps since last improvement
- **index_step**: Current iteration index
- **was_changed**: Whether solution changed
- **unseen**: Exploration indicator

### Reward Functions

- **5310**: Balanced rewards (5=new best, 3=improvement, 1=diversification, 0=no change)
- **10310**: Higher reward for new best solutions
- **delta_change**: Proportional to actual improvement
- **new_best**: Reward only for finding new best solutions

### Normalization Methods

- **last_100k_normalize**: Running statistics from recent experiences (recommended)
- **max_normalize**: Scale by maximum observed values  
- **no_normalization**: Use raw state values

## Data Format

### PDPTW Instance Files (.txt)

PDPTW instances use a structured text format with sections for:
- Problem dimensions (nodes, calls, vehicles)
- Node coordinates and time windows
- Vehicle information (capacity, start locations)
- Call information (pickup/delivery pairs, demands, time windows)
- Distance and cost matrices
- Vehicle-call compatibility matrices

Example structure:
```
%n_nodes%
39

%n_calls%  
100

%n_vehicles%
50

%vehicles%
0 0 100000.0
1 1 100000.0
...

%calls%
1 2 10 0 0 1440 0 1440
3 4 15 0 0 1440 0 1440
...
```

## Output Files

Training and testing generate several output files in the log directory:

- **Result.txt**: Best costs found per instance
- **ACTION_SEQUENCE.txt**: Sequence of heuristic operators used
- **DIST_SEQUENCE.txt**: Cost progression during optimization
- **MIN_DIST_SEQUENCE.txt**: Best cost progression
- **TIME.txt**: Computation time per instance
- **BEST_SOL.txt**: Best solutions found (route assignments)
- **test_results_MODEL_DATASET_TIMESTAMP.txt**: Formatted summary table with results overview (test mode only)
- **models/**: Directory containing saved model checkpoints
  - **checkpoint_N.pt**: Model saved at episode N
  - **final_model.pt**: Final trained model

## Performance Examples

Recent training results on 100-call, 50-vehicle instances:

| Instance | Initial Cost | Final Cost | Improvement | Time |
|----------|--------------|------------|-------------|------|
| SHORTSEA_HE_2 | 56.8M | 15.9M | 72.0% | 77s |
| SHORTSEA_HE_3 | 52.1M | 14.6M | 72.1% | 77s |
| SHORTSEA_HE_4 | 52.3M | 16.5M | 68.4% | 77s |

**Average improvement: >70% cost reduction**

## Tips for Good Results

1. **Dataset Size**: Ensure consistent problem dimensions within training sets
2. **Time Limits**: Use `--time-limit` for development, remove for production
3. **Learning Rate**: Start with 0.0003, reduce if training is unstable
4. **Batch Size**: Increase for stable gradients, decrease for memory constraints
5. **Episodes**: 500-2000 episodes typically sufficient for good performance
6. **State Representation**: Default configuration works well for most cases

## Development Workflow

```bash
# 1. Quick algorithm verification (1-5 minutes)
python main.py --mode train --dataset data/pdptw/pdptw-training-96 --time-limit 5

# 2. Parameter tuning (15-30 minutes)  
python main.py --mode train --dataset data/pdptw/pdptw-training-96 --time-limit 30 --learning-rate 0.0001

# 3. Full training (hours)
python main.py --mode train --dataset data/pdptw/pdptw-training-192 --episodes 2000

# 4. Model evaluation
python main.py --mode test --model-path logs/experiment/models/final_model.pt --dataset data/pdptw/pdptw-test-6
```

## Troubleshooting

### Common Issues

1. **Dataset Required**: Unlike CVRP, PDPTW requires actual instance files (no random generation)
2. **TensorBoard Issues**: Disable logging with `logging=False` if compatibility issues occur
3. **Memory Issues**: Reduce batch size with `--batch-size 32` or disable CUDA
4. **Import Errors**: Ensure you're running from the project root directory

### Performance Tuning

- **Problem Size**: Group instances by size (nodes/calls/vehicles) for consistent training
- **Learning Rate**: Reduce if training is unstable, increase if too slow
- **Temperature**: Adjust `--T-f` based on problem difficulty
- **Time Limit**: Use for development iterations, remove for final training

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]