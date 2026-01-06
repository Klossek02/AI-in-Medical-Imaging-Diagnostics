# WandB Sweep Usage Guide

Idea: You create a sweep_config.yaml file, then create and run a sweep of hyperparameters. At the end you get nicely formatted web-based interface will all the plots you need.

Examplar single run command executed by agent:
```bash
python wandb_sweep.py --base_c=16 --config=config.json --contrast_prob=0.3 --distorsion_distort_limit=0.05 --distorsion_num_cells=5 --distorsion_prob=0.3 --learning_rate=0.00010363370629152814 --noise_prob=0.5 --noise_std=0.3 --num_epochs=5 --weight_decay=0.0005791920758536536python wandb_sweep.py --config config.yaml --project my-project --name my-run
```
So it is important to specify base config as `config` in `sweep_config.yaml`.
List of sweepable parameters is contained in `wandb_sweep.py` script.

## Installation

First, follow installation guidelines from README.md 
Then login to wandb:
```bash
wandb login
```

## Quick Start

### 1. Initialize a Sweep

```bash
wandb sweep sweep_config.yaml
```
This will output a sweep ID like: `your-entity/spine-segmentation/abc123`

### 2. Run an Agent

```bash
wandb agent <sweep-id>
```

For example:
```bash
wandb agent your-entity/spine-segmentation/abc123
```

### 3. Run Multiple Agents (Parallel)

You can run multiple agents in parallel to speed up the sweep:

```bash
# Terminal 1
wandb agent <sweep-id>

# Terminal 2
wandb agent <sweep-id>

# Terminal 3
wandb agent <sweep-id>
```

