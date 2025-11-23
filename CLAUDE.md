# CLAUDE.md - AI Assistant Guide for Diffusion-DICE

## Project Overview

**Diffusion-DICE** is a PyTorch implementation of "Diffusion-DICE: In-Sample Diffusion Guidance for Offline Reinforcement Learning" (NeurIPS 2024).

- **Paper**: [Diffusion-DICE: In-Sample Diffusion Guidance for Offline Reinforcement Learning](https://arxiv.org/pdf/2407.20109)
- **Authors**: Liyuan Mao*, Haoran Xu*, Weinan Zhang†, Xianyuan Zhan, Amy Zhang†
- **License**: MIT
- **Primary Language**: Python (PyTorch)

### Core Concept
This project implements an offline RL method that combines diffusion models with DICE (Distributional Correction Estimation) for policy learning from fixed datasets, evaluated on D4RL benchmarks.

## Repository Structure

```
Diffusion_DICE/
├── diffusion_DICE/              # Core diffusion model components
│   ├── __init__.py
│   ├── model.py                 # ScoreNet, critic networks, guidance
│   ├── loss.py                  # Score-based loss function
│   ├── schedule.py              # Noise scheduling (marginal_prob_std)
│   └── dpm_solver_pytorch.py    # DPM solver for sampling
├── configs/                     # YAML configuration files
│   ├── mujoco.yaml             # MuJoCo locomotion settings
│   └── antmaze.yaml            # AntMaze navigation settings
├── main_diffusion_DICE.py      # Main training script for Diffusion-DICE
├── pretrain_behavior.py        # Behavior policy pretraining
├── pretrain_behavior.sh        # Batch pretraining script
├── dataset.py                  # D4RL dataset loader
├── utils.py                    # Argument parsing and evaluation
├── Readme.md                   # User-facing documentation
└── .gitignore                  # Git ignore rules

Generated during training:
├── models_rl/                  # Saved model checkpoints
├── logs/                       # Training logs
└── wandb/                      # Weights & Biases logging
```

## Architecture Components

### 1. Diffusion Model (`diffusion_DICE/model.py`)

#### Key Classes:

**ScoreNet (ScoreBase)**
- Main diffusion model for action generation
- Architecture: U-Net style with residual blocks
- Conditioning: State-conditioned via `self.condition`
- Forward pass: Returns score function normalized by noise std
- Sampling: DPM-Solver for efficient inference
- Key attributes:
  - `condition`: Current state conditioning
  - `v[0]`: Guidance critic network (SDICE_Critic)
  - `inference_sample`: Number of samples to generate per state
  - `output_dim`: Action dimension

**SDICE_Critic (Critic_Guide)**
- Implements SDICE value/Q-function updates
- Components:
  - `q0`: Twin Q-network with target network
  - `v0`: Value function
  - `q_ensemble`: Optional ensemble Q for uncertainty estimation
  - `wt`: Guidance weight network (energy function)
- Key methods:
  - `update_v0(data)`: Updates V and Q networks using SDICE loss
  - `update_wt(data)`: Updates guidance weights via dual loss
  - `calculate_guidance(a, t, condition)`: Computes gradient of wt for guidance

**Network Architectures**
- `TwinQ`: Twin Q-networks for stability
- `ValueFunction`: Simple V-network
- `GuidanceWt`: Time and action-conditioned guidance
- `VectorizedCritic/VectorizedQ`: Ensemble networks for uncertainty

### 2. Training Pipeline

#### Two-Stage Process:

**Stage 1: Behavior Pretraining** (`pretrain_behavior.py`)
- Train diffusion model to mimic dataset behavior
- 600 epochs, batch size 4096
- Loss: Score matching loss (denoising score matching)
- Saves checkpoints every 10 epochs
- Command: `bash pretrain_behavior.sh`

**Stage 2: Diffusion-DICE Training** (`main_diffusion_DICE.py`)
- Load pretrained behavior model
- Train guidance network (SDICE critic)
- 10,000 mini-batches per epoch
- Evaluation every N epochs with multiprocessing
- Command: See "Running Experiments" below

### 3. Data Flow

```
D4RL Dataset → D4RL_dataset class → DataLoader
                                        ↓
                              ┌─────────┴─────────┐
                              ↓                   ↓
                    Behavior Training    SDICE Critic Training
                         (Stage 1)            (Stage 2)
                              ↓                   ↓
                    ScoreNet weights ─→  Load + train v0/wt
                                                  ↓
                                        Guided Action Sampling
                                                  ↓
                                        Policy Evaluation (D4RL)
```

### 4. Configuration System

Configurations are split by environment type:

**configs/mujoco.yaml** - For locomotion tasks
- Guidance scales: `s: {medium: 4.0, medium-replay: 4.0, medium-expert: 2.0}`
- Q ensemble: 10 networks
- Support actions M: 16
- Filter type: max

**configs/antmaze.yaml** - For navigation tasks
- Guidance scales: `s: {umaze: 1.0, medium: 2.0, large: 4.0}`
- Q ensemble: 0 (disabled)
- Support actions M: 32
- Filter type: softmax
- More evaluation seeds: 50 vs 10

**Common parameters:**
- `diffusion_steps: 15` - Inference sampling steps
- `hidden_dim: 256` - Network hidden dimension
- `value_lr: 3e-4` - Learning rate for value networks
- `wt_lr: 3e-4` - Learning rate for guidance network

## Key Implementation Details

### Reward Tuning (`dataset.py:42-58`)
Different reward preprocessing based on environment:
- **AntMaze**: `reward = torch.where(reward > 0, 0.0, -1.0)` (IQL style)
- **Locomotion**: Normalize by return range, scale by 1000

### Multiprocessing Evaluation (`main_diffusion_DICE.py:18-41`)
- Evaluation runs in subprocess to avoid memory issues
- Temporary model saved with unique timestamp key
- Results passed via `torch.multiprocessing.Queue`
- Old processes joined before starting new ones

### Action Selection (`diffusion_DICE/model.py:455-482`)
1. Generate `inference_sample` actions per state via DPM-Solver
2. Filter by Q-value (max or softmax)
3. Uses ensemble Q if `q_ensemble_num > 0`, else twin Q
4. Returns highest-value action

### SDICE Loss Components

**V-function loss** (`model.py:312-333`):
```
residual_loss = where(sp_term >= 0, sp_term^2/4 + sp_term, exp(sp_term) - 1)
value_loss = mean(residual_loss + v/alpha)
```
- Piecewise f-divergence for stability
- `sp_term = (Q(s,a) - V(s)) / alpha`

**Guidance weight loss** (`model.py:370-420`):
- Perturb actions with diffusion noise
- Compute dual loss with importance weights
- Clip normalization to avoid overflow

## Development Workflows

### Running Experiments

**1. Pretrain Behavior Policy**
```bash
bash pretrain_behavior.sh
# Or for single environment:
python pretrain_behavior.py --env hopper-medium-v2 --seed 0
```

**2. Train Diffusion-DICE**
```bash
python main_diffusion_DICE.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --actor_load_path ./models_rl/hopper-medium-v2/behavior_ckpt600_seed0.pth \
  --inference_sample 64 \
  --alpha 0.5 \
  --batch_size 512
```

**3. Optional Arguments**
- `--use_lr_schedule 1`: Enable cosine annealing LR
- `--min_value_lr 1e-4`: Minimum LR for scheduler
- `--device_num 0`: GPU device ID
- `--weight_decay 1e-4`: AdamW weight decay

### Model Checkpoints

**Saved Files:**
- Behavior: `./models_rl/{env}/behavior_ckpt{epoch}_seed{seed}.pth`
- Critic: `./models_rl/{env}/critic_ckpt{epoch}_seed{seed}.pth`
- Temporary evaluation: `critic_ckpt_temp_{timestamp}.pth` (auto-deleted)

**Loading:**
```python
# Load behavior model
ckpt = torch.load(args.actor_load_path, map_location=args.device)
score_model.load_state_dict(ckpt)

# Load critic for inference
v_ckpt = torch.load(v_path, map_location=args.device)
score_model.v[0].load_state_dict(v_ckpt)
```

## Coding Conventions

### Style Guidelines

1. **Imports**: Standard library → Third-party → Local modules
2. **Tensor operations**: Use `.to(args.device)` for device placement
3. **Random seeds**: Set consistently (torch, numpy, gym env)
4. **Network initialization**: Custom init for stability (see VectorizedCritic)
5. **Gradient management**: Use `torch.no_grad()` for inference/target networks

### Naming Conventions

- **Variables**:
  - `s, a, r, s_, d` - state, action, reward, next state, done
  - `v0` - value at time 0 (initial state value)
  - `q0` - Q-value at time 0
  - `wt` - guidance weight at time t
- **Dimensions**:
  - `sdim, adim` - state/action dimensions
  - `bz` - batch size
  - `M` - number of support actions for SDICE

### Common Patterns

**Condition handling in ScoreNet:**
```python
# Set condition before sampling
score_model.condition = states
results = score_model.dpm_wrapper_sample(...)
score_model.condition = None  # Always reset
```

**Target network updates:**
```python
update_target(new_network, target_network, tau=0.005)
```

**W&B logging:**
```python
wandb.log({"metric_name": value}, step=step_count)
```

## Dependencies

### Required Packages
- **PyTorch**: Deep learning framework
- **Gym**: RL environment interface
- **D4RL**: Offline RL datasets
- **MuJoCo**: Physics simulator (license required)
- **Weights & Biases (wandb)**: Experiment tracking
- **NumPy, SciPy**: Numerical operations
- **scikit-learn**: Data utilities
- **PyYAML**: Configuration parsing

### Environment Setup
```bash
pip install torch gym d4rl wandb numpy scipy scikit-learn pyyaml
# Install MuJoCo separately (see https://www.roboti.us/download.html)
```

## Important Notes for AI Assistants

### When Modifying Code

1. **Device Consistency**: Always ensure tensors are on correct device
   - Check `args.device` usage
   - Use `.to(args.device)` when creating new tensors

2. **Multiprocessing**:
   - `torch.multiprocessing.set_start_method('spawn')` is required
   - Subprocess evaluations avoid memory leaks
   - Always clean up temporary files

3. **Gradient Flow**:
   - Score model training: gradients through score network only
   - Guidance: detach gradients when computing guidance signal
   - Target networks: `.requires_grad_(False)`

4. **Wandb Integration**:
   - Project names: `diffusion_DICE_{env}` or `diffusion_DICE_behavior`
   - Must call `wandb.init()` before logging
   - Step counts matter for proper visualization

5. **Config Loading**:
   - Configs auto-loaded based on env name (antmaze vs mujoco)
   - Command-line args override config values
   - Use `setattr(args, key, value)` pattern

### Common Debugging Points

1. **Batch dimension mismatches**: Check if tensors need `.unsqueeze(0)`
2. **Condition not set**: `score_model.condition` must be set before DPM sampling
3. **NaN losses**: Usually from exp overflow - check clipping in SDICE loss
4. **OOM errors**: Reduce `batch_size` or `inference_sample`
5. **Slow evaluation**: Ensure multiprocessing is working correctly

### Testing Suggestions

1. **Quick test run**: Use small `train_epoch` and `evaluation_interval`
2. **Sanity check**: Behavior model should achieve reasonable normalized scores
3. **Ablation studies**: Vary `alpha`, `guidance_scale`, `inference_sample`
4. **Environment check**: Verify D4RL installation with `gym.make(env_name)`

## File Modification Guidelines

### High-Risk Files (Modify with Caution)
- `diffusion_DICE/dpm_solver_pytorch.py` - Complex sampling logic
- `diffusion_DICE/model.py` - Core architecture, many dependencies
- `diffusion_DICE/loss.py` - Fundamental training objective

### Safe to Modify
- `configs/*.yaml` - Hyperparameter tuning
- `pretrain_behavior.sh` - Environment/seed lists
- `utils.py` - Argument parsing, evaluation logic
- `dataset.py` - Reward tuning, data preprocessing

### Typical Modifications

**Add new environment:**
1. Add to `pretrain_behavior.sh` env_list
2. Verify config (mujoco vs antmaze)
3. May need reward tuning in `dataset.py`

**Tune hyperparameters:**
1. Edit appropriate config YAML
2. Or pass via command line args
3. Document changes in experiment logs

**Change network architecture:**
1. Modify `ScoreNet` or `SDICE_Critic` in `model.py`
2. Ensure input/output dims match
3. Re-pretrain from scratch

## Git Workflow

**Current Branch**: `claude/claude-md-micav5tt8dxq8ojm-01XwFuogwJuxdersw4i65U3J`

### Branch Naming Convention
- Feature branches: `claude/claude-md-{identifier}-{session_id}`
- Main branch: (not specified in current repo)

### Commit Guidelines
1. Clear, descriptive messages
2. Reference issue/task if applicable
3. Separate logical changes into different commits

### .gitignore
Excludes:
- `models_rl/` - Model checkpoints (too large)
- `logs/` - Training logs
- `wandb/` - Experiment tracking data
- `__pycache__/` - Python bytecode
- `.vscodevenv/` - Virtual environment

## Quick Reference

### Supported Environments

**MuJoCo Locomotion:**
- hopper-{medium, medium-replay, medium-expert}-v2
- walker2d-{medium, medium-replay, medium-expert}-v2
- halfcheetah-{medium, medium-replay, medium-expert}-v2

**AntMaze Navigation:**
- antmaze-{umaze, umaze-diverse}-v2
- antmaze-{medium-play, medium-diverse}-v2
- antmaze-{large-play, large-diverse}-v2

### Key Hyperparameters by Task

| Parameter | MuJoCo | AntMaze |
|-----------|--------|---------|
| M (support actions) | 16 | 32 |
| Q ensemble size | 10 | 0 |
| Filter type | max | softmax |
| Guidance scale | 2-4 | 1-4 |
| Eval seeds | 10 | 50 |

### Typical Training Time
- Behavior pretraining: ~600 epochs
- Diffusion-DICE: ~100 epochs × 10k batches
- Evaluation: Every 4 epochs (configurable)

## Additional Resources

- **Paper**: https://arxiv.org/pdf/2407.20109
- **D4RL**: https://github.com/Farama-Foundation/D4RL
- **DPM-Solver**: Referenced in code for efficient sampling
- **Original code reference**: Based on https://github.com/ChenDRAG/CEP-energy-guided-diffusion

---

**Last Updated**: 2025-11-23
**Maintainer**: AI Assistant Analysis
**Status**: Active Research Code
