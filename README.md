# Spine segmentation using Deep Learning
Repository to track the progress for AI in Medical Imaging Diagnostics project about spine segmentation.

## Getting Started

### Prerequisites
- Python 3.9.21 (recommended)

### Setup

1. Create a virtual environment:
```bash
python3.9 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets from the sources listed below and place them in the appropriate directories as specified in `config.json` or modify config itself.

4. Run the training pipeline:
```bash
./cli.py
```

If configured properly, `cli.py` will:
- Preprocess the datasets
- Create dataloaders with train/val/test splits
- Start training the model

You can specify a custom config file using `--config`:
```bash
./cli.py --config my_config.json
```

It is recommended to place new configs for separate experiments in `experiments/` directory in appropriately named subdirectory.

## Preprocessing
Preprocessing is executed automatically if `preprocessed_data_dir` in `Config` do not exist and data sources in `PreprocessingConfig` are available. So if you want to create another version of preprocessing modify preprocessing code and run `cli.py` with different `preprocessed_data_dir`.

## Project Structure

The project follows a modular structure with components in the `src/` directory:
- `config.py`: Configuration management using dataclasses
- `preprocessing.py`: Data preprocessing pipeline
- `dataloader.py`: PyTorch dataloaders with data augmentation
- `model.py`: Model factory and architecture definitions
- `train.py`: Training loop and validation
- `loss.py`, `metrics.py`, `optimizer.py`, `scheduler.py`: Training components

### Dependency Injection

The project uses a single `Config` object (from `src.config`) that is dependency-injected throughout the codebase. This `Config` object contains all hyperparameters, paths, and settings organized into nested dataclasses:
- `TrainingConfig`: Training hyperparameters (learning rate, epochs, scheduler, etc.)
- `PreprocessingConfig`: Data preprocessing settings
- `DataLoaderConfig`: Dataloader configuration and augmentation parameters
- `ModelConfig`: Model architecture parameters
- `WandBConfig`: Weights & Biases logging configuration

All components receive the same `Config` instance, ensuring consistency and making it easy to manage experiments through JSON configuration files.

## Additional modules
After change of `Config` structure run `python -m src.config` in order to regenerate `default_config.json`.
After changes made to `dataloader.py` you can run `python -m src.dataloader` in order to test it.
Similarly for `preprocessing.py`

## Hyperparameter Sweeps

For hyperparameter optimization using Weights & Biases, see [WANDB_SWEEP_README.md](WANDB_SWEEP_README.md).

## EDA
EDA was done in `EDA/datasets_eda.ipynb`.

Datasets' sources:
- [SPIDER](https://zenodo.org/records/10159290)
- [Spine output (dataset given by tutor)](https://drive.google.com/file/d/18BqYjXgPiITNh4vhtzIw6PyFvUZbqCyk/view)
- [(OSF) Multi-scanner and multi-modal lumbar vertebral body and intervertebral disc segmentation database](https://www.nature.com/articles/s41597-022-01222-8)
