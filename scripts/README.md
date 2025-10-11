# Scripts Organization

This directory contains all the training scripts and utilities organized by functionality.

## Directory Structure

### `centralized/`
Contains centralized training scripts:
- `central_multimodal_gb.py` - Centralized multimodal training with gradient blending
- `central_multimodal_no_gb.py` - Centralized multimodal training without gradient blending
- `training_gb_centralized.py` - Training utilities for centralized gradient blending

### `federated/`
Contains federated learning training scripts:
- `fed_train_proposed_gb.py` - Federated training with gradient blending
- `fed_train_proposed_gb_pcw.py` - Federated training with gradient blending and PCW
- `fed_train_proposed_refactored.py` - Refactored federated training script

### `utils/`
Contains utility scripts and argument parsers:
- `dataset_utils.py` - Dataset utility functions
- `fed_arg_parser.py` - Argument parser for federated learning
- `fed_utils.py` - Federated learning utility functions
- `mrna_arg_parser.py` - Argument parser for mRNA training
- `set_up_save_dir.py` - Directory setup utilities
- `setup_utils.py` - General setup utilities

### `models/`
Contains model definitions and initializers:
- `models.py` - Model definitions
- `model_initializer.py` - Model initialization scripts

## Usage

To run scripts from their new locations, use:

```bash
# Centralized training
python scripts/centralized/central_multimodal_gb.py --data_path ../data --result_path ../results

# Federated training
python scripts/federated/fed_train_proposed_gb.py --data_path ../data --result_path ../results

# Model initialization
python scripts/models/model_initializer.py
```

## Import Updates

All scripts have been updated with proper import paths that include the root directory in the Python path, ensuring compatibility with the new structure.
