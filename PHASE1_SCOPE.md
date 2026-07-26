# Phase 1: Model Architecture & Training Pipeline

## Goal
Design and implement a BNN training pipeline that achieves >95% accuracy on MNIST, with automated weight export for FPGA deployment.

## Scope
- BNN model architecture design (layers, neurons, binarization strategy)
- PyTorch training pipeline with data augmentation
- Binarization-aware training (straight-through estimator)
- Quantization & weight export to hardware-compatible format
- Evaluation & validation pipeline
- Tests for model, training, and export

## Out of Scope
- Hardware RTL implementation (Phase 2)
- Vivado project generation (Phase 3)
- FPGA synthesis/implementation (Phase 3)

## Acceptance Criteria
- [ ] BNN model achieves >95% test accuracy on MNIST
- [ ] Training pipeline runs end-to-end with configurable hyperparameters
- [ ] Binarization-aware training using straight-through estimator
- [ ] Weight export produces hardware-compatible mem_files format
- [ ] Exported weights match software inference exactly
- [ ] Unit tests for model, training, export
- [ ] Configuration-driven (no hardcoded paths/values)