# MNIST_BNNv2 Complete Rebuild — Project Roadmap

## Phase 0: Repository Restructure & Setup
- [ ] Clean repository structure
- [ ] Create `src/`, `scripts/`, `docs/`, `tests/`, `hw/`, `sw/` directories
- [ ] Set up Python virtual environment requirements
- [ ] Create base configuration system

## Phase 1: Dataset Analysis & Preprocessing Pipeline
- [ ] Analyze MNIST dataset statistics
- [ ] Design preprocessing pipeline (normalization, binarization, augmentation)
- [ ] Create train/val/test splits with stratification
- [ ] Implement dataset loading utilities
- [ ] Verify data integrity

## Phase 2: Model Architecture & Training Pipeline
- [ ] Research BNN architectures for MNIST
- [ ] Design optimal architecture (layers, neurons, connectivity)
- [ ] Implement training pipeline with PyTorch
- [ ] Implement binarization strategy (weights/activations)
- [ ] Implement validation & evaluation metrics
- [ ] Achieve >95% test accuracy

## Phase 3: Quantization & Parameter Export
- [ ] Design quantization scheme for FPGA
- [ ] Export weights, thresholds, biases in hardware-friendly format
- [ ] Create automated export pipeline
- [ ] Verify exported parameters match software inference

## Phase 4: FPGA Hardware Architecture
- [ ] Design optimal hardware architecture
- [ ] Evaluate sequential vs parallel implementations
- [ ] Design memory hierarchy (BRAM, LUT RAM)
- [ ] Implement SystemVerilog modules
- [ ] Optimize for timing closure

## Phase 5: Simulation & Verification
- [ ] Create Icarus Verilog compatible testbench
- [ ] Implement automated test suite
- [ ] Verify all 10 digits
- [ ] Regression testing framework

## Phase 6: Vivado Integration & FPGA Implementation
- [ ] Create Vivado project structure
- [ ] Add constraints (timing, pin mapping)
- [ ] Synthesize, implement, verify timing
- [ ] Generate bitstream

## Phase 7: Documentation & CI
- [ ] Comprehensive documentation
- [ ] Automated build scripts
- [ ] CI pipeline (training -> export -> sim -> synth)
- [ ] Retraining workflow documentation

## Phase 8: Final Validation
- [ ] Test with `my_digit.png` unseen digits
- [ ] Measure accuracy on held-out test set
- [ ] Performance analysis
- [ ] Production readiness review