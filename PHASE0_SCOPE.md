# Phase 0: Repository Restructure & Foundation

## Goal
Restructure the repository into a clean, production-ready layout with proper Python environment, configuration system, and directory organization.

## Touches (Files/Modules Expected to Change)
- Repository root structure
- New directories: `src/`, `scripts/`, `docs/`, `tests/`, `hw/`, `sw/`, `configs/`
- Python environment setup (`requirements.txt`, `pyproject.toml`, `setup.py`)
- Configuration system (`configs/default.yaml`, `config/__init__.py`)
- Dataset loading utilities (`sw/dataset.py`)
- Basic build/run scripts (`scripts/`)

## Out of Scope
- Model training implementation (Phase 1+)
- Hardware RTL implementation (Phase 3+)
- Quantization/export pipeline (Phase 2+)
- Vivado project generation (Phase 4+)

## Acceptance Criteria
- [ ] Repository has clean, standard directory structure
- [ ] Python virtual environment can be created with `pip install -e .`
- [ ] Configuration system loads YAML configs with validation
- [ ] Dataset loading utilities can read MNIST IDX files
- [ ] Basic scripts work: `scripts/convert_image.py`, `scripts/verify_dataset.py`
- [ ] All existing functionality preserved (image_to_mem.py behavior)
- [ ] No hardcoded paths — everything configurable
- [ ] Code passes basic linting (black, ruff, mypy)