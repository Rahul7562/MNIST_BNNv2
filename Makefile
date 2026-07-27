.PHONY: all train export test clean

all: train export test

train:
	PYTHONPATH=. python scripts/train.py --config_model configs/model.yaml --config_train configs/training.yaml

export:
	PYTHONPATH=. python scripts/export.py --config_model configs/model.yaml --config_train configs/training.yaml

test:
	PYTHONPATH=. pytest tests/ -v

clean:
	rm -rf checkpoints/ mem_files/
	find . -type d -name __pycache__ -exec rm -r {} +
	rm -rf .pytest_cache/
