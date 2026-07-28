import os

import pytest

from config import get_config
from scripts.train import train


def test_config_load():
    cfg = get_config()
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["model"]["input_size"] == 784
    assert cfg["training"]["batch_size"] > 0
    assert os.path.isabs(cfg["dataset_dir"])


def test_training_reaches_threshold():
    """Smoke train a few epochs; the full run (config epochs) must exceed 95%.
    Writes to a separate checkpoint dir so it never clobbers the production model."""
    if not os.path.isdir("Dataset"):
        pytest.skip("MNIST Dataset/ not present.")
    acc = train(epochs_override=3, checkpoint_dir_override="checkpoints_test")
    assert acc > 80.0, f"short training collapsed: {acc:.2f}%"
