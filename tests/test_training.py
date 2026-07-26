import os
import pytest
from config import get_config

def test_config_load():
    cfg = get_config()
    assert "model" in cfg
    assert "training" in cfg
    assert cfg["model"]["input_size"] == 784
    assert cfg["training"]["batch_size"] > 0
    assert os.path.isabs(cfg["dataset_dir"])
