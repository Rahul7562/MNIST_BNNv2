import yaml
from pathlib import Path

def get_config():
    root_dir = Path(__file__).resolve().parent.parent
    config = {}

    # Load default
    default_path = root_dir / "configs" / "default.yaml"
    with open(default_path, "r") as f:
        config.update(yaml.safe_load(f) or {})

    # Load model config
    model_path = root_dir / "configs" / "model.yaml"
    with open(model_path, "r") as f:
        config["model"] = yaml.safe_load(f) or {}

    # Load training config
    training_path = root_dir / "configs" / "training.yaml"
    with open(training_path, "r") as f:
        config["training"] = yaml.safe_load(f) or {}

    # Resolve relative paths
    for key in ["dataset_dir", "mem_dir"]:
        if key in config:
            config[key] = str(root_dir / config[key])

    if "checkpoint_dir" in config.get("training", {}):
        config["training"]["checkpoint_dir"] = str(root_dir / config["training"]["checkpoint_dir"])

    return config
