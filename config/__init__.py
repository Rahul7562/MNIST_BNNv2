import yaml
from pathlib import Path

def get_config():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths
    for key in ["dataset_dir", "mem_dir"]:
        if key in config:
            config[key] = str(root_dir / config[key])

    return config
