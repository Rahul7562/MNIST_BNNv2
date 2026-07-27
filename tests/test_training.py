import os
import yaml
import pytest
import torch
from scripts.train import main, set_seed
from sw.model.bnn import BNN
from sw.training.dataset import get_dataloaders

def test_training_smoke(tmp_path):
    config_model = {
        'model': {
            'input_size': 784,
            'hidden_sizes': [32],
            'num_classes': 10
        }
    }

    config_train = {
        'training': {
            'epochs': 1,
            'batch_size': 100,
            'lr': 0.01,
            'seed': 42,
            'dataset_dir': "Dataset",
            'checkpoint_dir': str(tmp_path),
            'mem_dir': str(tmp_path / "mem_files")
        }
    }

    model_yaml = tmp_path / "model.yaml"
    with open(model_yaml, 'w') as f:
        yaml.dump(config_model, f)

    train_yaml = tmp_path / "training.yaml"
    with open(train_yaml, 'w') as f:
        yaml.dump(config_train, f)

    import sys
    sys.argv = ['train.py', '--config_model', str(model_yaml), '--config_train', str(train_yaml)]

    acc = main()
    assert acc > 10.0
    assert os.path.exists(tmp_path / 'best_model.pth')

def test_best_model_accuracy():
    if not os.path.exists('checkpoints/best_model.pth'):
        pytest.skip("best_model.pth not found. Run training first.")

    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu', weights_only=False)
    acc = checkpoint['best_acc']

    assert acc > 95.0, f"Expected >95% accuracy, but got {acc:.2f}%"
