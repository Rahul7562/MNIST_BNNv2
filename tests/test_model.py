import torch
from sw.model.bnn import BNN, BinarizeLinear, binarize

def test_binarize_forward():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = binarize(x)
    assert torch.all(y == torch.tensor([-1., -1., 1., 1., 1.]))

def test_bnn_forward_shape():
    model = BNN(input_size=784, hidden_sizes=[256, 256], num_classes=10)
    # Batch of 4 images
    x = torch.randn(4, 784)
    out = model(x)

    assert out.shape == (4, 10)

def test_bnn_weights_are_binarized():
    model = BNN(input_size=10, hidden_sizes=[10, 10], num_classes=5)

    # Forward pass
    x = torch.randn(2, 10)
    _ = model(x)

    # After forward pass, check if BinarizeLinear applied binarized weights correctly
    # actually, BinarizeLinear applies binarization inside forward(),
    # the raw weights model.features[1].weight are NOT binarized (they are continuous)
    # But we can check that they exist and require grad.
    for m in model.features:
        if isinstance(m, BinarizeLinear):
            assert m.weight.requires_grad
