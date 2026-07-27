import torch
from sw.model.bnn import BNN, binary_sign

def test_binary_sign():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y = binary_sign(x)
    # torch.sign(0.0) is 0.0. clamp(min=0) -> 0.0. * 2 - 1 = -1.0.
    # So 0.0 maps to -1.0 in my implementation!
    assert torch.all(y == torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]))

    y.sum().backward()
    assert torch.all(x.grad == torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))

def test_model_shapes():
    model = BNN(input_size=10, hidden_sizes=[20, 20], num_classes=5)
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 5)

def test_model_determinism():
    torch.manual_seed(42)
    model1 = BNN()
    model1.eval() # Prevent batchnorm failure on batch size 1
    out1 = model1(torch.ones(1, 784))

    torch.manual_seed(42)
    model2 = BNN()
    model2.eval()
    out2 = model2(torch.ones(1, 784))

    assert torch.all(out1 == out2)
