import torch
import torch.nn as nn
from sw.model.bnn import BNN, BinarizeLinear, binary_sign


def test_binary_sign():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = binary_sign(x)
    assert torch.all(y == torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))


def test_bnn_forward_shape():
    model = BNN(input_size=784, hidden_sizes=[256, 256], num_classes=10)
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_input_is_binarized():
    """Contract §2/§4: the first layer of the model is a Sign binarization of
    the standardized input, so every downstream unit sees {-1,+1} bits."""
    import torch.nn as nn

    from sw.model.bnn import BinarySign

    model = BNN(input_size=784, hidden_sizes=[256, 256], num_classes=10)
    assert isinstance(model.features[0], BinarySign), "first module must binarize input"
    x = torch.randn(3, 784)
    with torch.no_grad():
        y = model.features[0](x)
    assert torch.all((y == 1.0) | (y == -1.0)), "binarized input must be in {-1,+1}"


def test_bnn_deterministic():
    torch.manual_seed(42)
    m1 = BNN()
    torch.manual_seed(42)
    m2 = BNN()
    # Identical init => identical weights.
    p1 = [p.detach().clone() for p in m1.parameters()]
    p2 = [p.detach().clone() for p in m2.parameters()]
    assert all(torch.allclose(a, b) for a, b in zip(p1, p2)), "weights not deterministic"
    # Identical forward under same input.
    m1.eval()
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(m1(torch.ones(1, 784)), m2(torch.ones(1, 784)))
