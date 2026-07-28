"""Fully-binarized neural network (BNN) for MNIST digit recognition.

Contract source of truth: docs/ARCHITECTURE.md (esp. §2, §3, §5).

Key properties (must stay in sync with docs/ARCHITECTURE.md):
  * Weights AND activations are in {-1, +1}.
  * The 784-pixel INPUT is binarized by sign AFTER standardization
    (ARCHITECTURE.md §2/§4) so every layer is a uniform popcount unit.
  * Hidden layers use STE `Sign` as the activation (NOT Hardtanh).
  * The output layer has NO sign activation; argmax(logits) = digit.
  * BatchNorm is folded at export time into a single integer threshold
    per hidden neuron, and a float offset per output class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySignSTE(torch.autograd.Function):
    """Straight-through estimator for sign binarization to {-1, +1}.

    Forward:  sign(x), with sign(0) mapped to +1 for determinism.
    Backward: pass gradient through where |x| <= 1, else 0 (classic STE).
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0.0
        return grad_input


def binary_sign(x):
    return BinarySignSTE.apply(x)


class BinarizeWeightSTE(torch.autograd.Function):
    """Binarize weights to {0, 1} (mapped from {-1, +1}) with STE."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0.0
        return grad_input


def binarize_weight(w):
    return BinarizeWeightSTE.apply(w)


class BinarySign(nn.Module):
    """Module wrapper so Sign can sit in an nn.Sequential."""

    def forward(self, x):
        return binary_sign(x)


class BinarizeLinear(nn.Linear):
    """Linear layer with weights binarized to {-1, +1} on the forward pass.

    Bias is disabled (contract: bias-free). The STE keeps gradients flowing
    to the real-valued latent weights during training.
    """

    def __init__(self, in_features, out_features, bias=False):
        super(BinarizeLinear, self).__init__(in_features, out_features, bias=bias)
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, input):
        bw = binarize_weight(self.weight) * 2.0 - 1.0  # {0,1} -> {-1,+1}
        return F.linear(input, bw, self.bias)

    def clip_weights(self, clip_val=1.0):
        with torch.no_grad():
            self.weight.data.clamp_(-clip_val, clip_val)


class BNN(nn.Module):
    """784 -> [Lin, BN, Sign] -> 256 -> [Lin, BN, Sign] -> 256 -> [Lin, BN] -> 10

    The input is binarized by sign at the very first step (ARCHITECTURE.md §2/§4).
    Hidden activations are STE Sign. The output layer has no sign.
    """

    def __init__(self, input_size=784, hidden_sizes=(256, 256), num_classes=10):
        super(BNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes

        layers: list[nn.Module] = [BinarySign()]  # binarize the standardized input first
        in_features = input_size
        for h_size in hidden_sizes:
            layers.append(BinarizeLinear(in_features, h_size, bias=False))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(BinarySign())  # STE Sign between hidden layers
            in_features = h_size
        layers.append(BinarizeLinear(in_features, num_classes, bias=False))
        layers.append(nn.BatchNorm1d(num_classes))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 1, 28, 28) or (batch, 784) — standardize upstream in dataset.
        return self.features(x)

    def clip_weights(self, clip_val=1.0):
        for m in self.modules():
            if isinstance(m, BinarizeLinear):
                m.clip_weights(clip_val)
            elif isinstance(m, nn.BatchNorm1d):
                # keep gamma > 0 so export folding is unambiguous
                with torch.no_grad():
                    m.weight.data.clamp_(min=1e-3)
