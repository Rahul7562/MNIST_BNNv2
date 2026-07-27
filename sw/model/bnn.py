import torch
import torch.nn as nn
import torch.nn.functional as F

class BinarySignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input).clamp(min=0) * 2.0 - 1.0

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1.0)] = 0
        grad_input[input.lt(-1.0)] = 0
        return grad_input

def binary_sign(x):
    return BinarySignSTE.apply(x)

class BinaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input >= 0, 1.0, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1.0)] = 0
        grad_input[input.lt(-1.0)] = 0
        return grad_input

def binary_weight(x):
    return BinaryWeightSTE.apply(x)

class BinarizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BinarizeLinear, self).__init__(in_features, out_features, bias=bias)
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, input):
        bw = binary_weight(self.weight)
        return F.linear(input, bw, self.bias)

class BNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 256], num_classes=10):
        super(BNN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        layers = []
        in_features = input_size

        for h_size in hidden_sizes:
            layers.append(BinarizeLinear(in_features, h_size, bias=False))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.Hardtanh(inplace=True))
            in_features = h_size

        layers.append(BinarizeLinear(in_features, num_classes, bias=False))
        layers.append(nn.BatchNorm1d(num_classes))

        self.features = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(0, len(self.hidden_sizes)):
            idx = i * 3
            x = self.features[idx](x)
            x = self.features[idx+1](x)
            x = binary_sign(x)

        out_idx = len(self.hidden_sizes) * 3
        x = self.features[out_idx](x)
        x = self.features[out_idx+1](x)
        return x

    def clip_weights(self, clip_val=1.0):
        for m in self.modules():
            if isinstance(m, BinarizeLinear):
                m.weight.data.clamp_(-clip_val, clip_val)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.clamp_(min=0.001)
