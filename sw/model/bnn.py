import torch
import torch.nn as nn
import torch.nn.functional as F

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input).add_(input == 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

binarize = Binarize.apply

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # binarize weights
        bw = binarize(self.weight)
        # binarize activations is done before the linear layer usually, or handled explicitly
        # we will handle binarized inputs during the forward pass of the model
        return F.linear(input, bw, self.bias)

class BNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 256], num_classes=10):
        super(BNN, self).__init__()

        self.features = nn.Sequential(
            nn.Flatten(),

            # Input layer (not binarized input, but weights are binarized)
            BinarizeLinear(input_size, hidden_sizes[0], bias=False),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Hardtanh(inplace=True), # Using Hardtanh as a good activation before binarization

            # Hidden layer 1
            BinarizeLinear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Hardtanh(inplace=True),

            # Output layer
            BinarizeLinear(hidden_sizes[1], num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        # Flatten x
        x = x.view(x.size(0), -1)

        # Go through features
        for i, layer in enumerate(self.features):
            if isinstance(layer, BinarizeLinear) and i > 1:
                # Binarize inputs for hidden layers and output layer
                x = binarize(x)
            x = layer(x)

        return x
