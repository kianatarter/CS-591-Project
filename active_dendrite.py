import math
import torch
import torch.nn as nn


class ActiveDendriteSurrogate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, threshold, scale, smooth=False):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.scale = scale
        u = v - threshold
        if smooth:
            # "Smooth twin" used only for the gradient checks:
            # u / (1 + k|u|) is the function that has the exact derivative in our
            # surrogate, so the finite differences can verify the backward pass.
            return u / (1.0 + scale * u.abs())
        return (u > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        u = v - ctx.threshold
        surrogate = 1.0 / (1.0 + ctx.scale * u.abs()) ** 2
        #One return value per forward input: v, threshold, scale, smooth
        return grad_output * surrogate, None, None, None


class ActiveDendriteLayer(nn.Module):
    def __init__(self, in_features, num_dendrites, threshold=1.0, scale=10.0):
        # 1.Initializing parent PyTorch nn.Module state tracking
        super(ActiveDendriteLayer, self).__init__()

        # 2. Storing layer hyperparameters
        self.in_features = in_features
        self.num_dendrites = num_dendrites
        self.threshold = threshold
        self.scale = scale
        self.smooth = False  # this is ONLY switched on by the gradient check

        # 3.Allocate the learnable dendritic weights in [num_dendrites, in_features]
        # This maps raw input features to individual dendritic branch compartments
        self.w_dend = nn.Parameter(torch.empty(num_dendrites, in_features))

        # 4.Allocate learnable somatic coupling weights with [1, num_dendrites]
        # This maps active branch dSpikes to the single somatic output (cell body)
        self.w_soma = nn.Parameter(torch.empty(1, num_dendrites))

        # 4b. Learnable soma bias (shifting where the cell body's decision sits)
        self.b_soma = nn.Parameter(torch.zeros(1))

        # 5. Initializing weights with Kaiming Uniform for variance scaling
        nn.init.kaiming_uniform_(self.w_dend, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_soma, a=math.sqrt(5))

    def forward(self, x):
        # STAGE 1: Dendritic Branch Integration
        # x: [batch_size, in_features]
        # w_dend.t(): [in_features, num_dendrites]
        # v_dend result: [batch_size, num_dendrites]
        v_dend = torch.matmul(x, self.w_dend.t())

        # STAGE 1b: Active Non-Linear Thresholding (dSpike generation)
        # d_spikes result: [batch_size, num_dendrites] (binary 0.0 or 1.0)
        d_spikes = ActiveDendriteSurrogate.apply(
            v_dend, self.threshold, self.scale, self.smooth
        )

        # STAGE 2: Somatic Integration
        # d_spikes: [batch_size, num_dendrites]
        # w_soma.t(): [num_dendrites, 1]
        # v_soma result: [batch_size, 1]
        v_soma = torch.matmul(d_spikes, self.w_soma.t()) + self.b_soma

        return v_soma
