import math
import torch
import torch.nn as nn


class ActiveDendriteSurrogate(torch.autograd.Function):
    """
    The dendritic spike (dSpike) with a surrogate gradient.

    FORWARD:  a hard step. A branch fires (1.0) if its voltage crosses the
              threshold, otherwise it stays silent (0.0). This is the
              "biological" all-or-nothing behavior.

    BACKWARD: the true derivative of a step is 0 almost everywhere, which would
              stop learning cold. So we pretend the step was a smooth S-curve
              and use the "fast sigmoid" derivative instead:

                  dS/dV  ~=  1 / (1 + k * |V - V_thresh|)^2

              It is largest right at the threshold and fades out as V moves
              away from it. k (= scale) controls how sharp that bump is.
    """

    @staticmethod
    def forward(ctx, v, threshold, scale, smooth=False):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.scale = scale
        u = v - threshold
        if smooth:
            # "Smooth twin" used ONLY for the gradient check:
            # u / (1 + k|u|) is the function whose exact derivative is our
            # surrogate, so finite differences can verify the backward pass.
            return u / (1.0 + scale * u.abs())
        return (u > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        u = v - ctx.threshold
        surrogate = 1.0 / (1.0 + ctx.scale * u.abs()) ** 2
        # One return value per forward input: v, threshold, scale, smooth
        return grad_output * surrogate, None, None, None


class ActiveDendriteLayer(nn.Module):
    def __init__(self, in_features, num_dendrites, threshold=1.0, scale=10.0):
        # 1. Initialize parent PyTorch nn.Module state tracking
        super(ActiveDendriteLayer, self).__init__()

        # 2. Store layer hyperparameters
        self.in_features = in_features
        self.num_dendrites = num_dendrites
        self.threshold = threshold
        self.scale = scale
        self.smooth = False  # only switched on by the gradient check

        # 3. Allocate learnable dendritic weights: [num_dendrites, in_features]
        # Maps raw input features to individual dendritic branch compartments
        self.w_dend = nn.Parameter(torch.empty(num_dendrites, in_features))

        # 4. Allocate learnable somatic coupling weights: [1, num_dendrites]
        # Maps active branch dSpikes to the single somatic output (cell body)
        self.w_soma = nn.Parameter(torch.empty(1, num_dendrites))

        # 4b. Learnable soma bias (shifts where the cell body's decision sits)
        self.b_soma = nn.Parameter(torch.zeros(1))

        # 5. Initialize weights with Kaiming Uniform for variance scaling
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
