# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

from nn_models import MLP
from utils import rk4

class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, assume_canonical_coords=True, baseline=False):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None):
        if self.baseline:
            return self.differentiable_model(x)
        H = self.differentiable_model(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dH @ self.M  # Correct symplectic structure

    def permutation_tensor(self,n):
        # Canonical symplectic matrix [[0, I], [-I, 0]]
        M = torch.eye(n)
        M = torch.cat([M[n//2:], -M[:n//2]])
        return M


class PixelHNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, autoencoder,
                 field_type='solenoidal', nonlinearity='tanh', baseline=False):
        super(PixelHNN, self).__init__()
        self.autoencoder = autoencoder
        self.baseline = baseline

        output_dim = input_dim if baseline else 2
        nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity)
        self.hnn = HNN(input_dim, differentiable_model=nn_model, field_type=field_type, baseline=baseline)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def time_derivative(self, z, separate_fields=False):
        return self.hnn.time_derivative(z, separate_fields)

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.time_derivative(z)
        return self.decode(z_next)
