import torch
from torch.optim import SGD
import numpy as np


class MirrorMap(object):
    """ Provides functionality related to mirror decent. """

    def __init__(self):
        super().__init__()

    def compute_mirror_map(self, x):
        """ Computes grad psi (x). """
        raise NotImplementedError

    def compute_mirror_map_inverse(self, x):
        """ Computes (grad psi)^{-1}(x). """
        raise NotImplementedError

    def compute_bregman_divergence(self, x, y):
        """ Computes D_{psi}(x, y). """
        raise NotImplementedError


class MirrorDescentOptimizer(SGD):
    """ An implementation of mirror descent optimizer for linear model. """

    def __init__(self, mirror_map, params, **kwargs):
        """ :mirror_map: An instance of class MirrorMap defining the mirror
                map, its inverse and associated Bregman divergence.
            :params: Model parameters (will be passed to SGD base class).
            :**kwrags: Keyword arguments passed to SGD base class. """
        super().__init__(params, **kwargs)
        self.mirror_map = mirror_map

    @torch.no_grad()
    def step(self):
        # Apply the mirror map to parameters.
        for group in self.param_groups:
            for p in group['params']:
                p.data = self.mirror_map.compute_mirror_map(p.data)

        # Take a GD step.
        super().step()

        # Map parameters back to the primal space.
        for group in self.param_groups:
            for p in group['params']:
                p.data = self.mirror_map.compute_mirror_map_inverse(p.data)


class L2MirrorMap(object):

    def compute_mirror_map(self, x):
        return x

    def compute_mirror_map_inverse(self, x):
        return x

    def compute_bregman_divergence(self, x, y):
        return torch.sum((x-y)**2).item()/2.0


class PreconditionedL2MirrorMap(object):
    """ Implements a mirror map psi(x) = x^t A x / 2, for some
        invertible matrix A. """

    def __init__(self, A):
        """ A should be an invertible matrix. """
        self.A = A
        self.A_inv = torch.inverse(A)

    def compute_mirror_map(self, x):
        return x @ self.A

    def compute_mirror_map_inverse(self, x):
        return x @ self.A_inv

    def compute_bregman_divergence(self, x, y):
        return 0


class HypentropyMirrorMap(object):

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def _arcsinh_gamma(self, x):
        """ Computes arcsinh(x/gamma) using float64 dtype. """
        x = x.to(dtype=torch.float64)
        log_gammas = torch.ones_like(x) * np.log(self.gamma)
        return torch.log(torch.sqrt(x**2 + self.gamma**2) + x) - log_gammas

    def compute_mirror_map(self, x):
        """ Computes arcsinh(x/gamma). """
        return self._arcsinh_gamma(x).to(dtype=torch.float32)

    def compute_mirror_map_inverse(self, x):
        x = x.to(dtype=torch.float64)
        output = torch.sinh(x) * self.gamma
        return output.to(dtype=torch.float32)

    def compute_bregman_divergence(self, x, y):
        x = x.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)
        divergence = torch.sum(
            x * (self._arcsinh_gamma(x) - self._arcsinh_gamma(y)))
        divergence -= torch.sum(torch.sqrt(x**2 + self.gamma**2))
        divergence += torch.sum(torch.sqrt(y**2 + self.gamma**2))
        return divergence.item()
