import torch
from .Normalizer import Normalizer


class AffineNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def forward(self, x, h, context=None):
        mu, sigma = h[:, :, 0].clamp_(-5., 5.), torch.exp(h[:, :, 1].clamp_(-5., 2.))
        z = x * sigma + mu
        return z, sigma

    def inverse_transform(self, z, h, context=None):
        mu, sigma = h[:, :, 0].clamp_(-5., 5.), torch.exp(h[:, :, 1].clamp_(-5., 2.))
        x = (z - mu)/sigma
        return x
