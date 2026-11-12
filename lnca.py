"""Paper-based LNCA depth reconstruction, NOT an official released implementation.

Source: https://link.springer.com/article/10.1007/s42979-026-05327-4
See LNCA_NOTES.md for supported details and assumptions.
"""
import torch
from torch import nn


class LNCADepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
        )
        self.perception = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.update = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.ReLU(), nn.Conv2d(64, 32, 1),
        )
        self.decoder = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.PixelShuffle(8))

    def forward(self, image):
        state = self.encoder(image)
        for _ in range(self.steps):
            perceived = torch.cat((state, self.perception(state)), dim=1)
            state = state + self.update(perceived)
        return torch.exp(self.decoder(state))
