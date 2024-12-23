import diffusion_model as G
import deformation_model as M
import STL as stl
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = G.UNet()
        self.M = M.UNet()
        self.stl = stl.Dense2DSpatialTransformer()

    def forward(self, m, f, ft, t):
        epsilonhat = self.G(m, f, ft, t)
        phi = self.M(epsilonhat, m)
        m = (m + 1)/2
        mphi = self.stl(m, phi)

        return epsilonhat, mphi, phi
