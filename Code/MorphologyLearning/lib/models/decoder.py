#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from ..utils import *


class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = False,
        t_period = 1.0,
        fourier_degree = 1
    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []

        self.fourier_degree = fourier_degree
        if (positional_encoding is True) and (self.fourier_degree >=1):
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        elif self.fourier_degree == 0.9:  # y,z
            dims = [latent_size + 5] + dims + [1]
        elif self.fourier_degree == 0.8:  # x,z
            dims = [latent_size + 5] + dims + [1]
        elif self.fourier_degree == 0.6:  # z
            dims = [latent_size + 4] + dims + [1]
        elif self.fourier_degree == 0.5:  # x,y
            dims = [latent_size + 5] + dims + [1]
        elif self.fourier_degree == 0.3:  # y
            dims = [latent_size + 4] + dims + [1]
        elif self.fourier_degree == 0.2:  # x
            dims = [latent_size + 4] + dims + [1]
        elif self.fourier_degree == 0.15:  # x-cos
            dims = [latent_size + 3] + dims + [1]
        elif self.fourier_degree == 0.05:  # x-sin
            dims = [latent_size + 3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]
        self.LMin, self.LMax = 6, 30
        self.positional_encoding = positional_encoding
        self.t_period = t_period
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def applyFourierMapping(self, x, fourierMap):
        c = torch.cos(2 * np.pi * torch.matmul(x, fourierMap))
        s = torch.sin(2 * np.pi * torch.matmul(x, fourierMap))
        xv = torch.cat((c, s), dim=-1)
        return xv

    # input: N x (L+3)
    def forward(self, latent, xyz):

        if self.positional_encoding:
            if self.fourier_degree > 1:
                coordnMap = np.zeros((3, 3 * self.fourier_degree))
                for i in range(coordnMap.shape[0]):
                    for j in range(coordnMap.shape[1]):
                        coordnMap[i, j] = np.random.choice([-1., 1.]) * np.random.uniform(
                            1. / (2 * self.LMax), 1. / (2 * self.LMin))  #
                fourierMap = torch.tensor(coordnMap).to(device=xyz.device)  #
                xyz_dim = 3 * self.fourier_degree
                xyz = self.applyFourierMapping(xyz, fourierMap)
            elif (self.fourier_degree <= 1) and (self.fourier_degree > 0): # x,y,z score: 0.2，0.3，0.6
                fourierMap = torch.eye(3).to(device=xyz.device) / self.t_period
                xyz_all = self.applyFourierMapping(xyz, fourierMap)
                if self.fourier_degree == 1:
                    xyz_dim = 3 * 2
                    xyz = xyz_all
                elif self.fourier_degree == 0.9:  #y,z
                    xyz_dim = 3 + 2
                    xyz = torch.stack((xyz[:,0], xyz_all[:,1], xyz_all[:,2], xyz_all[:,4], xyz_all[:,5]), dim=-1)
                elif self.fourier_degree == 0.8:  #x,z
                    xyz_dim = 3 + 2
                    xyz = torch.stack((xyz[:,1], xyz_all[:,0], xyz_all[:,2], xyz_all[:,3], xyz_all[:,5]), dim=-1)
                elif self.fourier_degree == 0.6:  #z
                    xyz_dim = 3 + 1
                    xyz = torch.stack((xyz[:,0], xyz[:,1], xyz_all[:,2], xyz_all[:,5]), dim=-1)
                elif self.fourier_degree == 0.5: #x,y
                    xyz_dim = 3 + 2
                    xyz = torch.stack((xyz[:,2], xyz_all[:,0], xyz_all[:,1], xyz_all[:,3], xyz_all[:,4]), dim=-1)
                elif self.fourier_degree == 0.3:  #y
                    xyz_dim = 3 + 1
                    xyz = torch.stack((xyz[:,0], xyz[:,2], xyz_all[:,1], xyz_all[:,4]), dim=-1)
                elif self.fourier_degree == 0.2:  #x
                    xyz_dim = 3 + 1
                    xyz = torch.stack((xyz[:,1], xyz[:,2], xyz_all[:,0], xyz_all[:,3]), dim=-1)
                elif self.fourier_degree == 0.15:  #x-cos
                    xyz_dim = 3
                    xyz = torch.stack((xyz[:,1], xyz[:,2], xyz_all[:,0]), dim=-1)
                elif self.fourier_degree == 0.05:  #x-sin
                    xyz_dim = 3
                    xyz = torch.stack((xyz[:,1], xyz[:,2], xyz_all[:,3]), dim=-1)
            else:  # plane symetry
                fourierMap = torch.eye(3).to(device=xyz.device) / self.t_period
                xyz_dim = 3
                xyz = torch.cos(2 * np.pi * torch.matmul(xyz, fourierMap))
        else:
            xyz_dim = 3
        if len(latent) > 0:
            input = torch.cat([latent, xyz.clone()], dim=1)
        else:
            input = xyz.clone()

        if input.shape[1] > xyz_dim and self.latent_dropout:
            latent_vecs = input[:, :-xyz_dim]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
