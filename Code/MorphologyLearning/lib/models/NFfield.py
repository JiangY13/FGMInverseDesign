import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from ..utils import * #lib.

@dataclass
class NF_Config:
    latent_dim: int
    steps: int = 1000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    batch_size: Optional[int] = None
    normalize: bool = True
    n_layers: int = 4
    hidden_dim: int = 96
    s_scale: float = 0.7
    noise_ratio: float = 0.025
    grad_clip: float = 5.0
    q: float = 0.1

class RealNVPCoupling(nn.Module):
    def __init__(self, dim, hidden, mask, s_scale=0.7):  # s_scale 可 1.0~2.0
        super().__init__()
        self.register_buffer("mask", mask)  # shape (dim,)
        in_dim, out_dim = dim, dim
        self.s_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.t_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.s_scale = s_scale

    def forward(self, x):
        x_mask = x * self.mask
        s_raw = self.s_net(x_mask)
        t = self.t_net(x_mask)
        s = torch.tanh(s_raw) * self.s_scale  # << 限幅
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = x_mask + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=1)
        return y, logdet

    def inverse(self, y):
        y_mask = y * self.mask
        s_raw = self.s_net(y_mask)
        t = self.t_net(y_mask)
        s = torch.tanh(s_raw) * self.s_scale
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        x = y_mask + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -s.sum(dim=1)
        return x, logdet

class RealNVP(nn.Module):
    def __init__(self, dim, n_layers=4, hidden=96, s_scale=0.7):  #s_scale 0.7–1.0
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            parity = i % 2  # parity=0 -> [1,0,1,0,...]；parity=1 -> [0,1,0,1,...]
            mask = torch.arange(dim) % 2
            if parity == 1:
                mask = 1 - mask
            layers.append(RealNVPCoupling(dim, hidden, mask.float(), s_scale=s_scale))
        self.layers = nn.ModuleList(layers)

    def _forward(self, x):
        z = x
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer(z)
            logdet += ld
        return z, logdet

    def _inverse(self, z):
        x = z
        logdet = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet += ld
        return x, logdet

    def log_prob(self, x):
        z, logdet = self._forward(x)
        logpz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return logpz + logdet
    
    @torch.no_grad()
    def sample(self, n: int, device=None, dtype=None):
        """
        Sample x from the learned data distribution.
        Returns:
            x: [n, dim]
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        z = torch.randn(n, self.dim, device=device, dtype=dtype)  # base Gaussian
        x, _ = self._inverse(z)
        return x

def train_flow(latent_bank: torch.Tensor,
               cfg: Optional[NF_Config] = None,
               save_path="flow", 
               device='cuda'):
    flow = RealNVP(dim=cfg.latent_dim, n_layers=cfg.n_layers, hidden=cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.normalize:
        Z_mu = latent_bank.mean(0, keepdim=True)
        Z_std = latent_bank.std(0, keepdim=True).clamp_min(1e-6)
        Z_norm = (latent_bank - Z_mu) / Z_std
    else:
        Z_norm = latent_bank.clone()
    Z_std_scalar = Z_norm.std().item()
    cfg.steps = cfg.steps+1
    for t in range(cfg.steps):
        if cfg.batch_size is None:
            batch = Z_norm
        else:
            idx = torch.randint(0, len(Z_norm), (cfg.batch_size,), device=Z_norm.device)
            batch = Z_norm[idx]
        batch_aug = batch + cfg.noise_ratio * Z_std_scalar * torch.randn_like(batch)
        loss = -flow.log_prob(batch_aug).mean()
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
        if t % 200 == 0:
            print(f"[flow] step {t:03d}/{cfg.steps} | " 
                  f"nll={loss.item():.4f}")
        if loss.item() < -2:
            print(f"[flow] step {t:03d}/{cfg.steps} | " 
                  f"nll={loss.item():.4f}")
            save_model(save_path, "latest.pth", flow, t)
            break
        opt.step()
    return flow


class NFField:
    def __init__(self, flow, latent, q=0.01):
        self.flow = flow
        with torch.no_grad():
            self.mu = latent.mean(0, keepdim=True)
            self.std = latent.std(0, keepdim=True).clamp_min(1e-6)
            norm = (latent - self.mu) / self.std
            lp = self.flow.log_prob(norm)
            self.mu_lp = lp.mean()
            self.sd_lp = lp.std().clamp_min(1e-6)
            s_train = (lp - self.mu_lp) / self.sd_lp
            self.s_min = torch.quantile(s_train, q)


    def s(self, z):  
        zn = (z - self.mu) / self.std
        lp = self.flow.log_prob(zn)
        return (lp - self.mu_lp) / self.sd_lp

    def s0(self, z):  # s0(z) >= 0 for training set
        return self.s(z) - self.s_min
    
    @torch.no_grad()
    def sample(self, n):
        zn = self.flow.sample(n)

        if isinstance(zn, tuple):
            zn = zn[0]

        z = zn * self.std + self.mu
        return z
