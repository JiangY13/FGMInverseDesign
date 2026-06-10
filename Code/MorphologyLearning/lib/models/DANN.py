from dataclasses import dataclass
from typing import Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import Tuple
# =========================
# 1. Training config
# =========================

@dataclass
class DA_Config:
    latent_dim: int = 32
    batch_size: int = 512
    steps: int = 50
    train_samples: int = 100000
    lr: float = 1e-4
    weight_decay: float = 1e-6
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.0
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0

    # Loss weights
    lambda_nf: float = 1.0
    lambda_smooth: float = 0.05
    lambda_regular: float = 1.0
    delta_margin: float =0.0


# =========================
# 1. Linear Interpolation with DA network
# =========================
class DA_Trans(nn.Module):
    def __init__(self, da_net: nn.Module, nf_model, cyclic=False):
        super().__init__()
        """
        waypoints: [K+1, D]（K sections）or [K, D]（if closed=True）
        t_vals: [N]，clamp into [0, K]
                for example: [0,0,0,0, 0.2,0.4,..., 1,1,1,1, 1.2,..., K, K,...]
        cyclic: if True，the last section (zK-1 → z0)，t in [0,K]
        return: [N, D]
        """
        self.cyclic = cyclic
        self.da_net = da_net  
        self.nf_model = nf_model

    def forward(self, LVs: torch.Tensor, t_vec: torch.Tensor, w: float = 0.0):  #-> Dict[str, torch.Tensor]:
        """
        Linear interpolation：z0,z1: [..., D], u: [..., 1] in [0,1]
        """
        L_length = LVs.shape[0]
        if LVs.dim() == 1 or L_length < 2:
            raise ValueError("At least two latent vectors are required for graded design.")
    
        t_vals = piecewise_transition(t_vec, L_length, w)
        if self.cyclic:
            # K sections：z0→z1, ..., z_{K-1}→z0
            K = L_length
            # t in [0,K)
            t = (t_vals % float(K))
            seg = torch.floor(t).long()
            seg_next = (seg + 1) % K

        else:
            # K = Z.shape[0]-1 sections
            K = L_length - 1
            if K <= 0:
                raise ValueError("waypoints must have at least 2 rows for open chain.")
            t = t_vals.clamp(0.0, float(K))
            seg = torch.floor(t).long()  # .clamp(0, K - 1)
            seg_next = (seg + 1).clamp(0, K)
        l_t = (t - seg).unsqueeze(-1)

        # Linear interpolation
        z0 = torch.index_select(LVs, 0, seg)  # [N, D]
        z1 = torch.index_select(LVs, 0, seg_next)  # [N, D]

        out = (1.0 - l_t) * z0 + l_t * z1
        #
        nf_out = self.nf_model.s0(out)
        sigmoid_weight = torch.sigmoid(- nf_out)[:,None]
        da_mapping = self.da_net(out)
        z0_map = z0 + torch.sigmoid(- self.nf_model.s0(z0))[:,None] * self.da_net(z0)["out"]
        z1_map = z1 + torch.sigmoid(- self.nf_model.s0(z1))[:,None] * self.da_net(z1)["out"]
        linear_mapping = (1.0 - l_t) * z0_map + l_t * z1_map
        return {
            "out": out + sigmoid_weight * da_mapping["out"],
            "linear": linear_mapping,
        }

def piecewise_transition(loc, L_length, half ):
    """
    loc: 1D tensor of coordinates, e.g. torch.arange(0., 4.0+1e-9, 0.1)
    w: transition width (e.g. 0.4)
    return: val(z) with unit-cell plateaus and width-w transitions
    """
    # int index：b = floor(z + 0.5)
    b = torch.floor(loc + 0.5)  # close to boundary（…0,1,2,3,4…elem_num）

    t = 0.5 * ((loc - b) / half).clamp(-1.0, 1.0) - 0.5  
    val = b + t
    val = val.clamp(min=0.0, max=float(L_length - 1))
    return val
# =========================
# 2. Residual transition operator
# =========================

class DANet(nn.Module):
    """
    T(l) = l + alpha(l) * D(l)
    - D(l): residual displacement
    - alpha(l): scalar gate in (0,1)
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_dim   
        # --- D(l) network ---
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.mlp = nn.Sequential(*layers)
        # --- alpha(l) gate ---
        self.alpha_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, l: torch.Tensor): #-> Dict[str, torch.Tensor]:
        if l.dim() == 1:
            l = l.unsqueeze(0)

        # displacement
        delta = self.mlp(l)
        alpha = torch.sigmoid(self.alpha_head(l)) 
        delta_scaled = alpha * delta

        return {
            "out": delta_scaled,
            "delta": delta,
            "alpha": alpha,
        }

# =========================
# 3. Dataset preparition for training
# =========================

@torch.no_grad()
def build_latent_bank_aug(
    latent_bank: torch.Tensor,
    nf_model,
    num_aug: int = 200,
    device: str = "cuda",
    max_energy_quantile: float = 0.98,
    seed: int = 42,
):
    """
    latent_bank: [N, D]
    nf_model: trained NF with sample() and log_prob()
    returns:
        latent_bank_aug: [N + num_aug, D]
        stats: dict
    """

    torch.manual_seed(seed)

    latent_bank = latent_bank.to(device)

    # ---------- Original energy ----------
    logp = nf_model.s0(latent_bank)
    if logp.dim() > 1:
        logp = logp.squeeze(-1)

    E_data = -logp

    # ---------- sampling from NF ----------
    z_nf = nf_model.sample(num_aug * 4)  # samples for selection

    if isinstance(z_nf, tuple):
        z_nf = z_nf[0]

    z_nf = z_nf.to(device)

    logp_nf = nf_model.s0(z_nf)
    if logp_nf.dim() > 1:
        logp_nf = logp_nf.squeeze(-1)

    E_nf = -logp_nf

    # ---------- remove low likelihood ----------
    E_thr = torch.quantile(E_nf, max_energy_quantile)

    mask = E_nf <= E_thr

    z_nf = z_nf[mask]
    E_nf = E_nf[mask]

    if z_nf.shape[0] < num_aug:
        raise RuntimeError(
            f"Not enough NF samples after truncation: {z_nf.shape[0]} < {num_aug}"
        )

    # ---------- random num_aug ----------
    idx = torch.randperm(z_nf.shape[0])[:num_aug]

    z_aug = z_nf[idx]

    latent_bank_aug = torch.cat([latent_bank, z_aug], dim=0)

    stats = {
        "orig_mean": E_data.mean().item(),
        "orig_std": E_data.std().item(),
        "aug_mean": E_nf.mean().item(),
        "aug_std": E_nf.std().item(),
        "threshold": E_thr.item(),
    }

    return latent_bank_aug, stats


@torch.no_grad()
def sample_from_bins(
    z: torch.Tensor,
    e: torch.Tensor,
    num_samples: int,
    bin_edges: torch.Tensor,
    bin_ratios: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified sampling from energy bins.

    z: [N, D]
    e: [N]
    bin_edges: [B+1], ascending
    bin_ratios: [B], sum not necessarily 1
    """
    assert bin_edges.ndim == 1
    assert bin_ratios.ndim == 1
    assert len(bin_edges) == len(bin_ratios) + 1

    device = z.device
    bin_ratios = bin_ratios / bin_ratios.sum()
    target_counts = torch.floor(bin_ratios * num_samples).long()
    # distribute remainders
    remainder = num_samples - target_counts.sum().item()
    if remainder > 0:
        frac = (bin_ratios * num_samples) - target_counts.float()
        add_idx = torch.argsort(frac, descending=True)[:remainder]
        target_counts[add_idx] += 1

    selected_idx = []
    selected_energy = []

    for i in range(len(bin_ratios)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == len(bin_ratios) - 1:
            mask = (e >= lo) & (e <= hi)
        else:
            mask = (e >= lo) & (e < hi)

        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0 or target_counts[i].item() == 0:
            continue

        k = min(target_counts[i].item(), idx.numel())
        perm = torch.randperm(idx.numel(), generator=generator, device=device)[:k]
        chosen = idx[perm]

        selected_idx.append(chosen)
        selected_energy.append(e[chosen])

    if len(selected_idx) == 0:
        raise RuntimeError("No samples selected from bins. Check bin edges / candidate quality.")

    selected_idx = torch.cat(selected_idx, dim=0)
    selected_energy = torch.cat(selected_energy, dim=0)

    # if not enough due to empty bins, backfill globally by closest-to-bin-center or random valid points
    if selected_idx.numel() < num_samples:
        need = num_samples - selected_idx.numel()
        chosen_mask = torch.zeros(z.shape[0], dtype=torch.bool, device=device)
        chosen_mask[selected_idx] = True
        remain_idx = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(-1)

        if remain_idx.numel() > 0:
            # random backfill from remaining valid candidates
            k = min(need, remain_idx.numel())
            perm = torch.randperm(remain_idx.numel(), generator=generator, device=device)[:k]
            extra = remain_idx[perm]
            selected_idx = torch.cat([selected_idx, extra], dim=0)
            selected_energy = torch.cat([selected_energy, e[extra]], dim=0)

    # final trim if needed
    if selected_idx.numel() > num_samples:
        perm = torch.randperm(selected_idx.numel(), generator=generator, device=device)[:num_samples]
        selected_idx = selected_idx[perm]
        selected_energy = selected_energy[perm]

    return z[selected_idx], selected_energy

@torch.no_grad()
def make_bins(
    latent_bank: torch.Tensor,
    nf_model,
    margin: float = 0.5,
    outer_margin: float = 1.0,
    device: str = "cuda",
):
    latent_bank = latent_bank.to(device)

    logp = nf_model.s0(latent_bank)
    if logp.dim() > 1:
        logp = logp.squeeze(-1)

    delta = - logp   # >0 outside, <0 inside

    inside_dist = -delta[delta <= 0]   # positive
    outside_dist = delta[delta > 0]   # positive

    if inside_dist.numel() > 10:
        margin = torch.quantile(inside_dist, 0.2).item()

    if outside_dist.numel() > 10:
        outer_margin = torch.quantile(outside_dist, 0.4).item()

    inside_mask = delta < -margin
    boundary_mask = torch.abs(delta) <= margin
    outside_near_mask = (delta > 0) & (delta <= outer_margin)
    outside_far_mask = delta > outer_margin

    bins = [
        torch.nonzero(inside_mask, as_tuple=False).squeeze(-1).cpu(),   # 0
        torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1).cpu(),      # 1
        torch.nonzero(outside_near_mask, as_tuple=False).squeeze(-1).cpu(),  # 2
        torch.nonzero(outside_far_mask, as_tuple=False).squeeze(-1).cpu(),   # 3
    ]

    return {
        "latent_bank": latent_bank.cpu(),
        "logp": logp.cpu(),
        "delta": delta.cpu(),
        "bins": bins,
    }


class IndexOnlyDataset(Dataset):
    def __init__(self, samples_per_epoch: int):
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 占位即可，真正采样在 collate_fn 里做
        return 0

import torch
from typing import Dict, Optional


class NFBoundaryWeightedInterleaveCollator:
    """
    bins:
      0: inside
      1: boundary
      2: outside-near

    Goal:
      - different sampling probabilities for different bins
      - within a batch, adjacent bins are encouraged to be different
    """
    def __init__(
        self,
        bank_info: Dict,
        seed: int = 42,
        bin_probs: Optional[torch.Tensor] = None,
        same_bin_penalty: float = 0.15,
    ):
        self.latent_bank = bank_info["latent_bank"].float()
        self.logp = bank_info["logp"].float()
        self.bins = bank_info["bins"]
        self.num_bins = len(self.bins)

        assert self.num_bins == 4, "Expected 3 bins: inside, boundary, outside-near."
        assert 0.0 <= same_bin_penalty <= 1.0

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        if bin_probs is None:
            bin_probs = torch.tensor([0.29, 0.4, 0.3, 0.01], dtype=torch.float32)

        self.bin_probs = bin_probs.float()
        self.bin_probs = self.bin_probs / self.bin_probs.sum()
        self.same_bin_penalty = same_bin_penalty

        for i, idx_tensor in enumerate(self.bins):
            if idx_tensor.numel() == 0:
                raise RuntimeError(f"Bin {i} is empty.")

    def _sample_index_from_bin(self, bin_id: int) -> int:
        idx_tensor = self.bins[bin_id]
        k = torch.randint(0, idx_tensor.numel(), (1,), generator=self.rng).item()
        return int(idx_tensor[k].item())

    def _sample_bin_sequence(self, B: int) -> torch.Tensor:
        seq = []

        # directly sample from original probs
        first = torch.multinomial(self.bin_probs, 1, generator=self.rng).item()
        seq.append(first)

        for _ in range(B - 1):
            prev = seq[-1]
            probs = self.bin_probs.clone()

            # penalize repeating the previous bin
            probs[prev] *= self.same_bin_penalty

            # if all probs collapse, fallback
            if probs.sum() <= 0:
                probs = self.bin_probs.clone()

            probs = probs / probs.sum()

            cur = torch.multinomial(probs, 1, generator=self.rng).item()
            seq.append(cur)

        return torch.tensor(seq, dtype=torch.long)

    def __call__(self, batch_indices):
        B = len(batch_indices)

        bin_seq = self._sample_bin_sequence(B)

        sample_indices = []
        for b in bin_seq:
            idx = self._sample_index_from_bin(int(b.item()))
            sample_indices.append(idx)

        sample_indices = torch.tensor(sample_indices, dtype=torch.long)

        z_seq = self.latent_bank[sample_indices]
        logp_seq = self.logp[sample_indices]

        return {
            "z_seq": z_seq,
            "logp_seq": logp_seq,
            "bin_seq": bin_seq,
            "sample_indices": sample_indices,
        }

@dataclass
class ChainTrainConfig:
    latent_dim: int
    batch_size: int
    batch_per_epoch: int
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-6
    hidden_dim: int = 256
    depth: int = 3
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    K_t: int = 5               #number of middle points
    t_mode: str = "uniform"
    delta_margin: float = 0.0  #energy weight delta_margin>=0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)
# =========================
# 5. Loss components
# =========================

def local_laplacian_mean_loss(
    net: nn.Module,
    l: torch.Tensor,
    m: int = 10,
):
    B, D = l.shape
    device = l.device

    out_main = net(l)
    r_main = out_main["out"]   # [B, D]

    sigma = 0.01 * l.std(dim=0).mean().item()
    eps = sigma * torch.randn(B, m, D, device=device)
    l_aug = l[:, None, :] + eps
    l_aug_flat = l_aug.reshape(B * m, D)

    out_aug = net(l_aug_flat)
    r_aug = out_aug["out"].reshape(B, m, D)

    r_neigh_mean = r_aug.mean(dim=1)    # [B, D]

    loss = ((r_main - r_neigh_mean) ** 2).sum(dim=-1).mean()
    return loss, l_aug_flat, out_aug['out']
##########################################################
def random_increasing(resolution, start, end):
    if resolution < 2:
        raise ValueError("resolution must >= 2")

    mid = torch.rand(resolution - 2) * (end-start) + start
    mid, _ = torch.sort(mid)

    t = torch.cat([
        torch.tensor([start]),
        mid,
        torch.tensor([end]),
    ])

    return t

# =========================
# 6. Main training step
# =========================

def train_DA_net(
    latent_info: torch.Tensor,
    nf_model,
    cfg: Optional[DA_Config] = None,
    save_path: str = "mcco",
    device='cuda',
    ):
    """
    Args
    ----
    latent_bank:
        [N, D] latent codes from your acoustic dataset.
    nf_model:
        Must expose log_prob(z) -> [B] or scalar.
    decoder:
        Optional. If given, should accept z and return geometry dict or tensor.
    """
    latent_bank = latent_info["latent_bank"]
    assert latent_bank.dim() == 2
    latent_dim = latent_bank.shape[1]

    if cfg is None:
        cfg = DA_Config(latent_dim=latent_dim)
    else:
        assert cfg.latent_dim == latent_dim, "cfg.latent_dim must match latent_bank.shape[1]"

    logp_all = latent_info["logp"]
    logp10 = torch.quantile(logp_all, 0.1)
    logp90 = torch.quantile(logp_all, 0.9)
    beta = 4.0 / (logp90 - logp10 + 1e-8)
    
    dataset = IndexOnlyDataset(samples_per_epoch=cfg.train_samples)
    collator = NFBoundaryWeightedInterleaveCollator(
        bank_info=latent_info,
        seed=42,
        bin_probs=torch.tensor([0.1, 0.3, 0.6, 0.05]),
        same_bin_penalty=0.2,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,   # batch_size == sequence length
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collator,
    )


    l_length = cfg.batch_size
    n_batches = len(loader)

# DA module
    print("Preparing DA model ...")
    DA_net = DANet(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        ).to(device)

    optimizer = torch.optim.AdamW(DA_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))


    best_loss = float("inf")
    print("Start DA training ...")
    for epoch in range(cfg.steps):
        DA_net.train()
        running = {
            "total": 0.0,
            "nf": 0.0,
            "nf_expand": 0.0,
            "smooth": 0.0,
            "time": 0.0,
        }
        loader_iter = 0
        t0 = time.perf_counter()
        for batch in loader:

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda")):
                batch_seq = batch["z_seq"].to(device)
                e_seq = batch["logp_seq"].to(device)  # [B]
                samples_number = len(batch_seq)
                out = DA_net(batch_seq)

                # 1) NF likelihood supervision
                logp = nf_model.s0(batch_seq + out["out"])
                loss_nf = torch.relu(- logp).mean()


                # 2) Laplacian Smoothness
                loss_smooth, expand_l, expand_da = local_laplacian_mean_loss(DA_net, batch_seq)
                logp_expand = nf_model.s0(expand_l.reshape(-1, cfg.latent_dim) + expand_da.reshape(-1, cfg.latent_dim))  
                loss_nf_expand = torch.relu(- logp_expand).mean()

                loss = (
                    cfg.lambda_nf * loss_nf
                    + cfg.lambda_nf * loss_nf_expand
                    + cfg.lambda_smooth * loss_smooth
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(DA_net.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running["total"] += loss.item()
            running["nf"] += loss_nf.item()
            running["nf_expand"] += loss_nf_expand.item()
            running["smooth"] += loss_smooth.item()  

            loader_iter += 1

        running["time"]=time.perf_counter() - t0
        epoch_loss = running["total"] / n_batches
        if epoch % 100 == 0:
            print(
                f"[DANN] Epoch {epoch+1:03d}/{cfg.steps} | "
                f"total={running['total']/n_batches:.6f} | "
                f"nf={running['nf']/n_batches:.6f} | "
                f"nf_expand={running['nf_expand']/n_batches:.6f} | "
                f"smooth={running['smooth']/n_batches:.6f} | "
                f"time={running['time']:.6f}"
            )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_operator(save_path, "latest.pth", DA_net, cfg, epoch)

    print(f"[DANN] Saved best model to: {save_path}")
    return DA_net

def test_DA_net(net: nn.Module,
                        LVs:torch.Tensor,
                        resolution: int,
                        gap: float,
                        nf_model,
                        device='cpu'):
    LVs = LVs.to(device)
    L_length = LVs.shape[0]
    layer_offset = torch.arange(L_length - 1).unsqueeze(1)
    random_t = random_increasing(resolution=resolution, start=0, end=gap)
    t_vector = random_t.unsqueeze(0) + layer_offset + 1 - gap / 2
    t_vector = t_vector.reshape(-1).to(device)
    out = net(LVs, t_vector, w=gap)
    logp = nf_model.s0(out["mcco"])
    loss_nf = (- logp).mean()
    loss_smooth = path_smoothness_loss(net, LVs)                 
    print(
    "Test | "
    f"nf={loss_nf.item():.6f} | "
    f"smooth={loss_smooth.item():.6f}"
)

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from MorphologyLearning.lib.models.NFfield import *
from MorphologyLearning.lib.utils import * #lib.

if __name__ == "__main__":
    
    set_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latent_dim = 32
    latent_bank = torch.randn(200, latent_dim).to(device)  # Replace with your encoded acoustic dataset latents

    nf_cfg = NF_Config(
        latent_dim=latent_dim,
        steps=5000,
        lr=5e-4, 
        weight_decay=1e-3, 
        n_layers=4,
        hidden_dim=96,
        noise_ratio=0.025,
        grad_clip=5.0,
        q=0.1,
    )

    flow = train_flow(latent_bank,
                        nf_cfg,
                        device=device).eval()
    nf_field = NFField(flow, latent_bank, q=nf_cfg.q) 

    cfg = DA_Config(
        latent_dim=latent_dim,
        cyclic=False,          
        batch_size=6,  #Decided by fea abillity
        steps = 100,
        lr=2e-4,
        hidden_dim=256,
        depth=3,
        lambda_nf=1.0,
        lambda_smooth=0.05,
        lambda_bias=0.01,
        lambda_geom=0.1,
        delta_margin=0.0,
        amp=True,
    )
    gap = 0.5

    latent_bank_aug, stats = build_latent_bank_aug(
        latent_bank=latent_bank,   # [N,32]
        nf_model=nf_field,
        num_aug=latent_bank.shape[0],
        device=device,
    )

    print(latent_bank_aug.shape)

    bank_info = make_energy_bins(
        latent_bank=latent_bank_aug,
        nf_model=nf_field,
        quantiles=(0.00, 0.40, 0.70, 0.90, 0.98),
        device=device,
    )

    DA = train_DA_net(
        latent_info=bank_info,
        # resolution=int(40*gap),
        # gap = gap,
        nf_model = nf_field,
        decoder=None,
        cfg=cfg,
        # chain_cfg=chain_cfg,
        # sw_cfg = sw_cfg,
        save_path="transition_net",
        device=device,
        ).eval()
    da_trans = DA_trans(DA)
    test_LVs = latent_bank[5:10]
    test_DA_net(da_trans, test_LVs, 
                        resolution=int(40*gap),
                        gap = gap,
                        nf_model = nf_field,
                        device=device,)
