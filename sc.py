import torch
import torch.nn as nn
import torch.nn.functional as F


class Predator_SCModule(nn.Module):
    """
    Structure Completion & Density Aware Module for Predator
    Patch-level, no voxel, no fake geometry
    """
    def __init__(
        self,
        in_dim=64,
        hidden_dim=128,
        k=16,
        nhead=4,
        num_layers=2
    ):
        super().__init__()

        self.k = k
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # -------- 1. feature adaptation --------
        if in_dim != hidden_dim:
            self.feat_adapt = nn.Linear(in_dim, hidden_dim)
        else:
            self.feat_adapt = nn.Identity()

        # -------- 2. density encoder (reliability, not semantic) --------
        self.density_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        # -------- 3. local geometry encoder (DGCNN-like) --------
        self.geo_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -------- 4. transformer encoder (structure completion in feature space) --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -------- 5. attention fusion --------
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, feat, coord):
        """
        Args:
            feat:  (B, N, C)
            coord: (B, N, 3) 真实点坐标（patch 内）
        Returns:
            feat_out: (B, N, hidden_dim)
        """
        B, N, _ = coord.shape

        # ---------- step 1: feature adapt ----------
        feat = self.feat_adapt(feat)  # (B, N, hidden_dim)

        # ---------- step 2: density estimation ----------
        # density = number of neighbors within patch (simple & stable)
        # shape: (B, 1)
        density = torch.full(
            (B, 1),
            float(N),
            device=coord.device
        )
        density_weight = self.density_mlp(density).unsqueeze(1)  # (B, 1, hidden_dim)

        # ---------- step 3: local geometry encoding ----------
        geo_feat = torch.cat([feat, coord], dim=-1)
        geo_feat = self.geo_mlp(geo_feat)

        # density as reliability gate
        geo_feat = geo_feat * density_weight

        # ---------- step 4: structure completion (Transformer) ----------
        # positional encoding: relative coordinates
        pos = coord - coord.mean(dim=1, keepdim=True)
        pos = pos / (pos.norm(dim=-1, keepdim=True) + 1e-6)

        completed_feat = self.transformer(geo_feat + pos)

        # ---------- step 5: attention fusion ----------
        fusion_weight = self.fusion_gate(
            torch.cat([feat, completed_feat], dim=-1)
        )

        feat_out = fusion_weight * feat + (1.0 - fusion_weight) * completed_feat

        return feat_out
