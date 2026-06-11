import torch
import torch.nn as nn
import torch.nn.functional as F


class Predator_DAPModule(nn.Module):
    """
    Density-Aware Patch Module for Predator
    Patch-level reliability modeling
    """
    def __init__(
        self,
        in_dim=128,
        out_dim=256
    ):
        super().__init__()

        # feature adaptation
        if in_dim != out_dim:
            self.feat_adapt = nn.Linear(in_dim, out_dim)
        else:
            self.feat_adapt = nn.Identity()

        # density (reliability) encoder
        self.density_mlp = nn.Sequential(
            nn.Linear(2, out_dim // 2),   # [num_points, avg_radius]
            nn.ReLU(),
            nn.Linear(out_dim // 2, out_dim),
            nn.Sigmoid()
        )

        # density-aware feature refinement
        self.refine_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, feat, coord):
        """
        Args:
            feat:  (B, N, C)   patch features
            coord: (B, N, 3)   patch coordinates
        Returns:
            feat_out: (B, N, out_dim)
        """
        B, N, _ = coord.shape

        # -------- step 1: adapt feature dim --------
        feat = self.feat_adapt(feat)  # (B, N, out_dim)

        # -------- step 2: density / reliability estimation --------
        # number of points (normalized)
        num_points = torch.full(
            (B, 1),
            float(N),
            device=coord.device
        ) / 64.0  # normalization constant

        # average spatial radius
        center = coord.mean(dim=1, keepdim=True)
        radius = torch.norm(coord - center, dim=-1).mean(dim=1, keepdim=True)

        density_feat = torch.cat([num_points, radius], dim=-1)
        density_weight = self.density_mlp(density_feat).unsqueeze(1)  # (B, 1, out_dim)

        # -------- step 3: density-aware refinement --------
        refined_feat = self.refine_mlp(feat)
        feat_out = density_weight * refined_feat + (1.0 - density_weight) * feat

        return feat_out
