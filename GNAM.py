import torch
import torch.nn as nn
import torch.nn.functional as F

class GNAMWeight(nn.Module):
    """
    GNAM-lite: global structure-aware edge weighting
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 1, bias=False),
            nn.InstanceNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, 1, 1)
        )

    def forward(self, edge_feat):
        """
        edge_feat: [B, C, N, k]
        return:
            weight:   [B, 1, N, k]
        """
        w = self.mlp(edge_feat)          # [B, 1, N, k]
        w = torch.softmax(w, dim=-1)     # neighbor-wise normalization
        return w





class SelfAttention(nn.Module):
    def __init__(self, feature_dim, k=10):
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, 1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(feature_dim * 2, feature_dim * 2, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, 1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        # === GNAM weights ===
        self.gnam1 = GNAMWeight(feature_dim * 2)
        self.gnam2 = GNAMWeight(feature_dim * 2)

        self.k = k

    def forward(self, coords, features):
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        # -------- layer 1 --------
        x1 = get_graph_feature(coords, x0.squeeze(-1), self.k)  # [B, 2C, N, k]

        w1 = self.gnam1(x1)           # [B, 1, N, k]
        x1 = x1 * w1                  # GNAM reweight

        x1 = F.leaky_relu(self.in1(self.conv1(x1)), 0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]

        # -------- layer 2 --------
        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k)

        w2 = self.gnam2(x2)
        x2 = x2 * w2

        x2 = F.leaky_relu(self.in2(self.conv2(x2)), 0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        # -------- fusion --------
        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), 0.2).view(B, -1, N)

        return x3
