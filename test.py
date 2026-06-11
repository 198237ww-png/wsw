import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Feature Interaction Module (Low-overlap safe version)
# ============================================================

class FeatureInteractionModule(nn.Module):
    def __init__(self, d_in, mlp_hidden=64):
        super().__init__()
        self.d_in = d_in

        self.global_mlp = nn.Sequential(
            nn.Linear(2 * d_in, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, d_in)
        )

        self.alpha = nn.Parameter(torch.zeros(d_in))
        self.beta = nn.Parameter(torch.zeros(d_in))

    def forward(self, F_p, F_q):
        B, M, d = F_p.shape
        B, N, d = F_q.shape

        # ---------- 全局特征提取 ----------
        def global_feat(feat):
            feat_t = feat.transpose(1, 2)  # (B, d, M)
            f_max = F.max_pool1d(feat_t, kernel_size=feat.shape[1]).squeeze(-1)
            f_avg = F.avg_pool1d(feat_t, kernel_size=feat.shape[1]).squeeze(-1)
            return self.global_mlp(torch.cat([f_max, f_avg], dim=-1))

        f_p_global = global_feat(F_p)  # (B, d)
        f_q_global = global_feat(F_q)  # (B, d)

        # ---------- 对象级门控 ----------
        sim = F.cosine_similarity(f_p_global, f_q_global, dim=-1)
        gate = sim.clamp(min=0.0).view(B, 1, 1)

        # ---------- 双线性交互 ----------
        cross_matrix = gate * torch.bmm(
            f_p_global.unsqueeze(2),
            f_q_global.unsqueeze(1)
        )

        F_p_emb = torch.bmm(F_p, cross_matrix)
        F_q_emb = torch.bmm(F_q, cross_matrix.transpose(1, 2))

        F_p_interact = torch.cat(
            [F_p, F_p_emb * self.alpha.view(1, 1, -1)],
            dim=-1
        )
        F_q_interact = torch.cat(
            [F_q, F_q_emb * self.beta.view(1, 1, -1)],
            dim=-1
        )

        return F_p_interact, F_q_interact



# ============================================================
# Skip Attention Module (Fixed & stable)
# ============================================================

class SkipAttentionModule(nn.Module):
    """
    跳注意力模块（编码器 → 解码器）
    """
    def __init__(self, d_enc, d_dec, attention_mode="cosine", mlp_hidden=64):
        super().__init__()
        self.attention_mode = attention_mode

        if attention_mode == "learnable":
            self.query_mlp = nn.Linear(d_dec, mlp_hidden)
            self.key_mlp = nn.Linear(d_enc, mlp_hidden)
            self.value_mlp = nn.Linear(d_enc, d_dec)

        elif attention_mode == "cosine":
            # 必须是参数层，不能在 forward 里 new
            self.enc_proj = nn.Conv1d(d_enc, d_dec, 1, bias=False)

        else:
            raise ValueError("attention_mode must be 'learnable' or 'cosine'")

    def forward(self, F_enc, F_dec):
        B, M, _ = F_enc.shape
        B, M_dec, _ = F_dec.shape
        assert M == M_dec

        if self.attention_mode == "learnable":
            Q = self.query_mlp(F_dec)
            K = self.key_mlp(F_enc)
            attn = torch.bmm(Q, K.transpose(1, 2)) / (K.shape[-1] ** 0.5)
            attn = F.softmax(attn, dim=-1)
            V = self.value_mlp(F_enc)
            enc_fused = torch.bmm(attn, V)

        else:  # cosine
            F_enc_n = F.normalize(F_enc, dim=-1)
            F_dec_n = F.normalize(F_dec, dim=-1)

            F_enc_proj = self.enc_proj(
                F_enc_n.transpose(1, 2)
            ).transpose(1, 2)

            attn = torch.bmm(F_dec_n, F_enc_proj.transpose(1, 2))
            attn = F.softmax(attn, dim=-1)
            enc_fused = torch.bmm(attn, F_enc_proj)

        return F_dec + enc_fused


# ============================================================
# Simple sanity check
# ============================================================

if __name__ == "__main__":
    B, M = 2, 4096
    d = 64
    d_dec = 128

    F_p = torch.randn(B, M, d)
    F_q = torch.randn(B, M, d)

    fim = FeatureInteractionModule(d)
    F_p_i, F_q_i = fim(F_p, F_q)
    print("FIM:", F_p_i.shape, F_q_i.shape)  # (B, M, 2d)

    enc = torch.randn(B, M, 2 * d)
    dec = torch.randn(B, M, d_dec)

    sa = SkipAttentionModule(2 * d, d_dec, attention_mode="cosine")
    out = sa(enc, dec)
    print("SkipAttn:", out.shape)
