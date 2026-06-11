import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransformerHybrid(nn.Module):
    """
    卷积-Transformer混合模块（C-TNet）：融合局部特征与全局关联
    即插即用特性：输入源/目标特征+LGP编码，输出融合特征，无需修改内部逻辑
    """
    def __init__(
        self,
        d_model: int = 128,        # 特征维度（需与LGP编码维度一致）
        num_heads: int = 4,        # 注意力头数（文档推荐：d_model需能被num_heads整除）
        conv_kernel_size: int = 1, # 卷积核大小（1D卷积，适配点云特征格式）
        conv_layers: int = 2,      # 卷积层数（文档用2层残差，平衡局部特征提取与效率）
        dropout: float = 0.1       # Dropout概率（防止过拟合）
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个注意力头的维度
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        # 1. 卷积局部特征提取（文档：捕捉局部结构，降维增强鲁棒性）
        self.conv_blocks = nn.ModuleList()
        for _ in range(conv_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=0),  # (B, d_model, N)
                nn.BatchNorm1d(d_model),  # 批量归一化，提升训练稳定性
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # 2. 多头交叉注意力参数（文档公式7、8：Q/K/V投影矩阵）
        self.w_q = nn.Linear(d_model, d_model)  # 源特征→Q（查询）
        self.w_k = nn.Linear(d_model, d_model)  # 目标特征→K（键）
        self.w_v = nn.Linear(d_model, d_model)  # 目标特征→V（值）
        self.attn_dropout = nn.Dropout(dropout)

        # 3. 特征融合MLP（文档公式9：cat(局部特征+全局特征)→融合）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # 输入：局部特征(d_model) + 全局特征(d_model)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """拆分特征为多个注意力头：(B, N, d_model) → (B, num_heads, N, d_k)"""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.d_k).transpose(1, 2)  # 交换N与num_heads维度

    def multi_head_cross_attn(
        self,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
        lgp_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        多头交叉注意力（文档公式7、8）：建立源-目标点云的全局关联
        Args:
            src_feat: 源点云特征 (B, N, d_model)
            tgt_feat: 目标点云特征 (B, M, d_model)
            lgp_encoding: LGP位置编码（源点云）(B, N, d_model)
        Returns:
            global_feat: 全局关联特征 (B, N, d_model)
        """
        B, N, _ = src_feat.shape
        B, M, _ = tgt_feat.shape

        # 1. 卷积提取源点云局部特征（文档：Conv(x_i^P')）
        src_conv = src_feat.transpose(1, 2)  # (B, d_model, N)
        for conv in self.conv_blocks:
            src_conv = conv(src_conv)
        src_conv = src_conv.transpose(1, 2)  # (B, N, d_model)

        # 2. Q/K/V投影（文档公式8：Q = Conv(x_i^P') + α_i^LGP）
        q = self.w_q(src_conv + lgp_encoding)  # (B, N, d_model)：Q融入LGP编码
        k = self.w_k(tgt_feat)  # (B, M, d_model)
        v = self.w_v(tgt_feat)  # (B, M, d_model)

        # 3. 拆分注意力头
        q_split = self.split_heads(q)  # (B, num_heads, N, d_k)
        k_split = self.split_heads(k)  # (B, num_heads, M, d_k)
        v_split = self.split_heads(v)  # (B, num_heads, M, d_k)

        # 4. 计算注意力权重（文档公式8：α_i,j^Cross = (Q·K^T)/√d_k）
        attn_scores = torch.matmul(q_split, k_split.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32).to(q.device))  # (B, num_heads, N, M)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, M)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 注意力输出（文档公式7：z_i^P',Q' = Σsoftmax(α)·V）
        global_feat = torch.matmul(attn_weights, v_split)  # (B, num_heads, N, d_k)
        # 合并注意力头
        global_feat = global_feat.transpose(1, 2).contiguous().view(B, N, self.d_model)  # (B, N, d_model)
        return global_feat

    def forward(
        self,
        src_feat: torch.Tensor,
        tgt_feat: torch.Tensor,
        lgp_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：局部特征+全局关联→融合特征
        Args:
            src_feat: 源点云特征 (B, N, d_model)
            tgt_feat: 目标点云特征 (B, M, d_model)
            lgp_encoding: LGP位置编码（源点云）(B, N, d_model)
        Returns:
            fused_feat: 融合后特征（用于后续配准）(B, N, d_model)
        """
        # 1. 多头交叉注意力：获取全局关联特征
        global_feat = self.multi_head_cross_attn(src_feat, tgt_feat, lgp_encoding)
        # 2. 特征融合：cat(源特征+全局特征) → MLP（文档公式9）
        fused_feat = self.fusion_mlp(torch.cat([src_feat, global_feat], dim=-1))
        return fused_feat