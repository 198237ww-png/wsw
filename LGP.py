import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
import numpy as np


class LocalGeometricPerception(nn.Module):
    """
    局部几何感知机制（LGP）：动态生成位置编码，适配点云局部密度与方向特征
    即插即用特性：输入点云坐标，输出位置编码，无需依赖外部模块
    """

    def __init__(
            self,
            d_model: int = 128,  # 位置编码维度（需与点云特征维度一致）
            k: int = 20,  # 邻域搜索点数（文档推荐：文化遗产点云k=15~30）
            sigma: float = 0.1,  # 高斯核标准差（调节邻域范围，文档公式3）
            ref_dir: list = [0, 0, 1],  # 参考方向β（文档公式4，默认Z轴，适配文物方向特征）
            use_torch_cluster: bool = True  # 是否用torch_cluster加速邻域搜索（推荐GPU）
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.sigma = sigma
        self.ref_dir = torch.tensor(ref_dir, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,3)
        self.use_torch_cluster = use_torch_cluster

    def get_neighbor_indices(self, points: torch.Tensor) -> torch.Tensor:
        """
        邻域搜索：获取每个点的k个邻域索引（适配GPU）
        Args:
            points: 点云坐标 (B, N, 3)
        Returns:
            neighbor_indices: 邻域索引 (B, N, k)
        """
        B, N, _ = points.shape
        neighbor_indices = []

        for b in range(B):
            if self.use_torch_cluster:
                # 用torch_cluster的KNNGraph快速搜索（GPU加速）
                graph = knn_graph(points[b], k=self.k, loop=False)  # (2, N*k)，格式：[目标点索引, 源点索引]
                # 整理为(B, N, k)格式
                idx = graph[1].view(N, self.k)  # (N, k)
            else:
                # 备用：numpy暴力搜索（无torch_cluster时使用，仅CPU）
                dist = torch.cdist(points[b], points[b])  # (N, N)，计算所有点对距离
                _, idx = torch.topk(-dist, k=self.k + 1, dim=-1)  # 取距离最近的k+1个（排除自身）
                idx = idx[:, 1:]  # (N, k)，排除自身

            neighbor_indices.append(idx)

        return torch.stack(neighbor_indices, dim=0).to(points.device)  # (B, N, k)

    def compute_local_density(self, points: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        计算局部密度ψ_i（文档公式2、3）：适配点云密度差异
        Args:
            points: 点云坐标 (B, N, 3)
            neighbor_indices: 邻域索引 (B, N, k)
        Returns:
            local_density: 局部密度 (B, N, 1)
        """
        B, N, k = neighbor_indices.shape
        # 提取邻域点坐标：(B, N, k, 3)
        neighbor_points = torch.gather(
            points.unsqueeze(2).expand(B, N, N, 3),  # (B, N, N, 3)
            dim=2,
            index=neighbor_indices.unsqueeze(-1).expand(B, N, k, 3)  # (B, N, k, 3)
        )
        # 计算点与邻域点的距离 r = ||p_i - p_j||
        dist = torch.norm(points.unsqueeze(2) - neighbor_points, dim=-1)  # (B, N, k)
        # 高斯核函数 K(r) = e^(-r²/(2σ²))（文档公式3）
        gaussian_kernel = torch.exp(-(dist ** 2) / (2 * self.sigma ** 2))  # (B, N, k)
        # 局部密度 ψ_i = ΣK(r)（文档公式2）
        local_density = torch.sum(gaussian_kernel, dim=-1, keepdim=True)  # (B, N, 1)
        return local_density

    def compute_normal_vector(self, points: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        计算法向量f_i（文档公式4）：基于PCA，适配不规则点云
        Args:
            points: 点云坐标 (B, N, 3)
            neighbor_indices: 邻域索引 (B, N, k)
        Returns:
            normals: 法向量 (B, N, 3)
        """
        B, N, k = neighbor_indices.shape
        normals = torch.zeros_like(points)  # (B, N, 3)

        for b in range(B):
            # 提取每个点的邻域点：(N, k, 3)
            neighbor_points = points[b][neighbor_indices[b]]  # (N, k, 3)
            # 邻域点中心化（减去邻域均值）
            neighbor_mean = neighbor_points.mean(dim=1, keepdim=True)  # (N, 1, 3)
            centered = neighbor_points - neighbor_mean  # (N, k, 3)
            # 计算协方差矩阵：(N, 3, 3)
            cov = torch.bmm(centered.transpose(1, 2), centered) / (k - 1)  # (N, 3, 3)
            # 求协方差矩阵的特征值与特征向量：最小特征值对应法向量
            eig_vals, eig_vecs = torch.linalg.eig(cov)  # 特征值(复数)，需转实数
            eig_vals = eig_vals.real  # (N, 3)
            eig_vecs = eig_vecs.real  # (N, 3, 3)
            # 取最小特征值对应的特征向量作为法向量
            min_eig_idx = torch.argmin(eig_vals, dim=-1)  # (N,)
            normals[b] = eig_vecs[torch.arange(N), :, min_eig_idx]  # (N, 3)

        # 法向量方向统一（与参考方向一致，避免符号歧义）
        ref_dir = self.ref_dir.to(points.device)  # (1,1,3)
        dot = torch.sum(normals * ref_dir, dim=-1, keepdim=True)  # (B, N, 1)
        normals = torch.where(dot < 0, -normals, normals)  # 反向法向量取反
        return normals

    def compute_angle_encoding(self, normals: torch.Tensor) -> torch.Tensor:
        """
        几何角度编码η_i（文档公式4、5）：捕捉点云方向特征
        Args:
            normals: 法向量 (B, N, 3)
        Returns:
            angle_encoding: 角度编码 [sinθ_i, cosθ_i] (B, N, 2)
        """
        ref_dir = self.ref_dir.to(normals.device)  # (1,1,3)
        # 计算法向量与参考方向的夹角 θ_i = arccos(f_i · β)（文档公式4）
        dot = torch.clamp(torch.sum(normals * ref_dir, dim=-1, keepdim=True), -1.0, 1.0)  # (B, N, 1)
        theta = torch.acos(dot)  # (B, N, 1)
        # 角度编码 η_i = [sinθ_i, cosθ_i]（文档公式5）
        angle_encoding = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B, N, 2)
        return angle_encoding

    def dynamic_fusion(self, points: torch.Tensor, local_density: torch.Tensor,
                       angle_encoding: torch.Tensor) -> torch.Tensor:
        """
        动态融合位置编码（文档公式6）：关联密度与角度信息
        Args:
            points: 点云坐标 (B, N, 3)
            local_density: 局部密度 (B, N, 1)
            angle_encoding: 角度编码 (B, N, 2)
        Returns:
            lgp_encoding: 动态位置编码 (B, N, d_model)
        """
        B, N, _ = points.shape
        D = 3  # 点云数据维度（X/Y/Z）
        d = torch.arange(self.d_model, dtype=torch.float32).to(points.device)  # (d_model,)

        # 计算pos_i = √(x_i² + y_i² + z_i²)（文档公式6）
        pos = torch.norm(points, dim=-1, keepdim=True)  # (B, N, 1)

        # 广播维度：适配d_model
        pos_expand = pos.expand(B, N, self.d_model)  # (B, N, d_model)
        d_expand = d.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # (B, N, d_model)
        psi_expand = local_density.expand(B, N, self.d_model)  # (B, N, d_model)

        # 文档公式6：分母 10000^(2d/D · ψ_i)
        denominator = 10000 ** (2 * d_expand / D * psi_expand)  # (B, N, d_model)
        # 角度项：pos_i/denominator + η_i（η_i重复适配d_model维度）
        eta_expand = angle_encoding.unsqueeze(-1).repeat(1, 1, 1, self.d_model // 2)  # (B, N, 2, d_model//2)
        eta_expand = eta_expand.reshape(B, N, self.d_model)  # (B, N, d_model)
        angle_term = pos_expand / denominator + eta_expand  # (B, N, d_model)

        # 动态位置编码：sin(angle_term) + cos(angle_term)
        lgp_encoding = torch.sin(angle_term) + torch.cos(angle_term)  # (B, N, d_model)
        return lgp_encoding

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播：点云坐标 → 动态位置编码
        Args:
            points: 点云坐标 (B, N, 3)，B=批次大小，N=点数，3=X/Y/Z
        Returns:
            lgp_encoding: LGP位置编码 (B, N, d_model)，可直接叠加到点云特征
        """
        # 1. 邻域搜索
        neighbor_indices = self.get_neighbor_indices(points)
        # 2. 计算局部密度（LDI）
        local_density = self.compute_local_density(points, neighbor_indices)
        # 3. 计算法向量与角度编码（GAE）
        normals = self.compute_normal_vector(points, neighbor_indices)
        angle_encoding = self.compute_angle_encoding(normals)
        # 4. 动态融合生成编码
        lgp_encoding = self.dynamic_fusion(points, local_density, angle_encoding)
        return lgp_encoding