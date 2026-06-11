import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple  # 3D卷积用_triple（替代2D的_pair）


class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1,
                 bias=False):
        super(wConv3d, self).__init__()
        # 修正1：用_triple处理3D参数（确保kernel_size/stride等是3维）
        self.stride = _triple(stride)
        self.padding = _triple(padding)  # 3D padding: (depth_pad, height_pad, width_pad)
        self.kernel_size = _triple(kernel_size)  # 3D kernel: (kd, kh, kw)
        self.groups = groups
        self.dilation = _triple(dilation)

        # 修正2：权重形状为5维（符合3D卷积要求）：(out_channels, in_channels//groups, kd, kh, kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 构造3D的Phi张量（原逻辑扩展到3维）
        device = torch.device('cpu')
        # alfa形状：(L,)，L = len(den)*2 + 1（示例中den=[0.5,0.75]，L=5）
        self.register_buffer('alfa', torch.cat([
            torch.tensor(den, device=device),
            torch.tensor([1.0], device=device),
            torch.flip(torch.tensor(den, device=device), dims=[0])
        ]))

        # 修正3：Phi从2D扩展为3D（kd, kh, kw），保持原权重分布逻辑（三维乘积）
        # 方法：通过unsqueeze扩展维度，实现3D张量的外积
        alfa_3d = self.alfa.unsqueeze(1).unsqueeze(2)  # (L, 1, 1)
        Phi = alfa_3d * alfa_3d.transpose(0, 1) * alfa_3d.transpose(0, 2)  # (L, L, L)
        self.register_buffer('Phi', Phi)

        # 验证Phi形状与3D kernel_size一致
        if self.Phi.shape != self.kernel_size:
            raise ValueError(
                f"Phi shape {self.Phi.shape} must match 3D kernel size {self.kernel_size}")

    def forward(self, x):
        # 将Phi转移到输入设备（如CUDA）
        Phi = self.Phi.to(x.device)
        # 3D权重逐元素相乘（5维张量：out, in/groups, kd, kh, kw）
        weight_Phi = self.weight * Phi
        # 3D卷积前向传播（参数均为3维，匹配输入格式）
        return F.conv3d(
            x, weight_Phi, bias=self.bias,
            stride=self.stride, padding=self.padding,
            groups=self.groups, dilation=self.dilation
        )


# Test wConv3d（验证修正后是否正常运行）
print("Testing wConv3d...")
den = [0.5, 0.75]
# 3D kernel_size设为(5,5,5)（与Phi形状一致，Phi长度=2*2+1=5）
block = wConv3d(
    in_channels=3, out_channels=3,
    kernel_size=5,  # 会被_triple处理为(5,5,5)
    den=den,
    padding=2  # 3D padding=(2,2,2)，保持输入输出的depth/height/width一致
).to('cuda' if torch.cuda.is_available() else 'cpu')

# 3D输入格式：(batch, channels, depth, height, width)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input = torch.rand(1, 3, 8, 32, 32).to(device)  # (1,3,8,32,32)：depth=8, height=32, width=32
output = block(input)

print(f"Device: {device}")
print(f"Input size: {input.size()}")
print(f"Output size: {output.size()}")
print(f"Phi shape: {block.Phi.shape}")
print(f"Weight shape: {block.weight.shape}")
print(f"Weight*Phi shape: {block.weight * block.Phi}.shape")