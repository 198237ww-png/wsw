import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

class wConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=2, groups=1, dilation=1, bias=False):
        super().__init__()
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.kernel_size = _triple(kernel_size)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # 构建 3D Phi
        alfa = torch.tensor(den + [1.0] + den[::-1])
        x, y, z = torch.meshgrid(alfa, alfa, alfa, indexing='ij')
        self.register_buffer('Phi', x * y * z)

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv3d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding,
                        groups=self.groups, dilation=self.dilation)

# 测试
print("Testing wConv3d...")
den = [0.5, 0.75]
block = wConv3d(in_channels=3, out_channels=3, kernel_size=5, den=den).to('cuda')
input = torch.rand(1, 3, 8, 32, 32).to('cuda')  # ✅ 5D
output = block(input)
print("Input size:", input.size())
print("Output size:", output.size())
