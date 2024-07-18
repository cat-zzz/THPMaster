"""
@project: THPMaster
@File   : residual_conv.py
@Desc   : 残差卷积块
@Author : gql
@Date   : 2024/7/13 13:26
"""
from torch import nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    残差卷积块
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, add_relu=True):
        """
        初始化残差卷积块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param add_relu: 是否在残差连接后添加ReLU激活函数
        """
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            # 通常在残差块的最后一个卷积层之后不立即使用ReLU，
            # 而是在将卷积输出和跳跃连接的输出相加之后再应用ReLU。
        )

        # 下采样层（调整输入维度以匹配输出维度）
        self.downsample = nn.Sequential()
        # 有两个因素导致维度不匹配：1是步长，步长大于1会导致尺寸缩小；2是通道数
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.add_relu = add_relu  # 是否添加ReLU激活

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.downsample(identity)  # 调整输入维度
        out += identity  # 残差连接：输入和输出相加
        if self.add_relu:
            out = F.relu(out)  # 在残差连接后应用ReLU激活
        return out


if __name__ == '__main__':
    pass
