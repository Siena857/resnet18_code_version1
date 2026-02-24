import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet18的基础残差块（2层卷积）
    核心思想：通过残差连接（shortcut）解决深度网络的梯度消失问题
    """
    expansion = 1  # 通道扩张系数（ResNet18/34=1，ResNet50+/101+/152+=4）

    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化残差块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 第一个卷积层的步长（控制下采样）
        """
        super(BasicBlock, self).__init__()
        # 主路径：3x3卷积 → BN → ReLU → 3x3卷积 → BN
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化，加速训练、稳定分布
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接（shortcut）：当步长≠1或通道数变化时，用1x1卷积匹配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 残差块输出
        """
        # 主路径前向
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差相加：主路径输出 + 捷径连接输出
        out += self.shortcut(x)
        # 最终激活
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    """
    ResNet18主模型（适配CIFAR-10：32x32输入，10分类）
    结构：初始卷积层 → 4个残差层 → 全局平均池化 → 全连接分类头
    """
    def __init__(self, num_classes=10):
        """
        初始化ResNet18
        :param num_classes: 分类类别数（CIFAR-10默认10）
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64  # 初始通道数

        # 初始卷积层（适配CIFAR-10，去掉原ResNet的7x7卷积和最大池化）
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # 4个残差层（ResNet18标准结构：[2,2,2,2]个BasicBlock）
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化 + 分类头
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建残差层（堆叠多个BasicBlock）
        :param block: 残差块类型（BasicBlock）
        :param out_channels: 输出通道数
        :param num_blocks: 该层包含的残差块数量
        :param stride: 第一个块的步长（控制下采样）
        :return: 包含多个残差块的Sequential
        """
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块可能下采样，其余块步长为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        :param x: 输入图像张量 (batch_size, 3, 32, 32)
        :return: 分类 logits (batch_size, num_classes)
        """
        # 初始卷积层
        out = F.relu(self.bn1(self.conv1(x)))
        # 4个残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全局平均池化（将特征图压缩为1x1）
        out = F.avg_pool2d(out, 4)
        # 展平为向量
        out = out.view(out.size(0), -1)
        # 分类头
        out = self.linear(out)
        return out