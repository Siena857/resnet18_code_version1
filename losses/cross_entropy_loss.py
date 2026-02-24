import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失函数（封装PyTorch自带的nn.CrossEntropyLoss）
    常用于分类任务，输入为模型输出logits和真实标签
    """
    def __init__(self, weight=None, reduction='mean'):
        """
        初始化交叉熵损失
        :param weight: 各类别的权重张量，用于不平衡样本
        :param reduction: 损失计算方式（'none'|'mean'|'sum'）
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, logits, targets):
        """
        前向传播计算损失
        :param logits: 模型输出的未归一化概率 (batch_size, num_classes)
        :param targets: 真实标签 (batch_size,)，每个元素是类别索引
        :return: 交叉熵损失值
        """
        return self.loss_fn(logits, targets)