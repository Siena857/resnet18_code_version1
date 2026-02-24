# 导入核心依赖库
import torch  # PyTorch核心库，用于张量计算和模型构建
import torch.nn as nn  # 神经网络模块，包含各类层和损失函数
import torch.optim as optim  # 优化器模块，如SGD、Adam
from tqdm import tqdm  # 进度条库，可视化训练/验证进度
import os  # 系统路径操作，用于创建保存目录
import logging  # 日志模块，记录训练过程

class EarlyStopping:
    """早停类，实现保存验证集表现最佳的模型权重功能"""
    def __init__(self, save_path, patience=5, verbose=False, delta=0):
        self.patience = patience  # 验证损失不下降时，最多等待多少轮
        self.verbose = verbose    # 是否打印日志
        self.counter = 0          # 等待轮数计数器
        self.best_score = None    # 最佳分数（用负的验证损失表示）
        self.early_stop = False   # 是否触发早停
        self.val_loss_min = float('inf')  # 最小验证损失
        self.delta = delta        # 损失下降的最小阈值（小于这个值视为无提升）
        self.save_path = save_path  # 最佳模型保存路径

    def __call__(self, val_loss, model):
        # 用负损失作为分数（因为我们要最大化分数，等价于最小化损失）
        score = -val_loss

        if self.best_score is None:
            # 第一轮初始化：保存第一个模型，记录最佳分数
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 验证损失无显著下降，计数器+1
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 验证损失下降，更新最佳分数，保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, val_loss, model):
        """保存当前最佳模型"""
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}).  Saving model ...')
        # 保存模型权重（包含模型参数+优化器状态，方便断点续训）
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, self.save_path)
        self.val_loss_min = val_loss

class Trainer:
    """
    ResNet18训练器类（适配CIFAR-10分类任务）
    核心功能：封装训练、验证、测试全流程，简化训练脚本的调用逻辑
    """
    def __init__(
        self,
        model,               # 待训练的模型（如ResNet18实例）
        train_loader,        # 训练集DataLoader，提供批次化训练数据
        val_loader,          # 验证集DataLoader，评估模型泛化能力
        test_loader=None,    # 测试集DataLoader（改为可选，默认None）
        criterion=None,      # 损失函数（改为可选，默认None）
        optimizer=None,      # 优化器（改为可选，默认None）
        device='cuda',       # 训练设备：'cuda'（GPU）/'cpu'（CPU）
        save_dir='./work_dirs',  # 模型/日志保存目录
        epochs=50            # 总训练轮数
    ):
        # 初始化模型并移至指定设备（GPU/CPU）
        self.model = model.to(device)
        # 初始化数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # 初始化损失函数和优化器（添加默认值兜底）
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # 训练配置
        self.device = device  # 训练设备
        self.save_dir = save_dir  # 保存目录
        self.epochs = epochs  # 总训练轮数
        # 初始化日志器（复用全局日志配置）
        self.logger = logging.getLogger(__name__)
        # 创建保存目录（不存在则自动创建）
        os.makedirs(save_dir, exist_ok=True)
        self.best_model_path = os.path.join(save_dir, 'best_model.pth')  # 最佳权重保存路径
        self.early_stopper = EarlyStopping(save_path=self.best_model_path, patience=5, verbose=True)

    def train_one_epoch(self, epoch):
        """
        训练单个Epoch（一轮完整的训练集遍历）
        :param epoch: 当前训练轮数（从0开始计数）
        :return: 该Epoch的平均训练损失、训练准确率
        """
        # 将模型设为训练模式：启用Dropout、BatchNorm更新等
        self.model.train()
        # 初始化统计变量
        total_loss = 0.0  # 累计总损失
        correct = 0       # 累计正确预测数
        total = 0         # 累计样本总数

        # 创建进度条，可视化训练进度
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        # 遍历训练集批次
        for inputs, targets in pbar:
            # 将数据移至指定设备（GPU/CPU）
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 清空优化器梯度（避免梯度累积）
            self.optimizer.zero_grad()
            
            # 前向传播：模型预测
            outputs = self.model(inputs)
            # 计算损失（预测值 vs 真实标签）
            loss = self.criterion(outputs, targets)
            # 反向传播：计算梯度
            loss.backward()
            # 优化器更新模型参数
            self.optimizer.step()

            # 累计损失（乘以批次大小，保证平均损失计算准确）
            total_loss += loss.item() * inputs.size(0)
            # 计算预测结果：取输出概率最大的类别
            _, predicted = outputs.max(1)
            # 累计样本数和正确数
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条显示：实时损失、准确率
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',  # 当前批次损失
                'Acc': f'{100.*correct/total:.2f}%'  # 累计准确率
            })

        # 计算该Epoch的平均损失和准确率
        avg_loss = total_loss / total
        avg_acc = 100. * correct / total
        # 记录日志：保存该Epoch的训练指标
        self.logger.info(f"Epoch {epoch+1} Train - Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
        return avg_loss, avg_acc

    @torch.no_grad()  # 装饰器：禁用梯度计算，节省显存、加速计算
    def validate(self, epoch, loader, desc='Val'):
        """
        验证/测试函数（通用逻辑，适配验证集/测试集）
        :param epoch: 当前训练轮数（用于日志显示）
        :param loader: 验证集/测试集DataLoader
        :param desc: 日志显示前缀（Val/Test）
        :return: 平均损失、准确率
        """
        # 将模型设为评估模式：禁用Dropout、固定BatchNorm统计值
        self.model.eval()
        # 初始化统计变量
        total_loss = 0.0
        correct = 0
        total = 0

        # 创建进度条，可视化验证/测试进度
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.epochs} [{desc}]')
        # 遍历数据批次（无梯度计算）
        for inputs, targets in pbar:
            # 将数据移至指定设备
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 前向传播：模型预测
            outputs = self.model(inputs)
            # 计算损失
            loss = self.criterion(outputs, targets)

            # 累计统计值
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条显示
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        # 计算平均损失和准确率
        avg_loss = total_loss / total
        avg_acc = 100. * correct / total
        # 记录日志
        self.logger.info(f"Epoch {epoch+1} {desc} - Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
        return avg_loss, avg_acc

    def fit(self):
        """
        完整训练流程（核心入口函数）
        逻辑：遍历所有Epoch → 训练 → 验证 → 训练结束后测试
        """
        # 记录训练开始日志
        self.logger.info(f"开始训练：总Epoch={self.epochs} | 设备={self.device}")
        
        # 遍历每个训练轮数
        for epoch in range(self.epochs):
            # 训练一个Epoch
            self.train_one_epoch(epoch)
            # 验证当前Epoch的模型性能（接收返回值：验证损失+准确率）
            val_loss, val_acc = self.validate(epoch, self.val_loader, desc='Val')

            # 调用EarlyStopping，自动保存最佳权重
            self.early_stopper(val_loss, self.model)
            
            # 如果触发早停，提前结束训练
            if self.early_stopper.early_stop:
                self.logger.info("Early stopping triggered!")
                break

        # 训练结束后，仅执行一次测试集评估（如果有测试集）
        if self.test_loader is not None:
            self.logger.info("训练结束，进行测试集评估")
            self.validate(0, self.test_loader, desc='Test')
        # 打印最佳模型路径
        self.logger.info(f"最佳模型权重已保存至：{self.best_model_path}")

# 为Trainer类添加train别名方法（兼容train()调用）
def train(self):
    """兼容 train() 调用，等价于fit()"""
    return self.fit()

# 将train方法绑定到Trainer类
Trainer.train = train