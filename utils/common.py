import logging
import os
import sys
import random
import torch
import numpy as np

def setup_logging(log_dir='./work_dirs', log_file='train.log'):
    """
    设置训练日志：同时输出到控制台和日志文件，方便后续查看训练过程
    :param log_dir: 日志/模型保存根目录
    :param log_file: 日志文件名
    :return: 配置好的logger对象
    """
    # 创建日志目录（不存在则自动创建）
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # 配置日志格式：时间 + 日志级别 + 内容
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),  # 输出到文件
            logging.StreamHandler(sys.stdout)                 # 输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志配置完成，日志文件保存至：{log_path}")
    return logger

def save_checkpoint(model, optimizer, epoch, best_acc, save_dir='./work_dirs', filename='checkpoint.pth'):
    """
    保存模型权重和优化器状态（断点续训/保存最佳模型用）
    :param model: 待保存的模型
    :param optimizer: 优化器（保存学习率等状态）
    :param epoch: 当前训练轮数
    :param best_acc: 当前最佳验证准确率
    :param save_dir: 保存目录
    :param filename: 保存的文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 保存核心信息：模型权重、优化器状态、训练轮数、最佳准确率
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'random_seed': torch.initial_seed()  # 可选：保存随机种子，保证复现
    }
    torch.save(checkpoint, save_path)
    
    # 记录日志
    logger = logging.getLogger(__name__)
    logger.info(f"模型已保存至：{save_path} | Epoch: {epoch} | Best Acc: {best_acc:.2f}%")

def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """
    加载模型权重和优化器状态（断点续训用）
    :param model: 待加载权重的模型
    :param optimizer: 待加载状态的优化器
    :param checkpoint_path: 模型文件路径
    :param device: 加载设备（cuda/cpu）
    :return: 加载后的epoch、best_acc
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在：{checkpoint_path}")
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    # 记录日志
    logger = logging.getLogger(__name__)
    logger.info(f"模型加载完成：{checkpoint_path} | 恢复至 Epoch: {epoch} | Best Acc: {best_acc:.2f}%")
    return epoch, best_acc

def set_random_seed(seed=42):
    """
    设置全局随机种子，保证实验可复现
    :param seed: 随机种子（默认42，可自定义）
    """
    random.seed(seed)          # Python原生随机数
    np.random.seed(seed)       # NumPy随机数
    torch.manual_seed(seed)    # CPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 单GPU随机数
        torch.cuda.manual_seed_all(seed)    # 多GPU随机数
        torch.backends.cudnn.deterministic = True  # 固定cudnn算法
        torch.backends.cudnn.benchmark = False     # 关闭自动优化（保证复现）
    
    logger = logging.getLogger(__name__)
    logger.info(f"全局随机种子已设置为：{seed}")