import argparse
import torch
import torch.optim as optim
from models.resnet18 import ResNet18
from datasets.cifar10_dataset import get_cifar10_dataloader
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.trainer import Trainer
from utils.common import setup_logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResNet18 for CIFAR-10 Classification (RILAB招新)')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers (default: 4)')
    # 路径参数
    parser.add_argument('--data_root', type=str, default='./data', help='dataset root path')
    parser.add_argument('--log_dir', type=str, default='./work_dirs/logs', help='log save path')
    parser.add_argument('--checkpoint_dir', type=str, default='./work_dirs/checkpoints', help='model save path')
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device (cuda/cpu, default: auto)')
    return parser.parse_args()

def main():
    # 1. 解析参数
    args = parse_args()
    # 2. 配置日志
    logging = setup_logging(args.log_dir)
    logging.info(f'===== RILAB ResNet18 Training =====')
    logging.info(f'Using device: {args.device}')
    logging.info(f'Args: {args}')

    # 3. 初始化组件
    # 3.1 模型
    model = ResNet18(num_classes=10)
    logging.info(f'Model: ResNet18 (CIFAR-10)')
    # 3.2 数据加载器
    train_loader = get_cifar10_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True
    )
    val_loader = get_cifar10_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False
    )
    test_loader = get_cifar10_dataloader(
    data_root=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    train=False  # CIFAR-10 测试集就是 train=False
)
    logging.info(f'Dataset: CIFAR-10 (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)})')
    # 3.3 损失函数
    criterion = CrossEntropyLoss()
    # 3.4 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # 加权重衰减防止过拟合

    # 4. 启动训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs
    )
    trainer.fit()

if __name__ == '__main__':
    main()