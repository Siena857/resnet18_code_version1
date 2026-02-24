import argparse
import torch
from models.resnet18 import ResNet18
from datasets.cifar10_dataset import get_cifar10_dataloader
from losses.cross_entropy_loss import CrossEntropyLoss
from utils.common import setup_logging

def parse_args():
    """解析测试参数"""
    parser = argparse.ArgumentParser(description='Test ResNet18 on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--data_root', type=str, default='./data', help='dataset root path')
    parser.add_argument('--checkpoint', type=str, default='./work_dirs/checkpoints/best_model.pth',
                        help='best model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device (cuda/cpu)')
    return parser.parse_args()

def main():
    # 1. 解析参数 + 配置日志
    args = parse_args()
    logging = setup_logging('./work_dirs/logs')
    logging.info(f'===== RILAB ResNet18 Testing =====')
    logging.info(f'Using device: {args.device}')
    logging.info(f'Checkpoint path: {args.checkpoint}')

    # 2. 加载模型
    model = ResNet18(num_classes=10).to(args.device)
    # 加载最佳权重
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f'Loaded model from epoch {checkpoint["epoch"]} (best acc: {checkpoint["best_acc"]:.2f}%)')

    # 3. 加载测试集
    test_loader = get_cifar10_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False
    )
    logging.info(f'Test dataset size: {len(test_loader.dataset)}')

    # 4. 测试
    model.eval()
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 统计结果
    avg_loss = total_loss / total
    avg_acc = 100. * correct / total
    logging.info(f'===== Test Result =====')
    logging.info(f'Average Loss: {avg_loss:.4f}')
    logging.info(f'Average Accuracy: {avg_acc:.2f}%')

if __name__ == '__main__':
    main()