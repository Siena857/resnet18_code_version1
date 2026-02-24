# ResNet18 图像分类（CIFAR-10）
基于 PyTorch 实现的 ResNet18 模型，针对 CIFAR-10 数据集完成图像分类任务，采用 `argparse` 管理超参数（无需配置文件）。

## 项目简介
- 模型：ResNet18（残差网络），适配 CIFAR-10 数据集的输入尺寸与类别数
- 超参数管理：使用 `argparse` 命令行传参替代配置文件（`configs/` 文件夹为空）
- 数据集：CIFAR-10（torchvision 自动下载，无需手动处理）
- 训练框架：PyTorch 2.0+

## 环境准备
### 1. 克隆项目

git clone https://github.com/Siena857/resnet18_code_version1.git
cd resnet18_code_version1

### 2. 创建并激活虚拟环境

Windows 系统（PowerShell/CMD）
# 创建虚拟环境（Python 3.11 为例）
python -m venv .venv311
# 激活虚拟环境（PowerShell）
.venv311\Scripts\Activate.ps1
# 若报错“执行策略禁止”，先执行：Set-ExecutionPolicy RemoteSigned -Scope CurrentUser（按提示输入 Y 确认）
# 激活虚拟环境（CMD）
.venv311\Scripts\activate.bat

macOS/Linux 系统
# 创建虚拟环境
python3 -m venv .venv311
# 激活虚拟环境
source .venv311/bin/activate

### 3. 安装依赖包
pip install -r requirements.txt
# 若下载慢，可添加清华镜像源：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

快速开始
1. 查看所有可配置超参数
通过以下命令查看所有支持的超参数及默认值，无需翻阅代码：
运行
python train.py -h   或 python train.py --help
2. 训练模型
项目无配置文件，所有超参数通过命令行传入，示例如下：
基础训练（使用默认超参数）

运行
python train.py

自定义超参数训练（常用参数示例）
运行
python train.py \
  --epochs 100 \          # 训练总轮数
  --batch_size 64 \       # 批次大小
  --lr 0.001 \            # 初始学习率
  --data_path ./data \    # 数据集下载/保存路径
  --device cuda \         # 训练设备（cuda/cpu，优先使用GPU）
  --num_classes 10 \      # 分类类别数（CIFAR-10 固定为10）
  --weight_decay 1e-4     # 权重衰减（防止过拟合）
3. 测试模型
训练完成后，加载最优权重测试模型性能：
bash
运行
# 基础测试（使用默认权重路径）
python test.py

# 自定义测试参数
python test.py \
  --model_path ./best_resnet18.pth \  # 训练好的模型权重路径
  --device cuda \                     # 测试设备
  --batch_size 64                     # 测试批次大小
项目结构
plaintext
resnet18_code_version1/
├── configs/                  # 配置文件夹（空）：采用 argparse 命令行传参替代配置文件
├── losses/                   # 自定义损失函数（如交叉熵、标签平滑损失）
├── models/                   # ResNet18 模型核心定义
│   └── resnet18.py           # ResNet18 网络结构实现
├── trainers/                 # 训练逻辑封装（含数据加载、训练循环、验证逻辑）
├── utils/                    # 工具函数（日志打印、指标计算、模型保存等）
├── .gitignore                # Git 忽略规则（排除虚拟环境、数据集、权重等）
├── best_resnet18.pth         # 训练生成的最优模型权重（不纳入版本控制）
├── requirements.txt          # 项目依赖清单
├── train.py                  # 训练入口（含 argparse 超参数解析）
├── test.py                   # 测试入口（含 argparse 超参数解析）
└── README.md                 # 项目说明文档
