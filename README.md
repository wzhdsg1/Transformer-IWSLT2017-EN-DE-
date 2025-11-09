🚀 Transformer 模型实现与训练说明
🧠 项目简介

本项目实现了一个 基于 Transformer Encoder–Decoder 架构 的神经机器翻译模型（English → German），支持 相对位置编码（Relative Positional Encoding）。

主要特性包括：

✅ 多头自注意力（Multi-Head Self-Attention）

✅ 前馈网络（Position-wise Feed Forward Network）

✅ 残差连接 + 层归一化（Residual + LayerNorm）

✅ 相对与绝对位置编码机制

✅ BLEU 分数计算与可视化分析

⚙️ 硬件与运行环境
💻 硬件配置
项目	推荐配置
GPU	NVIDIA GeForce RTX 4090 (24GB VRAM)
CPU	Intel i9 / AMD Ryzen 9 或更高
内存	≥ 32 GB
硬盘	≥ 100 GB 可用空间
CUDA 版本	11.8

若使用其他 GPU（如 3090、A100），仅需保证 CUDA 兼容性一致即可。

🧩 软件依赖
📦 安装步骤

建议使用 Python 3.10+ 与 conda 环境：

# 创建虚拟环境
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖
pip install -r requirements.txt

📋 requirements.txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
sentencepiece>=0.1.99
nltk>=3.8.1
matplotlib>=3.7.0
tqdm>=4.65.0
loguru>=0.7.0
numpy>=1.23.0
pandas>=2.0.0

📂 项目结构
Transformer_Assignment/
├── src/
│   ├── model.py              # Transformer 模型定义（Encoder、Decoder、Attention等）
│   ├── train.py              # 主训练脚本
│
├── data/
│
├── result/                   # 模型与曲线输出
│   ├── loss_curve.png
│   ├── bleu_curve.png
│   ├── learning_rate.png
│   ├── epoch_time.png
│   ├── performance_summary.png
│   └── best_model.pt
│
├── requirements.txt
└── README.md

🧪 可复现实验命令

以下命令可在 RTX 4090 + CUDA 11.8 环境下直接运行：

python src/train.py \
  --epochs 20 \
  --lr 5e-4 \
  --batch-size 64 \
  --save-dir result \
  --warmup-steps 4000 \
  --max-steps 50000 \
  --max-train-samples 200000 \
  --relative-position \
  --seed 42


✅ 可复现性说明：
实验中使用 --seed 42 固定随机种子，以保证结果在不同环境中一致。
所有训练日志与模型文件将自动保存到 result/ 文件夹中。

📊 结果输出与可视化

训练完成后将自动生成以下结果图表：

图表	文件路径	说明
📈 训练 & 验证 Loss 曲线	result/loss_curve.png	模型收敛趋势
🧾 BLEU 分数曲线	result/bleu_curve.png	翻译性能变化
🧮 学习率变化曲线	result/learning_rate.png	Noam Scheduler 可视化
⏱️ 每轮训练耗时统计	result/epoch_time.png	性能分析
🧠 综合性能对比	result/performance_summary.png	全面实验对比
📈 BLEU 评估示例

训练日志示例输出：

Epoch [10/20] | Train Loss: 4.85 | Valid Loss: 4.67 | BLEU: 27.4 | Time: 2.12 min
Epoch [20/20] | Train Loss: 3.92 | Valid Loss: 3.70 | BLEU: 32.8 | Time: 2.18 min


计算方式：

bleu = calculate_bleu(model, valid_loader, sp, device)

🧮 可选实验变量
实验目标	参数	示例命令
关闭相对位置编码	无 --relative-position	baseline
调整学习率	--lr	--lr 3e-4
增加 warmup 步数	--warmup-steps	--warmup-steps 8000
限制训练样本量	--max-train-samples	--max-train-samples 100000
调整 batch 大小	--batch-size	--batch-size 128
