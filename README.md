# 🚀 Transformer Neural Machine Translation (EN→DE)
🧠 Transformer 模型实现与训练说明
📋 项目简介

本项目实现了一个 基于 Transformer Encoder–Decoder 架构 的神经机器翻译模型（English → German），支持 相对位置编码（Relative Positional Encoding）。
主要内容包括：

手工实现多头自注意力（Multi-Head Self-Attention）

前馈网络（Position-wise FeedForward）

残差连接 + 层归一化（Residual + LayerNorm）

相对与绝对位置编码机制

BLEU 分数计算与可视化分析

⚙️ 硬件与环境要求
🖥️ 硬件配置
项目	推荐配置
GPU	NVIDIA GeForce RTX 4090 (24GB VRAM)
CPU	Intel i9 / AMD Ryzen 9 
内存	≥ 32 GB
硬盘	≥ 100 GB 可用空间
CUDA 版本	11.8

若使用其他 GPU（如 3090、A100），只需保持相同的 CUDA 版本和 PyTorch 兼容性即可。

🧩 软件环境
📦 安装依赖

建议使用 Python 3.10+ 与 虚拟环境（conda 或 venv）：

# 创建并激活环境
conda create -n transformer python=3.10
conda activate transformer

# 安装依赖
pip install -r requirements.txt


requirements.txt 内容如下：

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
│   ├── train.py              # 训练与验证主脚本            
│
├── data/
│
├── result/                   # 模型权重、曲线与日志输出文件夹
│   ├── loss_curve.png
│   ├── bleu_curve.png
│   ├── learning_rate.png
│   ├── epoch_time.png
│   ├── performance_summary.png
│   └── best_model.pt
│
├── requirements.txt
└── README.md


🚀 运行与复现实验
🎯 单次训练命令

以下命令可在 RTX 4090 上直接复现实验结果：

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


建议使用 --seed 42 以保证可复现性。
所有训练日志和模型文件将自动保存在 result/ 文件夹中。

📊 输出与结果

训练结束后，程序会自动生成以下可视化结果：

图表	文件	说明
训练 & 验证 Loss 曲线	result/loss_curve.png	模型收敛情况
BLEU 分数曲线	result/bleu_curve.png	翻译质量变化
学习率变化曲线	result/learning_rate.png	Noam Scheduler 可视化
每轮训练耗时	result/epoch_time.png	性能分析
综合性能对比图	result/performance_summary.png	实验总结
📈 BLEU 评估

模型训练完成后会在验证集与测试集上自动计算 BLEU 分数：

bleu = calculate_bleu(model, valid_loader, sp, device)


输出格式如下：

Epoch [10/20] | Train Loss: 4.85 | Valid Loss: 4.67 | BLEU: 27.4

🧪 可选实验设置
实验变量	参数名	示例
关闭相对位置编码	移除 --relative-position	baseline
修改学习率	--lr	--lr 3e-4
调整warmup步数	--warmup-steps	--warmup-steps 8000
控制最大样本量	--max-train-samples	--max-train-samples 100000
改变batch size	--batch-size	--batch-size 128
