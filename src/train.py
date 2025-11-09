# ============================================================
# train.py (支持相对位置编码)
# ============================================================

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Transformer
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import logging

# -----------------------------
# 1. 超参数设置
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping threshold")
    parser.add_argument("--log-file", type=str, default="training.log")
    parser.add_argument("--max-train-samples", type=int, default=None, help="可选限制训练样本数量")
    parser.add_argument("--relative-position", action="store_true",
                        help="是否使用相对位置编码")
    return parser.parse_args()

# -----------------------------
# 日志配置
# -----------------------------
def setup_logger(save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_file)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# -----------------------------
# 2. 数据加载 & collate_fn
# -----------------------------
def collate_fn(batch, sp, device="cpu"):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_ids = [sp.bos_id()] + sp.encode(item["src"], out_type=int) + [sp.eos_id()]
        tgt_ids = [sp.bos_id()] + sp.encode(item["tgt"], out_type=int) + [sp.eos_id()]
        src_batch.append(torch.tensor(src_ids, dtype=torch.long, device=device))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long, device=device))
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=sp.pad_id())
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=sp.pad_id())
    return src_batch, tgt_batch

def load_datasets(device):
    base_path = os.path.join("data", "processed")
    train_path = os.path.join(base_path, "train.pt")
    valid_path = os.path.join(base_path, "valid.pt")
    test_path = os.path.join(base_path, "test.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"未找到 {train_path}，请先运行数据预处理脚本！")

    train_data = torch.load(train_path, weights_only=False)
    valid_data = torch.load(valid_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    sp_model_path = "data/tokenizer/iwslt_bpe.model"
    if not os.path.exists(sp_model_path):
        raise FileNotFoundError(f"未找到 SentencePiece 模型：{sp_model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    print(f"✅ Datasets loaded: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    return train_data, valid_data, test_data, sp

# -----------------------------
# 3. Noam 学习率调度器
# -----------------------------
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

# -----------------------------
# 4. BLEU 计算
# -----------------------------
def calculate_bleu(model, dataloader, sp, device, max_len=100, max_samples=None, print_samples=1, logger=None):
    model.eval()
    refs, hyps = [], []
    smooth_fn = SmoothingFunction().method4
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    unk_id = sp.unk_id() if hasattr(sp, "unk_id") else None

    sample_printed = 0
    total_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            preds = model.generate(src, sp=sp, max_len=max_len, device=device)

            tgt = tgt.cpu().tolist()
            for ref_ids, pred_ids in zip(tgt, preds):
                def filter_ids(ids):
                    filtered = [int(i) for i in ids if i not in (pad_id, bos_id)]
                    if unk_id is not None:
                        filtered = [i for i in filtered if i != unk_id]
                    if eos_id in filtered:
                        idx = filtered.index(eos_id)
                        filtered = filtered[:idx + 1]
                    return filtered

                ref_f = filter_ids(ref_ids)
                pred_f = filter_ids(pred_ids)
                ref_text = sp.decode_ids(ref_f).strip()
                pred_text = sp.decode_ids(pred_f).strip()
                ref_tokens = ref_text.split()
                pred_tokens = pred_text.split()
                if len(ref_tokens) == 0:
                    continue
                refs.append([ref_tokens])
                hyps.append(pred_tokens)

                if sample_printed < print_samples:
                    msg = f"=== BLEU DEBUG SAMPLE ===\nREF : {ref_text}\nHYP : {pred_text}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    sample_printed += 1

                total_processed += 1
                if max_samples and total_processed >= max_samples:
                    break
            if max_samples and total_processed >= max_samples:
                break

    if len(hyps) == 0:
        return 0.0
    return corpus_bleu(refs, hyps, smoothing_function=smooth_fn) * 100.0

# -----------------------------
# 5. 训练函数
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, clip):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
      ![](../result/epoch_time.png)  optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # ✅ 仅在使用学习率调度器时才执行
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# 6. 设置随机种子
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# 7. 主流程
# -----------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.save_dir, args.log_file)
    logger.info(f"🚀 Using device: {device}")
    
    # 打印参数
    logger.info("📋 Using arguments:")
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")

    train_data, valid_data, test_data, sp = load_datasets(device)
    if args.max_train_samples is not None:
        train_data = train_data[:args.max_train_samples]
        logger.info(f"⚠️ 训练样本已限制为前 {len(train_data)} 条")

    vocab_size = len(sp)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, sp, device))
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, sp, device))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, sp, device))

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        use_relative_position=args.relative_position
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())

    best_valid_loss = float("inf")
    best_model_path = None
    train_losses, valid_losses, bleu_scores, epoch_times, lr_records = [], [], [], [], []

    logger.info("🚀 Starting training...")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.clip)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        bleu = calculate_bleu(model, valid_loader, sp, device, max_len=args.max_seq_len, logger=logger)
        elapsed = time.time() - start

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        bleu_scores.append(bleu)
        epoch_times.append(elapsed / 60.0)
        # 记录当前学习率（如果用 scheduler）
        current_lr = optimizer.param_groups[0]["lr"]
        lr_records.append(current_lr)

        logger.info(f"Epoch [{epoch}/{args.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | "
                    f"BLEU: {bleu:.2f} | LR: {current_lr:.6f} | Time: {elapsed / 60:.2f} min")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = os.path.join(args.save_dir, f"best_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"💾 Saved best model to {best_model_path}")

    # -----------------------------
    # ✅ 绘图部分（报告可视化）
    # -----------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-muted")

    # 1️⃣ Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(valid_losses, label="Valid Loss", marker="s")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    plt.close()

    # 2️⃣ BLEU 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(bleu_scores, color="tab:blue", marker="^")
    plt.title("Validation BLEU Score Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "bleu_curve.png"))
    plt.close()

    # 3️⃣ 学习率变化曲线
    plt.figure(figsize=(8, 5))
    plt.plot(lr_records, color="tab:green")
    plt.title("Learning Rate Schedule (Noam)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "lr_curve.png"))
    plt.close()

    # 4️⃣ 每轮耗时
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(epoch_times) + 1), epoch_times, color="tab:orange")
    plt.title("Training Time per Epoch (minutes)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "epoch_time.png"))
    plt.close()

    # 5️⃣ 综合性能对比
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(train_losses, color="tab:red", label="Train Loss", marker="o")
    ax1.plot(valid_losses, color="tab:purple", label="Valid Loss", marker="s")
    ax2.plot(bleu_scores, color="tab:blue", label="BLEU Score", marker="^")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("BLEU")
    ax1.set_title("Overall Performance Summary")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "performance_summary.png"))
    plt.close()

    logger.info("📊 Saved all visualizations (loss, BLEU, LR, time, performance summary).")

    # -----------------------------
    # ✅ 测试集评估
    # -----------------------------
    if best_model_path and os.path.exists(best_model_path):
        logger.info("🔍 Loading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        test_loss = evaluate(model, test_loader, criterion, device)
        test_bleu = calculate_bleu(model, test_loader, sp, device, max_len=args.max_seq_len, logger=logger)
        logger.info(f"🏁 Test Results | Loss: {test_loss:.4f} | BLEU: {test_bleu:.2f}")
    else:
        logger.warning("⚠️ 未找到最佳模型文件，跳过测试阶段。")

    logger.info("✅ All training and evaluation finished successfully!")


if __name__ == "__main__":
    main()
