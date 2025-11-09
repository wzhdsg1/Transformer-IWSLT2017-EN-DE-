# ============================================================
# src/dataset.py
# ============================================================

import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# collate_fn
# -----------------------------
def collate_fn(batch):
    """
    batch: list of {"src": tensor, "tgt": tensor}
    Pads all sequences to equal length for batching.
    Returns:
        src_batch: (batch_size, src_len)
        tgt_batch: (batch_size, tgt_len)
    """
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch

# -----------------------------
# load datasets
# -----------------------------
def load_datasets(data_dir="data/processed"):
    train_path = os.path.join(data_dir, "train.pt")
    valid_path = os.path.join(data_dir, "valid.pt")
    test_path = os.path.join(data_dir, "test.pt")

    # 直接加载 list of dicts，无需自定义 Dataset 类
    train_dataset = torch.load(train_path)
    valid_dataset = torch.load(valid_path)
    test_dataset = torch.load(test_path)

    print(f"✅ Datasets loaded: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset

# -----------------------------
# create DataLoaders
# -----------------------------
def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader

# -----------------------------
# 测试
# -----------------------------
if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = load_datasets()
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset
    )

    for src, tgt in train_loader:
        print("✅ Batch shapes:", src.shape, tgt.shape)
        break
