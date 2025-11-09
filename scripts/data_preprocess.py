# ============================================================
# scripts/data_preprocess.py
# Preprocessing for IWSLT17 English→German Translation
# - Load train/dev/test from local files
# - Train SentencePiece tokenizer
# - Tokenize and save as safe list[dict] datasets
# ============================================================

import os
import torch
import sentencepiece as spm
import xml.etree.ElementTree as ET

# ============================================================
# 1. Paths & Config
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Transformer_Assignment/
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "iwslt17_en_de/en-de-data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

VOCAB_SIZE = 16000
MAX_TRAIN_SAMPLES = 200000  # ✅ 改为 10 万
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# 2. Read train files
# ============================================================

def read_train_file(src_path, tgt_path):
    """读取训练文件，去掉 <...> 标签行"""
    pairs = []
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_line, tgt_line = src_line.strip(), tgt_line.strip()
            if not src_line or not tgt_line:
                continue
            if src_line.startswith("<") or tgt_line.startswith("<"):
                continue
            pairs.append({"src": src_line, "tgt": tgt_line})
    print(f"📖 Read {len(pairs)} lines from {os.path.basename(src_path)} / {os.path.basename(tgt_path)}")
    return pairs

# ============================================================
# 3. Parse XML dev/test files
# ============================================================

def parse_xml_file(xml_path_en, xml_path_de):
    """解析XML文件，提取 <seg> 标签"""
    pairs = []
    tree_en = ET.parse(xml_path_en)
    tree_de = ET.parse(xml_path_de)

    segs_en = tree_en.findall(".//seg")
    segs_de = tree_de.findall(".//seg")

    if len(segs_en) != len(segs_de):
        raise ValueError(f"Segment count mismatch: {xml_path_en} vs {xml_path_de}")

    for en_seg, de_seg in zip(segs_en, segs_de):
        en_text = en_seg.text.strip()
        de_text = de_seg.text.strip()
        if en_text and de_text:
            pairs.append({"src": en_text, "tgt": de_text})

    print(f"📖 Parsed {len(pairs)} segments from {os.path.basename(xml_path_en)} / {os.path.basename(xml_path_de)}")
    return pairs

# ============================================================
# 4. Load datasets
# ============================================================

def load_iwslt17_local(max_train_samples=None):
    train_pairs = read_train_file(
        os.path.join(RAW_DIR, "train.tags.en-de.en"),
        os.path.join(RAW_DIR, "train.tags.en-de.de")
    )
    if max_train_samples:
        train_pairs = train_pairs[:max_train_samples]
        print(f"⚠️ Training samples limited to {len(train_pairs)}")

    val_pairs = parse_xml_file(
        os.path.join(RAW_DIR, "IWSLT17.TED.dev2010.en-de.en.xml"),
        os.path.join(RAW_DIR, "IWSLT17.TED.dev2010.en-de.de.xml")
    )

    test_pairs = parse_xml_file(
        os.path.join(RAW_DIR, "IWSLT17.TED.tst2010.en-de.en.xml"),
        os.path.join(RAW_DIR, "IWSLT17.TED.tst2010.en-de.de.xml")
    )

    print(f"✅ Loaded datasets: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    return train_pairs, val_pairs, test_pairs

# ============================================================
# 5. Train SentencePiece tokenizer
# ============================================================

def train_tokenizer(pairs, vocab_size=VOCAB_SIZE, model_prefix="iwslt_bpe"):
    print("🔤 Training SentencePiece tokenizer...")
    corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair["src"] + "\n")
            f.write(pair["tgt"] + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=os.path.join(TOKENIZER_DIR, model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_id=0
    )
    model_path = os.path.join(TOKENIZER_DIR, f"{model_prefix}.model")
    print(f"✅ Tokenizer trained and saved to {model_path}")
    return model_path

# ============================================================
# 6. Save datasets
# ============================================================

def save_dataset(dataset, filename):
    path = os.path.join(PROCESSED_DIR, filename)
    torch.save(dataset, path)
    print(f"💾 Saved {filename} with {len(dataset)} samples.")

# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":
    train_pairs, val_pairs, test_pairs = load_iwslt17_local(max_train_samples=MAX_TRAIN_SAMPLES)

    tokenizer_model_path = train_tokenizer(train_pairs + val_pairs)
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

    save_dataset(train_pairs, "train.pt")
    save_dataset(val_pairs, "valid.pt")
    save_dataset(test_pairs, "test.pt")

    # Print sample
    print("\n📌 Sample from train dataset:")
    for i in range(2):
        print(f"SRC: {train_pairs[i]['src']}")
        print(f"TGT: {train_pairs[i]['tgt']}")
