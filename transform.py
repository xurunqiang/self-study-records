# -*- coding: utf-8 -*-
"""
train_transformer_debug.py
一个“可调试+可视化”的小样本 Transformer 机器翻译训练脚本（空格分词版）
- 读取你的 CSV: 表头 0,1；每行 src,tgt；英文里有逗号也能稳解析（只按第一个逗号切分）
- 可视化：
  * 训练前/后 Cross-Attention 热力图对比（默认 head=0, sample=0）
  * 可选：Encoder self-attn / Decoder self-attn
  * loss 曲线
- 调试：
  * 打印小样本文本、tokens、ids、padding矩阵形状
  * 保存图像到 out_dir

运行示例：
python train_transformer_debug.py --data_csv ./wmt_data/wmt_zh_en_training_corpus.csv --out_dir ./debug_out --epochs 1 --steps 80 --sample_size 64
"""
from tqdm import tqdm
import os
import math
import html
import random
import argparse
from dataclasses import dataclass
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========== 中文显示（尽量兼容 Windows 常见字体） ==========
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# =========================
# 0) 基础设置
# =========================
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_space(text: str):
    return [t for t in text.strip().split() if t]


def ids_to_tokens(ids, itos):
    out = []
    for i in ids:
        if 0 <= i < len(itos):
            out.append(itos[i])
        else:
            out.append("<bad_id>")
    return out


# =========================
# 1) CSV 读取（只按第一个逗号切分，适配你这种数据）
# =========================
def load_pairs_from_csv_first_comma(path, max_rows=None):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 跳过表头：0,1 或 "0","1"
            if line_idx == 0:
                head = line.replace('"', "").replace(" ", "")
                if head == "0,1":
                    continue

            k = line.find(",")
            if k == -1:
                continue

            src = line[:k].strip()
            tgt = line[k + 1 :].strip()

            # 去掉两侧引号（若有）
            if len(src) >= 2 and src[0] == '"' and src[-1] == '"':
                src = src[1:-1].strip()
            if len(tgt) >= 2 and tgt[0] == '"' and tgt[-1] == '"':
                tgt = tgt[1:-1].strip()

            src = html.unescape(src)
            tgt = html.unescape(tgt)

            if src and tgt:
                pairs.append((src, tgt))

            if max_rows is not None and len(pairs) >= max_rows:
                break
    return pairs


def split_train_valid(pairs, valid_ratio=0.01, seed=42):
    pairs = list(pairs)
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_valid = max(1, int(len(pairs) * valid_ratio))
    valid = pairs[:n_valid]
    train = pairs[n_valid:]
    return train, valid


# =========================
# 2) 词表构建 + 编码
# =========================
def build_vocab(pairs, max_vocab=50000, min_freq=2, side="src"):
    assert side in ("src", "tgt")
    counter = Counter()
    for src, tgt in pairs:
        text = src if side == "src" else tgt
        toks = tokenize_space(text)
        counter.update(toks)

    items = [(tok, c) for tok, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)

    if max_vocab is not None:
        items = items[: max(0, max_vocab - len(SPECIAL_TOKENS))]

    itos = SPECIAL_TOKENS + [tok for tok, _ in items]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi, add_bos_eos=True, max_len=None):
    ids = [stoi.get(t, UNK_ID) for t in tokens]
    if add_bos_eos:
        ids = [BOS_ID] + ids + [EOS_ID]
    if max_len is not None:
        ids = ids[:max_len]
        if add_bos_eos and len(ids) > 0 and ids[-1] != EOS_ID:
            ids[-1] = EOS_ID
    return ids


def pad_1d(seqs, pad_value=PAD_ID):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lengths


def make_padding_mask(x, pad_id=PAD_ID):
    return (x == pad_id)  # True=mask掉


def make_subsequent_mask(L):
    return torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)  # True=mask future


# =========================
# 3) Dataset & Collate
# =========================
class TranslationPairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return {"src_text": src, "tgt_text": tgt}


def collate_fn_space(batch, src_stoi, tgt_stoi, max_src_len=64, max_tgt_len=64):
    src_ids, tgt_in_ids, tgt_out_ids = [], [], []

    for item in batch:
        src = item["src_text"]
        tgt = item["tgt_text"]

        src_tok = tokenize_space(src)
        tgt_tok = tokenize_space(tgt)

        src_seq = encode(src_tok, src_stoi, add_bos_eos=True, max_len=max_src_len)
        tgt_full = encode(tgt_tok, tgt_stoi, add_bos_eos=True, max_len=max_tgt_len)
        if len(tgt_full) < 2:
            tgt_full = [BOS_ID, EOS_ID]

        tgt_in = tgt_full[:-1]
        tgt_out = tgt_full[1:]

        src_ids.append(src_seq)
        tgt_in_ids.append(tgt_in)
        tgt_out_ids.append(tgt_out)

    src_pad, src_len = pad_1d(src_ids, pad_value=PAD_ID)
    tgt_in_pad, tgt_in_len = pad_1d(tgt_in_ids, pad_value=PAD_ID)
    tgt_out_pad, _ = pad_1d(tgt_out_ids, pad_value=PAD_ID)

    src_kpm = make_padding_mask(src_pad)      # [B,S]
    tgt_kpm = make_padding_mask(tgt_in_pad)   # [B,T]
    T = tgt_in_pad.size(1)
    tgt_sub = make_subsequent_mask(T)         # [T,T]

    return {
        "src": src_pad,
        "tgt_in": tgt_in_pad,
        "tgt_out": tgt_out_pad,
        "src_key_padding_mask": src_kpm,
        "tgt_key_padding_mask": tgt_kpm,
        "tgt_sub_mask": tgt_sub,
        "src_len": src_len,
        "tgt_len": tgt_in_len,
    }


# =========================
# 4) 可解释 Multi-Head Attention（缓存 Q/K/V/scores/attn）
# =========================
class MultiHeadAttentionExplain(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.cache = {}

    def _split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,L,Dh]

    def _merge_heads(self, x):
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        Q_lin = self.Wq(q)
        K_lin = self.Wk(k)
        V_lin = self.Wv(v)

        Q = self._split_heads(Q_lin)
        K = self._split_heads(K_lin)
        V = self._split_heads(V_lin)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Lq,Lk]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn_drop = self.dropout(attn)

        out_heads = torch.matmul(attn_drop, V)
        out = self._merge_heads(out_heads)
        out = self.Wo(out)

        self.cache = {
            "scores": scores.detach(),  # [B,H,Lq,Lk]
            "attn": attn.detach(),      # [B,H,Lq,Lk]
            "Q": Q.detach(),
            "K": K.detach(),
            "V": V.detach(),
            "out": out.detach(),
        }
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,L,D]

    def forward(self, x):
        L = x.size(1)
        return self.dropout(x + self.pe[:, :L, :])


class EncoderLayerExplain(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionExplain(d_model, n_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_key_padding_mask=None):
        sa = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.addnorm1(x, sa)
        ff = self.ffn(x)
        x = self.addnorm2(x, ff)
        return x


class DecoderLayerExplain(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionExplain(d_model, n_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttentionExplain(d_model, n_heads, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, memory, tgt_sub_mask=None, tgt_key_padding_mask=None, src_key_padding_mask=None):
        sa = self.self_attn(x, x, x, attn_mask=tgt_sub_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.addnorm1(x, sa)
        ca = self.cross_attn(x, memory, memory, key_padding_mask=src_key_padding_mask)
        x = self.addnorm2(x, ca)
        ff = self.ffn(x)
        x = self.addnorm3(x, ff)
        return x


class TransformerExplain(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=32, n_heads=4, d_ff=128, dropout=0.1, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.enc = EncoderLayerExplain(d_model, n_heads, d_ff, dropout)
        self.dec = DecoderLayerExplain(d_model, n_heads, d_ff, dropout)

        self.out = nn.Linear(d_model, tgt_vocab)
        self.cache = {}

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_sub_mask=None):
        src_emb = self.src_emb(src) * math.sqrt(self.d_model)
        src_pe = self.pos(src_emb)
        memory = self.enc(src_pe, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.tgt_emb(tgt_in) * math.sqrt(self.d_model)
        tgt_pe = self.pos(tgt_emb)
        dec_out = self.dec(
            tgt_pe,
            memory,
            tgt_sub_mask=tgt_sub_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        logits = self.out(dec_out)
        return logits


# =========================
# 5) 可视化：热力图保存
# =========================
def save_heatmap(mat, path, title="", x_tokens=None, y_tokens=None, max_x=30, max_y=25, figsize=(12, 6)):
    if torch.is_tensor(mat):
        mat = mat.detach().cpu().numpy()

    H, W = mat.shape
    xN = min(W, max_x)
    yN = min(H, max_y)
    mat = mat[:yN, :xN]

    plt.figure(figsize=figsize)
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.colorbar()

    if x_tokens is not None:
        xt = x_tokens[:xN]
        plt.xticks(range(xN), xt, rotation=90)
    if y_tokens is not None:
        yt = y_tokens[:yN]
        plt.yticks(range(yN), yt)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_loss_curve(loss_hist, path):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist)
    plt.title("Training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# =========================
# 6) 训练 & 抽样可视化
# =========================
def compute_loss(logits, tgt_out, pad_id=PAD_ID):
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), tgt_out.reshape(B * T), ignore_index=pad_id)


def debug_print_batch(batch, src_itos, tgt_itos, n_show=2, max_len=40):
    print("=== Debug batch preview ===")
    for i in range(min(n_show, batch["src"].size(0))):
        src_ids = batch["src"][i].tolist()
        tgt_ids = batch["tgt_in"][i].tolist()
        src_toks = ids_to_tokens(src_ids, src_itos)[:max_len]
        tgt_toks = ids_to_tokens(tgt_ids, tgt_itos)[:max_len]
        print(f"[sample {i}] SRC TOKS:", " ".join(src_toks))
        print(f"[sample {i}] TGT TOKS:", " ".join(tgt_toks))
        print("-" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True, default=".\wmt_data\wmt_zh_en_training_corpus.csv", help="路径：wmt_zh_en_training_corpus.csv")
    ap.add_argument("--out_dir", type=str, default="./debug_out", help="输出目录（保存png等）")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_ratio", type=float, default=0.01)

    ap.add_argument("--sample_size", type=int, default=100, help="小样本训练规模（从train里抽）")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_tgt_len", type=int, default=256)

    ap.add_argument("--max_vocab", type=int, default=5000)
    ap.add_argument("--min_freq", type=int, default=2)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100, help="每个epoch最多跑多少step（小样本调试用）")

    ap.add_argument("--viz_self_attn", action="store_true", help="额外保存 encoder/decoder self-attn 图")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 读取全量 pairs
    pairs_all = load_pairs_from_csv_first_comma(args.data_csv, max_rows=None)
    print("pairs_all:", len(pairs_all))

    train_pairs, valid_pairs = split_train_valid(pairs_all, valid_ratio=args.valid_ratio, seed=args.seed)
    print("train:", len(train_pairs), "valid:", len(valid_pairs))

    # 构建词表（用 train 全量构建，调试更稳定）
    src_stoi, src_itos = build_vocab(train_pairs, max_vocab=args.max_vocab, min_freq=args.min_freq, side="src")
    tgt_stoi, tgt_itos = build_vocab(train_pairs, max_vocab=args.max_vocab, min_freq=args.min_freq, side="tgt")
    print("src_vocab:", len(src_itos), "tgt_vocab:", len(tgt_itos))

    # 从 train 里抽一个小样本训练集
    sample_size = min(args.sample_size, len(train_pairs))
    small_train_pairs = random.sample(train_pairs, sample_size)
    small_train_ds = TranslationPairsDataset(small_train_pairs)

    # 取一个固定的“可视化 batch”（训练前后对比要用同一个）
    # 用 valid 的前 batch（如果valid很小，直接用train样本也行）
    viz_source_pairs = valid_pairs if len(valid_pairs) >= args.batch_size else small_train_pairs
    viz_ds = TranslationPairsDataset(viz_source_pairs[: max(args.batch_size, 8)])
    viz_loader = DataLoader(
        viz_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_space(b, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len),
    )
    viz_batch = next(iter(viz_loader))

    # 训练 loader
    train_loader = DataLoader(
        small_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn_space(b, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len),
        drop_last=False,
    )

    # Debug：打印一些样本
    debug_print_batch(viz_batch, src_itos, tgt_itos, n_show=2, max_len=40)

    # 模型
    model = TransformerExplain(
        src_vocab=len(src_itos),
        tgt_vocab=len(tgt_itos),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(args.max_src_len, args.max_tgt_len, 128),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_forward_and_capture_attn(batch):
        model.eval()
        with torch.no_grad():
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            src_kpm = batch["src_key_padding_mask"].to(device)
            tgt_kpm = batch["tgt_key_padding_mask"].to(device)
            tgt_sub = batch["tgt_sub_mask"].to(device)
            _ = model(src, tgt_in, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm, tgt_sub_mask=tgt_sub)

        # 取 sample=0, head=0
        cross_attn = model.dec.cross_attn.cache["attn"][0, 0]  # [T,S]
        dec_self = model.dec.self_attn.cache["attn"][0, 0]     # [T,T]
        enc_self = model.enc.self_attn.cache["attn"][0, 0]     # [S,S]
        return enc_self, dec_self, cross_attn

    # 训练前注意力
    enc_before, dec_before, cross_before = run_forward_and_capture_attn(viz_batch)

    # tokens（用于坐标标签）
    b = 0
    src_ids = viz_batch["src"][b].tolist()
    tgt_ids = viz_batch["tgt_in"][b].tolist()
    src_tokens = ids_to_tokens(src_ids, src_itos)
    tgt_tokens = ids_to_tokens(tgt_ids, tgt_itos)

    # 保存训练前图
    save_heatmap(
        cross_before,
        os.path.join(args.out_dir, "cross_attn_before.png"),
        title="Cross-Attention BEFORE (head0) [T×S] (tgt->src)",
        x_tokens=src_tokens,
        y_tokens=tgt_tokens,
    )
    if args.viz_self_attn:
        save_heatmap(
            enc_before,
            os.path.join(args.out_dir, "enc_self_attn_before.png"),
            title="Encoder Self-Attn BEFORE (head0) [S×S]",
            x_tokens=src_tokens,
            y_tokens=src_tokens,
            max_x=30,
            max_y=30,
        )
        save_heatmap(
            dec_before,
            os.path.join(args.out_dir, "dec_self_attn_before.png"),
            title="Decoder Self-Attn BEFORE (head0) [T×T] (causal)",
            x_tokens=tgt_tokens,
            y_tokens=tgt_tokens,
            max_x=25,
            max_y=25,
        )

    # =========================
    # 训练循环（小样本调试）
    # =========================
    loss_hist = []
    step_count = 0

    for epoch in range(args.epochs):
        model.train()

        # 每个 epoch 计划跑多少 step（小样本调试就按 args.steps）
        steps_this_epoch = max(1, args.steps - step_count)
        pbar = tqdm(total=steps_this_epoch, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        it = iter(train_loader)
        while step_count < args.steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_kpm = batch["src_key_padding_mask"].to(device)
            tgt_kpm = batch["tgt_key_padding_mask"].to(device)
            tgt_sub = batch["tgt_sub_mask"].to(device)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm, tgt_sub_mask=tgt_sub)
            loss = compute_loss(logits, tgt_out, pad_id=PAD_ID)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            loss_hist.append(loss_val)
            step_count += 1

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            if step_count >= args.steps:
                break

        pbar.close()
        if step_count >= args.steps:
            break


    # 保存 loss 曲线
    save_loss_curve(loss_hist, os.path.join(args.out_dir, "loss_curve.png"))

    # 训练后注意力
    enc_after, dec_after, cross_after = run_forward_and_capture_attn(viz_batch)

    # 保存训练后图
    save_heatmap(
        cross_after,
        os.path.join(args.out_dir, "cross_attn_after.png"),
        title="Cross-Attention AFTER (head0) [T×S] (tgt->src)",
        x_tokens=src_tokens,
        y_tokens=tgt_tokens,
    )
    if args.viz_self_attn:
        save_heatmap(
            enc_after,
            os.path.join(args.out_dir, "enc_self_attn_after.png"),
            title="Encoder Self-Attn AFTER (head0) [S×S]",
            x_tokens=src_tokens,
            y_tokens=src_tokens,
            max_x=30,
            max_y=30,
        )
        save_heatmap(
            dec_after,
            os.path.join(args.out_dir, "dec_self_attn_after.png"),
            title="Decoder Self-Attn AFTER (head0) [T×T] (causal)",
            x_tokens=tgt_tokens,
            y_tokens=tgt_tokens,
            max_x=25,
            max_y=25,
        )

    # 训练前后对比：并排图（Cross-Attn）
    # 为了直观：这里生成一张 before/after 合并图
    def save_side_by_side(a, b, path, title_left="before", title_right="after"):
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        if torch.is_tensor(b):
            b = b.detach().cpu().numpy()

        yN = min(a.shape[0], 25)
        xN = min(a.shape[1], 30)
        a = a[:yN, :xN]
        b = b[:yN, :xN]

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(a, aspect="auto")
        ax1.set_title(title_left)
        ax1.set_xlabel("src key pos")
        ax1.set_ylabel("tgt query pos")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(b, aspect="auto")
        ax2.set_title(title_right)
        ax2.set_xlabel("src key pos")
        ax2.set_ylabel("tgt query pos")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle("Cross-Attention Head0: BEFORE vs AFTER")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    save_side_by_side(
        cross_before,
        cross_after,
        os.path.join(args.out_dir, "cross_attn_before_after.png"),
        title_left="BEFORE",
        title_right="AFTER",
    )

    print("\n✅ Done. Files saved to:", os.path.abspath(args.out_dir))
    print(" - cross_attn_before.png")
    print(" - cross_attn_after.png")
    print(" - cross_attn_before_after.png")
    print(" - loss_curve.png")
    if args.viz_self_attn:
        print(" - enc_self_attn_before/after.png")
        print(" - dec_self_attn_before/after.png")


if __name__ == "__main__":
    main()
