import csv
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import random
import html
import time
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from tqdm import tqdm

# ==========================================
# 1. 配置
# ==========================================
@dataclass
class CFG:
    data_dir: str = r"./wmt_data"
    csv_file: str = "wmt_zh_en_training_corpus.csv"

    # ====== 可开关：读取行数 ======
    max_rows: int | None = None   # None=全量；比如 200000=只读20万

    # ====== 可开关：截断长度（collate时截断，不丢样本）======
    max_src_len: int | None = None  # None=不截断
    max_tgt_len: int | None = None  # None=不截断（注意 tgt 会加 BOS/EOS）

    # ====== 可开关：词表 ======
    max_src_vocab: int | None = None  # None=不截断Top-K
    max_tgt_vocab: int | None = None
    min_freq: int = 2                 # 1=基本不过滤

    # 训练超参数
    batch_size: int = 64
    accum_steps: int = 4
    num_epochs: int = 100
    lr: float = 2e-4
    clip_grad_norm: float = 1.0

    # Transformer（base）
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1

    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio:  float = 0.1
    seed: int = 42

    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()
csv_path = Path(cfg.data_dir) / cfg.csv_file

# ==========================================
# 2. 数据读取
# ==========================================

def load_pairs_from_csv(path: str | Path, max_rows: int | None = None,
                        src_col: str = "C1", tgt_col: str = "C2"):
    path = Path(path)
    pairs = []
    if not path.exists():
        return pairs

    with path.open("r", encoding="utf-8", newline="") as f:
        # 先读一行，用来判断是否有表头
        first_line = f.readline()
        if not first_line:
            return pairs
        f.seek(0)

        # 用 csv 解析首行
        first_row = next(csv.reader([first_line]))
        first_row_norm = [x.strip().strip('"') for x in first_row]

        # 可能的表头情况
        has_header = False
        if (src_col in first_row_norm and tgt_col in first_row_norm):
            has_header = True
        # 有些文件首行是 0,1（你以前的代码也判断过）
        if first_row_norm == ["0", "1"]:
            has_header = True

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                # 兼容有些表头不是 C1/C2（比如 0/1）
                if src_col in row and tgt_col in row:
                    src = row[src_col]
                    tgt = row[tgt_col]
                else:
                    # fallback：取前两列
                    vals = list(row.values())
                    if len(vals) < 2:
                        continue
                    src, tgt = vals[0], vals[1]

                src = html.unescape(str(src)).strip()
                tgt = html.unescape(str(tgt)).strip()
                if src and tgt:
                    pairs.append((src, tgt))

                if max_rows is not None and len(pairs) >= max_rows:
                    break
        else:
            # 没有表头：按两列读
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                # 跳过可能的 "0,1" 行
                row_norm = [x.strip().strip('"') for x in row]
                if row_norm == ["0", "1"] or row_norm == [src_col, tgt_col]:
                    continue

                # 兼容多一列索引：取最后两列作为 src/tgt
                if len(row) >= 2:
                    src, tgt = row[-2], row[-1]
                else:
                    continue

                src = html.unescape(str(src)).strip()
                tgt = html.unescape(str(tgt)).strip()
                if src and tgt:
                    pairs.append((src, tgt))

                if max_rows is not None and len(pairs) >= max_rows:
                    break

    return pairs


def split_train_valid_test(pairs, train_ratio=0.98, valid_ratio=0.01, test_ratio=0.01, seed=42):
    if not pairs:
        return [], [], []
    assert abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-6
    pairs = list(pairs)
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:n_train + n_valid]
    test_pairs = pairs[n_train + n_valid:]
    return train_pairs, valid_pairs, test_pairs


# ==========================================
# 3. Tokenizer / Vocab（✅ 限制Top-K）
# ==========================================

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

def tokenize_space(text: str):
    return [t for t in text.strip().split() if t]

class Vocab:
    def __init__(self, pairs, lang_idx, min_freq=2, max_size=50000):
        self.stoi = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.itos = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

        counter = Counter()
        for src, tgt in pairs:
            text = src if lang_idx == 0 else tgt
            counter.update(tokenize_space(text))

        # ✅ 按频率排序，截断 Top-K
        items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda x: x[1], reverse=True)
        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIAL_TOKENS))]

        for token, _ in items:
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        tokens = tokenize_space(text)
        if max_len is not None:
            tokens = tokens[:max_len]
        ids = []
        if add_bos:
            ids.append(BOS_ID)
        for t in tokens:
            ids.append(self.stoi.get(t, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, indices):
        tokens = []
        for i in indices:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID, UNK_ID):
                continue
            tokens.append(self.itos.get(i, UNK))
        return " ".join(tokens)

# ==========================================
# 4. Dataset & Collate（✅ 限制句长）
# ==========================================

class TranslationPairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]  # (src, tgt)

class CollateTranslate:
    def __init__(self, src_vocab, tgt_vocab, max_src_len, max_tgt_len):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch):
        src_batch, tgt_batch = [], []
        for src_text, tgt_text in batch:
            src_ids = self.src_vocab.encode(src_text, add_bos=False, add_eos=False, max_len=self.max_src_len)
            tgt_ids = self.tgt_vocab.encode(tgt_text, add_bos=True, add_eos=True, max_len=self.max_tgt_len)

            if len(src_ids) == 0:
                src_ids = [UNK_ID]
            if len(tgt_ids) < 2:
                tgt_ids = [BOS_ID, EOS_ID]

            src_batch.append(torch.tensor(src_ids, dtype=torch.long))
            tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
        return src_padded, tgt_padded


# ==========================================
# 5. ✅ Bucket Sampler（减少 padding）
# ==========================================

class LengthBucketSampler(Sampler):
    """
    简单的“按长度分桶+打乱”采样器：
    - 先按 src 长度排序
    - 按 batch 切块
    - 每块内部顺序固定，但块的顺序每 epoch 打乱
    """
    def __init__(self, dataset: TranslationPairsDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        lengths = [(i, len(tokenize_space(dataset.pairs[i][0]))) for i in range(len(dataset))]
        lengths.sort(key=lambda x: x[1])
        self.sorted_indices = [i for i, _ in lengths]

    def __iter__(self):
        rnd = random.Random(self.seed + int(time.time()))
        chunks = [self.sorted_indices[i:i+self.batch_size] for i in range(0, len(self.sorted_indices), self.batch_size)]
        rnd.shuffle(chunks)
        for c in chunks:
            yield from c

    def __len__(self):
        return len(self.dataset)

# ==========================================
# 6. Transformer（bool mask，省 warning）
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# ========= Mask 工具 =========
def causal_mask_bool(T, device):
    # True = mask（遮住未来）
    return torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

def apply_attn_masks(scores, attn_mask=None, key_padding_mask=None):
    """
    scores: [B, H, Q, K]
    attn_mask: [Q, K] bool (True=mask)
    key_padding_mask: [B, K] bool (True=pad, mask)
    """
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    return scores

# ========= 位置编码 =========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ========= 多头注意力（返回 attn_weights） =========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        return x

    def merge_heads(self, x):
        # x: [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=True):
        """
        q: [B, Q, D], k/v: [B, K, D]
        attn_mask: [Q, K] bool
        key_padding_mask: [B, K] bool
        """
        Q = self.split_heads(self.Wq(q))  # [B,H,Q,Dh]
        K = self.split_heads(self.Wk(k))  # [B,H,K,Dh]
        V = self.split_heads(self.Wv(v))  # [B,H,K,Dh]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Q,K]
        scores = apply_attn_masks(scores, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        attn = torch.softmax(scores, dim=-1)  # [B,H,Q,K]
        attn = self.dropout(attn)

        out = attn @ V  # [B,H,Q,Dh]
        out = self.merge_heads(out)  # [B,Q,D]
        out = self.Wo(out)

        if need_weights:
            return out, attn
        return out, None

# ========= FFN =========
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ========= EncoderLayer =========
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None, debug_cache=None, layer_id=0):
        # Self-attn
        attn_out, attn_w = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=src_key_padding_mask, need_weights=True)
        x = self.norm1(x + self.drop(attn_out))

        if debug_cache is not None:
            debug_cache[f"enc_self_attn_L{layer_id}"] = attn_w.detach().cpu()  # [B,H,S,S]

        # FFN
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x

# ========= DecoderLayer =========
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                debug_cache=None, layer_id=0):
        # 1) masked self-attn (causal)
        self_out, self_w = self.self_attn(
            y, y, y,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True
        )
        y = self.norm1(y + self.drop(self_out))
        if debug_cache is not None:
            debug_cache[f"dec_self_attn_L{layer_id}"] = self_w.detach().cpu()  # [B,H,T,T]

        # 2) cross-attn (tgt attends to src memory)
        cross_out, cross_w = self.cross_attn(
            y, memory, memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True
        )
        y = self.norm2(y + self.drop(cross_out))
        if debug_cache is not None:
            debug_cache[f"dec_cross_attn_L{layer_id}"] = cross_w.detach().cpu()  # [B,H,T,S]

        # 3) ffn
        y = self.norm3(y + self.ffn(y))
        return y

# ========= Full Transformer (custom) =========
class CustomTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len=512)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt_in,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                tgt_mask=None, memory_key_padding_mask=None,
                debug=False):
        """
        src: [B,S]
        tgt_in: [B,T]
        debug=True 会返回 debug_cache
        """
        debug_cache = {} if debug else None

        x = self.pos(self.src_emb(src))     # [B,S,D]
        y = self.pos(self.tgt_emb(tgt_in))  # [B,T,D]

        # Encoder
        for li, layer in enumerate(self.enc_layers):
            x = layer(x, src_key_padding_mask=src_key_padding_mask, debug_cache=debug_cache, layer_id=li)

        memory = x

        # Decoder
        for li, layer in enumerate(self.dec_layers):
            y = layer(
                y, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                debug_cache=debug_cache,
                layer_id=li
            )

        logits = self.proj(y)  # [B,T,V]
        return logits, debug_cache


def generate_square_subsequent_mask_bool(sz, device):
    return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)

def create_masks(src, tgt_in, device):
    tgt_mask = generate_square_subsequent_mask_bool(tgt_in.size(1), device)
    src_padding_mask = (src == PAD_ID)
    tgt_padding_mask = (tgt_in == PAD_ID)
    return tgt_mask, src_padding_mask, tgt_padding_mask

# ==========================================
# 7. Train / Eval（✅ AMP + 梯度累积 + clip）
# ==========================================

def train_epoch(model, loader, optimizer, criterion, device, use_amp=True, accum_steps=1, clip_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.startswith("cuda")))

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, (src, tgt) in enumerate(pbar, 1):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_mask, src_pad, tgt_pad = create_masks(src, tgt_in, device)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
            logits, _ = model(
                src, tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad,
                debug=False
            )

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            if clip_grad_norm is not None and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=loss.item() * accum_steps)

    return total_loss / len(loader)

def save_attention_maps(debug_cache, out_dir="attn_imgs", title_prefix="", max_heads=4, dpi=160):
    """
    将注意力热力图保存为图片，不展示。
    debug_cache: model(..., debug=True) 返回的 dict
    out_dir: 保存目录
    title_prefix: 文件名前缀（比如 Epoch1-）
    """
    if not debug_cache:
        print("No attention cached.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, attn in debug_cache.items():
        if attn is None:
            continue
        # attn: [B,H,Q,K] 只取 batch 的第 0 个样本
        attn0 = attn[0]  # [H,Q,K]
        H = attn0.size(0)
        show_h = min(H, max_heads)

        for h in range(show_h):
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(attn0[h].numpy(), aspect="auto")
            plt.colorbar()
            plt.title(f"{title_prefix}{name} | head {h}")
            plt.xlabel("Key positions")
            plt.ylabel("Query positions")
            plt.tight_layout()

            save_path = out_dir / f"{title_prefix}{name}_head{h}.png"
            fig.savefig(save_path, dpi=dpi)
            plt.close(fig)  # ✅ 关键：关闭释放内存


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0

    for src, tgt in tqdm(loader, desc="Validating", leave=False):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_mask = causal_mask_bool(tgt_in.size(1), device)   # [T,T] bool
        src_pad = (src == PAD_ID)                              # [B,S] bool
        tgt_pad = (tgt_in == PAD_ID)                           # [B,T] bool

        with torch.cuda.amp.autocast(enabled=(use_amp and str(device).startswith("cuda"))):
            logits, _ = model(
                src, tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_pad,
                debug=False
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# ==========================================
# 8. 主程序
# ==========================================

if __name__ == "__main__":
    print("Using device:", cfg.device)
    print("Loading data...")
    pairs_all = load_pairs_from_csv(csv_path, max_rows=cfg.max_rows, src_col="C1", tgt_col="C2")

    if not pairs_all:
        print("⚠️ 数据不存在，使用模拟数据")
        pairs_all = [
            ("i love ai", "我 爱 人工智能"),
            ("deep learning is fun", "深度 学习 很 有趣"),
            ("transformer is powerful", "变形金刚 很 强大"),
            ("hello world", "你 好 世界"),
        ] * 2000

    print("Total pairs:", len(pairs_all))

    train_pairs, valid_pairs, test_pairs = split_train_valid_test(
        pairs_all, cfg.train_ratio, cfg.valid_ratio, cfg.test_ratio, cfg.seed
    )

    print("Building vocabs (Top-K)...")
    src_vocab = Vocab(train_pairs, lang_idx=0, min_freq=cfg.min_freq, max_size=cfg.max_src_vocab)
    tgt_vocab = Vocab(train_pairs, lang_idx=1, min_freq=cfg.min_freq, max_size=cfg.max_tgt_vocab)
    print("Src Vocab:", len(src_vocab))
    print("Tgt Vocab:", len(tgt_vocab))

    train_ds = TranslationPairsDataset(train_pairs)
    valid_ds = TranslationPairsDataset(valid_pairs)

    collate = CollateTranslate(src_vocab, tgt_vocab, cfg.max_src_len, cfg.max_tgt_len)


    # ✅ 用 LengthBucketSampler，减少 padding
    train_sampler = LengthBucketSampler(train_ds, batch_size=cfg.batch_size, seed=cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and cfg.device.startswith("cuda")),
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and cfg.device.startswith("cuda")),
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
    )

    model = CustomTransformer(
        len(src_vocab), len(tgt_vocab),
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    train_losses, valid_losses = [], []
    best_valid = float("inf")

    print("\nStarting training...")
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        tr = train_epoch(
            model, train_loader, optimizer, criterion, cfg.device,
            use_amp=cfg.use_amp, accum_steps=cfg.accum_steps,
            clip_grad_norm=cfg.clip_grad_norm
        )
        va = evaluate(model, valid_loader, criterion, cfg.device, use_amp=cfg.use_amp)

        train_losses.append(tr)
        valid_losses.append(va)

        print(f"Epoch {epoch} | time {time.time()-t0:.1f}s | train {tr:.4f} | valid {va:.4f}")
        model.eval()
        if epoch % 5 == 0:
            with torch.no_grad():
                src_vis, tgt_vis = next(iter(valid_loader))  # 取验证集第一批
                src_vis = src_vis.to(cfg.device)
                tgt_vis = tgt_vis.to(cfg.device)
                tgt_in_vis = tgt_vis[:, :-1]

                T = tgt_in_vis.size(1)
                tgt_mask_vis = causal_mask_bool(T, cfg.device)
                src_pad_vis = (src_vis == PAD_ID)
                tgt_pad_vis = (tgt_in_vis == PAD_ID)

                # 只在可视化时 debug=True
                _, debug_cache = model(
                    src_vis, tgt_in_vis,
                    src_key_padding_mask=src_pad_vis,
                    tgt_key_padding_mask=tgt_pad_vis,
                    tgt_mask=tgt_mask_vis,
                    memory_key_padding_mask=src_pad_vis,
                    debug=True
                )

            save_attention_maps(
                debug_cache,
                out_dir="attn_imgs",
                title_prefix=f"Epoch{epoch}-",
                max_heads=4,
                dpi=160
            )

        if va < best_valid:
            best_valid = va
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ saved best_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, cfg.num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, cfg.num_epochs + 1), valid_losses, label="Valid Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.grid(True)
    plt.show()
    # 保存词表（src_vocab / tgt_vocab）
    with open("vocabs.pkl", "wb") as f:
        pickle.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)
    print("✅ saved vocabs.pkl")
    with open("train_config.pkl", "wb") as f:
        pickle.dump({
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_encoder_layers": cfg.num_encoder_layers,
            "num_decoder_layers": cfg.num_decoder_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "dropout": cfg.dropout,
            "max_len": 512
        }, f)
    print("✅ saved train_config.pkl")

    print("Training complete.")
