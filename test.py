import os
import pickle
import math
import torch
import torch.nn as nn

# ====== special tokens ======
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


# ====== Positional Encoding ======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ====== Transformer (nn.Transformer) ======
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_key_padding_mask=None):
        # src: [B,S]
        src_emb = self.pos(self.src_embedding(src))
        return self.tf.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # tgt: [B,T]
        tgt_emb = self.pos(self.tgt_embedding(tgt))
        return self.tf.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


def subsequent_mask_bool(sz, device):
    # True 表示 mask（遮住未来）
    return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)


@torch.no_grad()
def translate_zh2en(model, src_vocab, tgt_vocab, zh_text: str, device, max_len=60):
    """
    Greedy decoding: Chinese -> English
    注意：你的训练 tokenize_space=按空格切词，所以中文最好也带空格（如：我 爱 人工智能）
    """
    model.eval()

    src_ids = src_vocab.encode(zh_text, add_bos=False, add_eos=False)
    if len(src_ids) == 0:
        src_ids = [UNK_ID]

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,S]
    src_kpm = (src == PAD_ID)  # [1,S] bool

    memory = model.encode(src, src_key_padding_mask=src_kpm)

    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)  # [1,1]
    for _ in range(max_len):
        tgt_mask = subsequent_mask_bool(ys.size(1), device)  # [T,T] bool

        out = model.decode(
            tgt=ys,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_kpm,
            tgt_key_padding_mask=None
        )
        logits = model.generator(out[:, -1, :])  # [1,V]
        next_id = int(torch.argmax(logits, dim=-1))

        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == EOS_ID:
            break

    return tgt_vocab.decode(ys[0].tolist())


def load_train_config():
    """
    推荐训练时保存 train_config.pkl，这里自动读取。
    如果没有该文件，就用 fallback 参数（你需要手动改成与训练一致）。
    """
    if os.path.exists("train_config.pkl"):
        with open("train_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        return cfg

    # ===== fallback：没有 train_config.pkl 时（⚠️必须与你训练时一致）=====
    return {
        "d_model": 16,
        "nhead": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "max_len": 512
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) load vocabs
    with open("vocabs.pkl", "rb") as f:
        obj = pickle.load(f)
    src_vocab = obj["src_vocab"]
    tgt_vocab = obj["tgt_vocab"]

    # 2) load model config
    cfg = load_train_config()
    print("Model config:", cfg)

    # 3) build model
    model = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg.get("max_len", 512)
    ).to(device)

    # 4) load weights
    state = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    print("✅ Loaded best_model.pth")

    # 5) interactive
    print("\n输入中文（按空格分词的中文更有效，比如：我 爱 人工智能），输入 q 退出。\n")
    while True:
        zh = input("中文> ").strip()
        if zh.lower() in ("q", "quit", "exit"):
            break
        en = translate_zh2en(model, src_vocab, tgt_vocab, zh, device=device, max_len=60)
        print("英文>", en)


if __name__ == "__main__":
    main()
