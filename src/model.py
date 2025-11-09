import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================
# 1. 位置编码 (Positional Encoding)
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, relative=False, max_relative_position=16):
        super().__init__()
        self.relative = relative
        if not relative:
            # 绝对位置编码
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
            self.register_buffer("pe", pe)
        else:
            self.max_relative_position = max_relative_position
            self.relative_embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)

    def forward(self, x):
        if not self.relative:
            x = x + self.pe[:, :x.size(1)]
        return x

# ======================
# 2. Scaled Dot-Product Attention
# ======================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, use_relative=False, max_relative_position=16):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.use_relative = use_relative
        if use_relative:
            self.max_relative_position = max_relative_position
            self.relative_embeddings = nn.Embedding(2 * max_relative_position + 1, d_k)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B,H,L,L]

        if self.use_relative:
            seq_len = Q.size(2)
            range_vec = torch.arange(seq_len, device=Q.device)
            distance_mat = range_vec[None, :] - range_vec[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
            final_mat = distance_mat_clipped + self.max_relative_position
            rel_emb = self.relative_embeddings(final_mat)  # [L,L,d_k]
            rel_score = torch.einsum('bhld,lrd->bhlr', Q, rel_emb)  # [B,H,L,L]
            scores = scores + rel_score

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

# ======================
# 3. 多头注意力机制
# ======================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_relative=False, max_relative_position=16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.use_relative = use_relative

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k, use_relative=use_relative,
                                                   max_relative_position=max_relative_position)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.linear_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.linear_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.linear_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        out = self.fc_out(out)
        return out

# ======================
# 4. 前馈网络
# ======================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 使用 GELU 激活函数替代 ReLU
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

# ======================
# 5. Encoder Layer
# ======================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_relative=False, max_relative_position=16):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, use_relative=use_relative,
                                            max_relative_position=max_relative_position)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attn_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_out))
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout(ffn_out))
        return src

# ======================
# 6. Decoder Layer
# ======================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_relative=False, max_relative_position=16):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, use_relative=use_relative,
                                            max_relative_position=max_relative_position)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, use_relative=False)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        _tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))
        _tgt2 = self.enc_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt2))
        _tgt3 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt3))
        return tgt

# ======================
# 7. Encoder
# ======================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout,
                 use_relative=False, max_relative_position=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, relative=use_relative,
                                               max_relative_position=max_relative_position)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_relative=use_relative,
                         max_relative_position=max_relative_position) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# ======================
# 8. Decoder
# ======================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_len, dropout,
                 use_relative=False, max_relative_position=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, relative=use_relative,
                                               max_relative_position=max_relative_position)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_relative=use_relative,
                         max_relative_position=max_relative_position) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        output = self.fc_out(tgt)
        return output

# ======================
# 9. Transformer
# ======================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_seq_len,
                 d_model=512, n_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, dropout=0.1, use_relative_position=False, max_relative_position=16):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, num_encoder_layers, d_ff,
                               max_seq_len, dropout, use_relative=use_relative_position,
                               max_relative_position=max_relative_position)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, num_decoder_layers, d_ff,
                               max_seq_len, dropout, use_relative=use_relative_position,
                               max_relative_position=max_relative_position)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
        return tgt_pad_mask & tgt_sub_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, src_mask)
        return out

    def generate(self, src, sp, max_len=100, device=None):
        if device is None:
            device = src.device
        bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), sp.pad_id()
        self.eval()
        with torch.no_grad():
            src = src.to(device)
            src_mask = self.make_src_mask(src)
            memory = self.encoder(src, src_mask)

            batch_size = src.size(0)
            ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(ys)
                out = self.decoder(ys, memory, tgt_mask, src_mask)
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)
                if finished.all():
                    break

            results = ys[:, 1:].tolist()
        return results
