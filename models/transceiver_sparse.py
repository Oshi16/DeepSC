# -*- coding: utf-8 -*-

"""
Sparse Transformer includes:
    Encoder
        1. Positional coding
        2. SparseMultiHeadedAttention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. SparseMultiHeadedAttention (self-attention)
        3. SparseMultiHeadedAttention (encoder-decoder attention)
        4. PositionwiseFeedForward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
      
class SparseMultiheadAttention(nn.Module):
    """Sparse multihead attention using a limited attention span."""
    def __init__(self, embed_dim, num_heads, dropout=0.1, attn_span=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_span = attn_span
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.query_ff = nn.Linear(embed_dim, embed_dim)
        self.key_ff = nn.Linear(embed_dim, embed_dim)
        self.value_ff = nn.Linear(embed_dim, embed_dim)
        self.out_ff = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, **kwargs):
        m = query.size(0)
        n = key.size(0)
        if key.size(0) != value.size(0):
            raise RuntimeError("key and value must have same length")
        query = self.query_ff(query).view(m, -1, self.head_dim).transpose(0, 1)
        key = self.key_ff(key).view(n, -1, self.head_dim).transpose(0, 1)
        value = self.value_ff(value).view(n, -1, self.head_dim).transpose(0, 1)
        rows = torch.arange(m, device=query.device).repeat(2 * self.attn_span + 1, 1).transpose(0, 1).flatten()
        cols = torch.cat([torch.arange(i - self.attn_span, i + self.attn_span + 1, device=query.device) for i in range(n)])
        bounds = (cols >= 0) & (cols < n)
        cols[~bounds] = 0
        idxs = torch.stack([rows, cols])
        vals = (query[:, rows, :] * key[:, cols, :] * bounds.view(1, -1, 1)).sum(-1) / math.sqrt(n)
        vals[:, ~bounds] = -float("inf")
        vals = torch.dropout(torch.softmax(vals.view(-1, n, 2 * self.attn_span + 1), dim=-1), self.dropout, self.training).view(-1, idxs.size(1))
        attn_matrix = [torch.sparse.FloatTensor(idxs[:, bounds], val[bounds], (m, n)) for val in vals]
        out = self.out_ff(torch.stack([torch.sparse.mm(attn, val) for attn, val in zip(attn_matrix, value)]).transpose(0, 1).contiguous().view(n, -1, self.embed_dim))
        return out, attn_matrix

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x
      
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout=0.1, attn_span=50):
        super(EncoderLayer, self).__init__()
        self.mha = SparseMultiheadAttention(d_model, num_heads, dropout=dropout, attn_span=attn_span)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask=mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout=0.1, attn_span=50):
        super(DecoderLayer, self).__init__()
        self.self_mha = SparseMultiheadAttention(d_model, num_heads, dropout=dropout, attn_span=attn_span)
        self.src_mha = SparseMultiheadAttention(d_model, num_heads, dropout=dropout, attn_span=attn_span)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        attn_output, _ = self.self_mha(x, x, x, mask=look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output, _ = self.src_mha(x, memory, memory, mask=trg_padding_mask)
        x = self.layernorm2(x + src_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + ffn_output)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout=0.1, attn_span=50):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout, attn_span) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout=0.1, attn_span=50):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout, attn_span) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x

class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output
        
class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, 
                 trg_max_len, d_model, num_heads, dff, dropout = 0.1):
        super(DeepSC, self).__init__()
        
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                             #nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))


        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.dense = nn.Linear(d_model, trg_vocab_size)
