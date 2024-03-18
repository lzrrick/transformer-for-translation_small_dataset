import torch
import torch.nn as nn
import math


class scaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def mask(self, x, mask_len, value=0):
        '''
        x:[batch_size, seq_len, feature_size] / [batch_size, seq_len]
        mask_len:[batch_size]
        '''
        max_len = x.shape[1]
        mask = torch.arange(
            (max_len),
            device=x.device)[None, :] < (max_len - mask_len[:, None])
        x[~mask] = value
        return x

    def mask_softmax(self, x, mask_len):
        shape = x.shape

        if mask_len is not None:

            if mask_len.dim() == 1:
                mask_len = torch.repeat_interleave(mask_len, shape[1])
            else:
                mask_len = mask_len.reshape(-1)

            x = self.mask(x.reshape(-1, shape[-1]), mask_len, -1e6)

        return nn.functional.softmax(x.reshape(shape), dim=-1)

    def get_attn_mask_len(self, batch_size, tgt_len):
        return torch.arange(tgt_len).__reversed__().repeat(batch_size).reshape(
            batch_size, -1)

    def forward(self,
                query,
                key,
                value,
                key_padding_len=None,
                is_attn_mask_len=False):
        '''
        query:[B, N, D]
        key:[B, M, D]
        value:[B, M, V]
        key_padding_len:[B]
        '''
        dim = query.shape[-1]
        if key_padding_len is not None:
            key = self.mask(key, key_padding_len, 0)
        score = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(
            dim)  # [B, N, M]

        attn_mask_len = None
        if is_attn_mask_len:
            attn_mask_len = self.get_attn_mask_len(key.shape[0], key.shape[1])
        attention_weight = self.mask_softmax(score, attn_mask_len)

        return torch.bmm(self.dropout(attention_weight),
                         value), attention_weight


class MultiheadAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0,
                 bias=True,
                 kdim=None,
                 vdim=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.attention = scaledDotProductAttention(dropout)
        self.num_heads = num_heads
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.wv = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=bias)
        for m in self.modules():
            if isinstance(m, (nn.Linear, )):
                nn.init.xavier_uniform_(m.weight)

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self,
                query,
                key,
                value,
                key_padding_len=None,
                is_attn_mask_len=False):
        '''
        query:[B, N, D]
        key:[B, M, D]
        value:[B, M, V]
        key_padding_len:[B]
        '''
        batch, q_len, _ = query.shape

        q = self.transpose_qkv(self.wq(query), self.num_heads)
        k = self.transpose_qkv(self.wk(key), self.num_heads)
        v = self.transpose_qkv(self.wv(value), self.num_heads)

        if key_padding_len is not None:
            key_padding_len = key_padding_len.reshape(batch, 1).expand(
                -1, self.num_heads).reshape(-1)

        output, weights = self.attention(q, k, v, key_padding_len,
                                         is_attn_mask_len)
        output = self.transpose_output(output, self.num_heads)
        output = self.wo(output)
        return output, torch.mean(weights.reshape(batch, self.num_heads, q_len,
                                                  -1),
                                  dim=1)


if __name__ == '__main__':
    m1 = MultiheadAttention(32, 4)
    x = torch.rand((2, 4, 32))
    key_padding_len = torch.tensor([1, 2])
    x, w = m1(x, x, x, key_padding_len, True)
