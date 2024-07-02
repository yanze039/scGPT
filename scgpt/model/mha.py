import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
    

class TorchPackedMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        # self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        # self.inner_attn = nn.MultiheadAttention(
        #             embed_dim=embed_dim,
        #             num_heads=num_heads,
        #             dropout=dropout,
        #             batch_first=batch_first,
        #             **factory_kwargs,
                # )
        self.dropout = dropout
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        # qkv = rearrange(qkv, 'b s (three h) -> b s three h', three=3)
        # (B, nh, T, hs)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        # context, attn_weights = self.inner_attn(q, k, v, key_padding_mask=key_padding_mask,
        #                                         need_weights=need_weights, is_causal=self.causal)
        context = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, is_causal=True)
        # return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
        attn_weights = None
        return self.out_proj(rearrange(context, 'b h s d -> b s (h d)')), attn_weights