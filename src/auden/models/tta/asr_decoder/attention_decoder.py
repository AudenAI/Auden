from __future__ import annotations

import math
from typing import Dict, List, Optional

try:
    import k2
except Exception as e:
    raise ImportError(
        "k2 is required for RaggedTensor utilities used by attention decoder IO. "
        "Install via conda: `conda install -c k2-fsa k2` (ensure PyTorch/CUDA match), "
        "or see https://k2-fsa.github.io/k2/."
    ) from e
import torch
import torch.nn as nn

from ...asr.loss import LabelSmoothingLoss
from ...asr.utils import add_eos, add_sos
from ...zipformer.utils.padding import make_pad_mask
from ...zipformer.utils.scaling import penalize_abs_values_gt


class AttentionDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int = 512,
        num_decoder_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        sos_id: int = 1,
        eos_id: int = 1,
        dropout: float = 0.1,
        ignore_id: int = -1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.ignore_id = ignore_id

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=decoder_dim,
            num_decoder_layers=num_decoder_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            memory_dim=memory_dim,
            dropout=dropout,
        )

        self.loss_fun = LabelSmoothingLoss(
            ignore_index=ignore_id, label_smoothing=label_smoothing, reduction="mean"
        )

    def _pre_ys_in_out(self, ys: k2.RaggedTensor, ys_lens: torch.Tensor):
        ys_in = add_sos(ys, sos_id=self.sos_id)
        ys_in_pad = ys_in.pad(mode="constant", padding_value=self.eos_id)
        ys_in_lens = ys_lens + 1

        ys_out = add_eos(ys, eos_id=self.eos_id)
        ys_out_pad = ys_out.pad(mode="constant", padding_value=self.ignore_id)
        return ys_in_pad.to(torch.int64), ys_in_lens, ys_out_pad.to(torch.int64)

    def calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys: k2.RaggedTensor,
        ys_lens: torch.Tensor,
        prefix_ignore_indices: Optional[List[List]] = None,
    ) -> torch.Tensor:
        ys_in_pad, ys_in_lens, ys_out_pad = self._pre_ys_in_out(ys, ys_lens)
        if prefix_ignore_indices is not None:
            for i, ignored in enumerate(prefix_ignore_indices):
                ys_out_pad[i, ignored] = self.ignore_id

        decoder_out = self.decoder(
            x=ys_in_pad,
            x_lens=ys_in_lens,
            memory=encoder_out,
            memory_lens=encoder_out_lens,
        )
        loss = self.loss_fun(x=decoder_out, target=ys_out_pad)
        return loss, decoder_out

    def forward_one_step(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys: torch.Tensor,
        cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        ys_lens = torch.tensor(
            [ys.size(1)] * ys.size(0), dtype=torch.long, device=ys.device
        )
        decoder_out = self.decoder(
            x=ys,
            x_lens=ys_lens,
            memory=encoder_out,
            memory_lens=encoder_out_lens,
            cache=cache,
        )
        return decoder_out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_decoder_layers: int = 6,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos = PositionalEncoding(d_model, dropout_rate=0.1)
        self.num_layers = num_decoder_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    attention_dim=attention_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    memory_dim=memory_dim,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_lens: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos(x)
        x = x.permute(1, 0, 2)

        padding_mask = make_pad_mask(x_lens)
        causal_mask = subsequent_mask(x.shape[0], device=x.device)
        attn_mask = torch.logical_or(
            padding_mask.unsqueeze(1), torch.logical_not(causal_mask).unsqueeze(0)
        )

        if memory is not None:
            memory = memory.permute(1, 0, 2)
            memory_padding_mask = make_pad_mask(memory_lens)
            memory_attn_mask = memory_padding_mask.unsqueeze(1)
        else:
            memory_attn_mask = None

        for i, mod in enumerate(self.layers):
            layer_name = f"layer-{i}"
            layer_cache = None
            if cache is not None:
                layer_cache = [
                    cache["self_attn_cache"].get(layer_name, None),
                    cache["src_attn_cache"].get(layer_name, None),
                ]
            x = mod(
                x,
                attn_mask=attn_mask,
                memory=memory,
                memory_attn_mask=memory_attn_mask,
                cache=layer_cache,
            )
            if cache is not None:
                cache["self_attn_cache"][layer_name] = layer_cache[0]
                cache["src_attn_cache"][layer_name] = layer_cache[1]

        x = x.permute(1, 0, 2)
        x = self.output_layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        memory_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, attention_dim, num_heads, dropout=0.0
        )
        self.norm_src_attn = nn.LayerNorm(d_model)
        self.src_attn = MultiHeadAttention(
            d_model, attention_dim, num_heads, memory_dim=memory_dim, dropout=0.0
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        self_attn_cache, src_attn_cache = (None, None) if cache is None else cache
        if self_attn_cache is not None:
            x = x[-1:, :, :]
            attn_mask = attn_mask[:, -1:, :]
        residual = x
        if src_attn_cache is not None:
            memory = memory[0:0, :, :]

        qkv = self.norm_self_attn(x)
        self_attn_out, self_attn_cache = self.self_attn(
            query=qkv, key=qkv, value=qkv, attn_mask=attn_mask, cache=self_attn_cache
        )
        x = residual + self.dropout(self_attn_out)

        q = self.norm_src_attn(x)
        src_attn_out, src_attn_cache = self.src_attn(
            query=q,
            key=memory,
            value=memory,
            attn_mask=memory_attn_mask,
            cache=src_attn_cache,
        )
        x = x + self.dropout(src_attn_out)

        x = x + self.dropout(self.feed_forward(self.norm_ff(x)))
        if cache is not None:
            cache[0] = self_attn_cache
            cache[1] = src_attn_cache
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim
        self.dropout = dropout

        self.linear_q = nn.Linear(embed_dim, attention_dim, bias=True)
        self.linear_k = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.linear_v = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.out_proj = nn.Linear(attention_dim, embed_dim, bias=True)

    @staticmethod
    def update_cache(
        k: torch.Tensor, v: torch.Tensor, cache: Optional[torch.Tensor] = None
    ):
        if cache is not None:
            kc, vc = torch.split(cache, cache.shape[-1] // 2, dim=-1)
            k = torch.cat([kc, k], dim=0)
            v = torch.cat([vc, v], dim=0)
        cache = torch.cat([k, v], dim=-1)
        return k, v, cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        k, v, cache = self.update_cache(k, v, cache)

        tgt_len, batch, _ = query.shape
        src_len = k.shape[0]
        q = q.reshape(tgt_len, batch, num_heads, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.reshape(src_len, batch, num_heads, head_dim)
        k = k.permute(1, 2, 3, 0)
        v = v.reshape(src_len, batch, num_heads, head_dim)
        v = v.reshape(src_len, batch * num_heads, head_dim).transpose(0, 1)

        attn_weights = torch.matmul(q, k) / math.sqrt(head_dim)
        attn_weights = penalize_abs_values_gt(attn_weights, limit=50.0, penalty=1.0e-04)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        if attn_mask is not None:
            assert attn_mask.shape == (batch, 1, src_len) or attn_mask.shape == (
                batch,
                tgt_len,
                src_len,
            )
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(1), float("-inf")
            )

        attn_weights = attn_weights.view(batch * num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(tgt_len, batch, num_heads * head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, cache


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Swish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def subsequent_mask(size, device="cpu", dtype=torch.bool):
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)
