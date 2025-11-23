from ast import arg
import math
from optparse import Option
from pkgutil import extend_path
import re
from typing import Optional, Tuple

from numpy import diag
from transformers import (
    PretrainedConfig,
)


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,  # type: ignore
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        # MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x) * x


# define RoPE
def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / rope_base ** torch.arange(0, dim, 2)[: dim // 2].float() / dim

    # if we use scaling (YaRN or LLaMA-NTK) then we need to modify the freqs
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )

        corr_dim = next(
            (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2
        )

        power = torch.arange(0, dim // 2, device=freqs.device).float() / (
            max(dim // 2 - 1, 1)
        )

        beta = beta_slow + (beta_fast - beta_slow) * power

        scale = torch.where(
            torch.arange(dim // 2, device=freqs.device) < corr_dim,
            (beta * factor - beta + 1) / (beta * factor),
            1.0 / factor,
        )

        freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]

    # We need to double the size of the matrix
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1
        )

    # the unsqueeze here is to make sure the cos and sin can be broadcasted to q/k
    # because some layout of q/k may be [batch, seq_len, num_heads, head_dim]
    # instead of [batch, head_dim, seq_len, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * cos.unsqueeze(unsqueeze_dim)
    )

    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # expand: does not allocate new memory. It only adjusts strides so that the
        # tensor appears repeated along the new dimension. The result is non-contiguous.
        # reshape: used instead of view because expand produces a non-contiguous tensor.
        # reshape can create a contiguous copy if needed, while view requires
        # the tensor to be contiguous.
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is None
            else args.num_attention_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(  # attention weight with bias is meanningless
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

        def forward(
            self,
            x: torch.Tensor,
            position_embedding: Tuple[torch.Tensor, torch.Tensor],
            past_kv_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

            # calculate q, k, v
            bsz, seq_len, _ = (
                x.shape
            )  # all encoder/decoder inputs have shape [bsz, seq_len, hidden_dim] for transformer models
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            # slipt to multi-heads
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            # apply RoPE to q, k
            cos, sin = position_embedding
            xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
            # repeat k, v if needed (k v cache)
            if past_kv_value is not None:
                xk = torch.cat([past_kv_value[0], xk], dim=1)
                xv = torch.cat([past_kv_value[1], xv], dim=1)
            past_kv = (xk, xv) if use_cache else None

            xq, xk, xv = (
                # pytorch will treat the last two dims as matmul dims, the rest as batch dims
                xq.transpose(1, 2),
                repeat_kv(xk, self.n_rep).transpose(1, 2),
                repeat_kv(xv, self.n_rep).transpose(1, 2),
            )
            # attention mechanism
            if (
                self.flash
                and seq_len > 1
                and (attention_mask is None or torch.all(attention_mask == 1))
            ):  # flash attention only support causal mask
                attn_mask = (
                    None
                    if attention_mask is None
                    else attention_mask.view(bsz, 1, 1, -1)
                    .expand(bsz, self.n_local_heads, seq_len, -1)
                    .bool()
                )
                
                output = F.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=True
                )
                
            else:
                scores = (xq@xk.transpose(-2,-1)) / math.sqrt(self.head_dim)
                scores = scores + torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                
                if attention_mask is not None:
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                    scores = scores + extended_attention_mask
            
            # use float32 for numerical stability then cast back to save memory
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            
            # output projection
            output = scores @ xv
            output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))

            return output, past_kv