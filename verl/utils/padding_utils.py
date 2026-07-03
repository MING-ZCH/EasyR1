"""
Compatibility shim for flash_attn padding utilities.

transformers >= 4.49 removed index_first_axis/pad_input/unpad_input from
modeling_flash_attention_utils. This module provides fallback implementations
so that verl's padding-free code works without flash_attn installed.
"""
import torch
from torch import Tensor


def index_first_axis(x: Tensor, indices: Tensor) -> Tensor:
    """Select elements from the first axis using indices.

    Equivalent to x[indices] but handles the reshape for multi-dim tensors.
    """
    # indices shape: (total_nnz,)
    # x shape: (batch*seqlen, ...) or (batch*seqlen, num_heads, head_dim)
    return x[indices]


def unpad_input(hidden_states: Tensor, attention_mask: Tensor):
    """Remove padding from hidden_states based on attention_mask.

    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen) with 1 for valid tokens, 0 for padding

    Returns:
        hidden_states_unpad: (total_nnz, ...)
        indices: (total_nnz,) - indices into flattened (batch*seqlen) dim
        cu_seqlens: (batch+1,) - cumulative sequence lengths
        max_seqlen: int - max sequence length in batch
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.zeros(
        attention_mask.shape[0] + 1, dtype=torch.int32, device=attention_mask.device
    )
    cu_seqlens[1:] = seqlens_in_batch.cumsum(dim=0)

    # Flatten batch and seq dims, then index
    hidden_states_flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    hidden_states_unpad = hidden_states_flat[indices]

    return hidden_states_unpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states: Tensor, indices: Tensor, batch: int, seqlen: int) -> Tensor:
    """Pad hidden_states back to (batch, seqlen, ...) shape.

    Args:
        hidden_states: (total_nnz, ...) - unpadded hidden states
        indices: (total_nnz,) - original positions in flattened (batch*seqlen)
        batch: int - batch size
        seqlen: int - sequence length

    Returns:
        output: (batch, seqlen, ...) - padded hidden states
    """
    output = torch.zeros(
        batch * seqlen,
        *hidden_states.shape[1:],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])
