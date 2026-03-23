"""KV cache extract / inject utilities for paged KV buffers."""

import torch


def make_slot_mapping(token_ids: list[int], block_ids: list[int],
                      block_size: int) -> torch.Tensor:
    """Compute absolute slot indices for the given tokens.

    Returns a CPU LongTensor of shape [num_tokens].
    """
    num_tokens = len(token_ids)
    block_ids_t = torch.tensor(block_ids, dtype=torch.long)
    offsets = torch.arange(block_size, dtype=torch.long)
    # slots[b, o] = block_ids[b] * block_size + o
    slots = (block_ids_t.unsqueeze(1) * block_size + offsets).flatten()
    return slots[:num_tokens]


def extract_kv_from_layer(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Extract tokens for one request from a paged KV buffer.

    Args:
        kv_layer: [2, num_pages, page_size, ...] on GPU.
        slot_mapping: [num_tokens] absolute slot indices (CPU or GPU).

    Returns:
        [2, num_tokens, kv_dim] on the same device as kv_layer.
    """
    num_pages = kv_layer.shape[1]
    page_size = kv_layer.shape[2]
    flat = kv_layer.reshape(2, num_pages * page_size, -1)
    sm = slot_mapping.to(flat.device)
    return flat[:, sm, :].clone()


def inject_kv_into_layer(
    kv_cache_layer: torch.Tensor,
    kv_data: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Inject KV tokens for one request into a paged KV buffer in-place.

    Args:
        kv_cache_layer: [2, num_pages, page_size, ...] on GPU.
        kv_data: [2, num_tokens, kv_dim] (CPU or GPU).
        slot_mapping: [num_tokens] absolute slot indices (CPU or GPU).
    """
    num_pages = kv_cache_layer.shape[1]
    page_size = kv_cache_layer.shape[2]
    flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
    sm = slot_mapping.to(flat.device)
    flat[:, sm, :] = kv_data.to(flat.device)
