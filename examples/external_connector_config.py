"""Minimal KVServe connector configuration for vLLM V1.

Install this repository into the same Python environment as vLLM, then pass one
of these configs to vLLM's LLM/engine construction. The orchestration layer must
send the same ``transfer_id`` to both prefill and decode for each logical
request; the connector uses it as the wire key.
"""

from __future__ import annotations

from vllm import SamplingParams
from vllm.config import KVTransferConfig


CONNECTOR_MODULE = "kvserve_v1.connector.compressed_kv_connector"
CONNECTOR_NAME = "CompressedKVConnector"


def make_kv_transfer_config(
    *,
    role: str,
    decode_host: str,
    base_port: int = 25010,
    compression: object | None = None,
) -> KVTransferConfig:
    """Build the KV transfer config for one side of a two-engine PD deployment.

    Args:
        role: "kv_producer" for prefill or "kv_consumer" for decode.
        decode_host: IP or hostname reachable from the prefill process. For
            single-node tests, "127.0.0.1" is fine.
        base_port: Base port. Homogeneous TP uses ``base_port + tp_rank``.
        compression: None, "default", a custom compression dict, or controller
            config accepted by ``kv_connector_extra_config["compression"]``.
    """
    if role not in {"kv_producer", "kv_consumer"}:
        raise ValueError(f"Unsupported KV role: {role}")

    return KVTransferConfig(
        kv_connector=CONNECTOR_NAME,
        kv_connector_module_path=CONNECTOR_MODULE,
        kv_role=role,
        kv_rank=0 if role == "kv_producer" else 1,
        kv_parallel_size=2,
        kv_ip=decode_host,
        kv_port=base_port,
        kv_connector_extra_config={"compression": compression},
    )


def make_sampling_params(
    *,
    transfer_id: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> SamplingParams:
    """Attach the stable PD transfer id required by the connector.

    The same ``transfer_id`` must be used on producer and consumer for the same
    logical request. A router can derive it from its own globally unique request
    id. Do not let prefill and decode generate unrelated ids locally.
    """
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        extra_args={"kv_transfer_params": {"transfer_id": transfer_id}},
    )
