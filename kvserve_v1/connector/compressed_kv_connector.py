"""CompressedKVConnector — KVConnectorBase_V1 implementation for PD separation.

Phase 1: NCCL transport, identity compression (no-op).
Phase 2: Real compression via KVCompressionAdapter (kvserve pipelines).

Key design decisions (learned from conversation log):
- LLM (sync) not AsyncLLM.
- get_num_new_matched_tokens returns (n, False) — no WAITING_FOR_REMOTE_KVS.
- start_load_kv proactively drains transport; WORKER has its own _worker_received_kv.
- NCCL recv is only called on-demand (after ZMQ signal) — safe for CUDA graph capture.
- build_connector_meta clears _requests_need_load at end.
- Compression: EasyDist-packed uint8 tensor sent as a single NCCL payload.
  Compressed messages are identified by a "__compressed__" sentinel in layer_names.
"""

import os
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from kvserve_v1.compression.manager import (
    add_sentinel, is_compressed_layer_names, pack_compressed,
    strip_sentinel, unpack_compressed)
from kvserve_v1.utils.kv_utils import (extract_kv_from_layer,
                                        inject_kv_into_layer, make_slot_mapping)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 60.0


def _sorted_rids(rids: list[str] | set[str]) -> list[str]:
    return sorted(rids)


def _build_transfer_key(token_ids: list[int]) -> str:
    """Build a stable, connector-owned transfer key from prompt tokens."""
    token_str = ",".join(map(str, token_ids))
    digest = hashlib.sha1(token_str.encode("utf-8")).hexdigest()[:16]
    return f"tok-{len(token_ids)}-{digest}"


@dataclass
class ReqMeta:
    request_id: str
    transfer_key: str
    token_ids: list[int]
    slot_mapping: torch.Tensor  # CPU LongTensor [num_tokens]


@dataclass
class CompressedKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, request_id: str, token_ids: list[int],
                    block_ids: list[int], block_size: int) -> None:
        self.requests.append(ReqMeta(
            request_id=request_id,
            transfer_key=_build_transfer_key(token_ids),
            token_ids=token_ids,
            slot_mapping=make_slot_mapping(token_ids, block_ids, block_size),
        ))


class CompressedKVConnector(KVConnectorBase_V1):
    """
    KVConnectorBase_V1 implementation.
    Transport: NcclTransport (ZMQ control + NCCL data).
    Compression: configurable via kv_connector_extra_config["compression"].
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

        cfg = vllm_config.kv_transfer_config
        self.is_producer = cfg.is_kv_producer
        self._block_size = vllm_config.cache_config.block_size

        # SCHEDULER side: consumer tracks which requests need KV load
        self._requests_need_load: dict[str, tuple["Request", list[int]]] = {}

        # WORKER side: consumer buffers received by transfer_key.
        # Queue avoids overwrite under duplicate prompts/high concurrency.
        self._worker_received_kv: dict[
            str, deque[tuple[list[str], torch.Tensor]]
        ] = defaultdict(deque)

        # WORKER side: producer accumulates per-layer KV before sending
        self._layer_buffers: dict[str, dict[str, torch.Tensor]] = {}

        # KV shape info — read from vllm_config so both producer and consumer have it
        self._num_kv_heads: int = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self._head_size: int = vllm_config.model_config.get_head_size()
        # Basename of the model path (e.g. "Qwen2.5-7B-Instruct") used to patch
        # the quantizer's model_name when "default" compression mode is selected.
        self._model_name: str = os.path.basename(
            vllm_config.model_config.model.rstrip("/"))

        # Compression spec: None | "default" | custom-dict | controller-dict
        # Built lazily on first use to avoid import overhead at init time.
        self._compressor: Optional[Any] = None
        self._compression_cfg = cfg.kv_connector_extra_config.get("compression")

        if role == KVConnectorRole.WORKER:
            from vllm.distributed.parallel_state import get_world_group
            local_rank = get_world_group().local_rank
            self._transport = self._build_transport(cfg, local_rank)
            logger.info(
                "[CompressedKVConnector] WORKER init: is_producer=%s "
                "local_rank=%d compression=%s",
                self.is_producer, local_rank,
                "enabled" if self._compression_cfg else "disabled")

    # ── Worker-side ────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.is_producer:
            return

        newly_recv = self._transport.drain_received()
        if newly_recv:
            logger.info(
                "[Connector][RID][RECV] drained from transport: %s",
                _sorted_rids(list(newly_recv.keys())),
            )
            for transfer_key, payload in newly_recv.items():
                self._worker_received_kv[transfer_key].append(payload)

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        expected_rids = [req_meta.request_id for req_meta in meta.requests]
        expected_keys = [req_meta.transfer_key for req_meta in meta.requests]
        logger.info(
            "[Connector][RID][RECV] expected this step: %s; buffered: %s",
            _sorted_rids(expected_rids),
            _sorted_rids(list(self._worker_received_kv.keys())),
        )
        logger.info(
            "[Connector][RID][RECV] expected keys this step: %s",
            _sorted_rids(expected_keys),
        )

        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_key = req_meta.transfer_key

            deadline = time.monotonic() + _LOAD_TIMEOUT_S
            while not self._worker_received_kv.get(transfer_key):
                newly_recv = self._transport.drain_received()
                if newly_recv:
                    logger.info(
                        "[Connector][RID][RECV] drained while waiting: %s",
                        _sorted_rids(list(newly_recv.keys())),
                    )
                    for key, payload in newly_recv.items():
                        self._worker_received_kv[key].append(payload)
                if self._worker_received_kv.get(transfer_key):
                    break
                if time.monotonic() > deadline:
                    logger.warning(
                        "[Connector] Timeout waiting KV for rid=%s key=%s",
                        rid, transfer_key)
                    break
                time.sleep(0.005)

            if not self._worker_received_kv.get(transfer_key):
                continue

            layer_names, payload = self._worker_received_kv[transfer_key].popleft()
            if not self._worker_received_kv[transfer_key]:
                self._worker_received_kv.pop(transfer_key, None)

            # Decompress if needed
            if is_compressed_layer_names(layer_names):
                layer_names = strip_sentinel(layer_names)
                compressor = self._get_compressor()
                if compressor is not None:
                    compressed = unpack_compressed(payload, rid)
                    stacked_kv = compressor.decompress(compressed)
                    if stacked_kv is None:
                        logger.error(
                            "[Connector] Decompression failed for %s", rid)
                        continue
                else:
                    logger.error(
                        "[Connector] Received compressed KV but no compressor "
                        "configured for %s", rid)
                    continue
            else:
                stacked_kv = payload  # GPU tensor from NCCL

            for i, layer_name in enumerate(layer_names):
                layer = forward_context.no_compile_layers.get(layer_name)
                if layer is None:
                    continue
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue
                kv_cache_layer = kv_cache[forward_context.virtual_engine]
                inject_kv_into_layer(
                    kv_cache_layer, stacked_kv[i], req_meta.slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        for req_meta in meta.requests:
            rid = req_meta.request_id
            if rid not in self._layer_buffers:
                self._layer_buffers[rid] = {}
            extracted = extract_kv_from_layer(kv_layer, req_meta.slot_mapping)
            self._layer_buffers[rid][layer_name] = extracted

    def wait_for_save(self) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        current_rids = {req_meta.request_id for req_meta in meta.requests}
        current_keys = {req_meta.transfer_key for req_meta in meta.requests}
        logger.info(
            "[Connector][RID][SEND] scheduled this step: %s",
            _sorted_rids(current_rids),
        )
        logger.info(
            "[Connector][RID][SEND] scheduled keys this step: %s",
            _sorted_rids(current_keys),
        )

        stale = set(self._layer_buffers) - current_rids
        if stale:
            logger.warning("[Connector] Dropping stale layer buffers: %s", stale)
            for rid in stale:
                self._layer_buffers.pop(rid, None)

        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_key = req_meta.transfer_key
            if rid not in self._layer_buffers:
                continue
            layer_kv = self._layer_buffers.pop(rid)
            if not layer_kv:
                continue

            layer_names = sorted(layer_kv.keys())
            stacked = torch.stack([layer_kv[n] for n in layer_names], dim=0)
            # stacked: [num_layers, 2, num_tokens, kv_dim] on GPU

            compressor = self._get_compressor()
            if compressor is not None:
                t0 = time.monotonic()
                compressed = compressor.compress(stacked, rid)
                if compressed is not None:
                    payload = pack_compressed(compressed)
                    send_names = add_sentinel(layer_names)
                    logger.info(
                        "[Connector][RID][SEND] sending compressed rid=%s key=%s "
                        "layers=%d payload_shape=%s",
                        rid, transfer_key, len(layer_names), list(payload.shape),
                    )
                    self._transport.send(transfer_key, send_names, payload)
                    elapsed_ms = (time.monotonic() - t0) * 1e3
                    compressor.update_controller(rid, elapsed_ms)
                    logger.debug(
                        "[Connector] Sent compressed KV for %s "
                        "(%d layers, %.2f MB → %.2f MB)",
                        rid, len(layer_names),
                        stacked.numel() * stacked.element_size() / 1e6,
                        payload.numel() * payload.element_size() / 1e6)
                    continue
                logger.warning(
                    "[Connector] Compression returned None for %s, "
                    "falling back to raw send", rid)

            logger.info(
                "[Connector][RID][SEND] sending raw rid=%s key=%s layers=%d shape=%s",
                rid, transfer_key, len(layer_names), list(stacked.shape),
            )
            self._transport.send(transfer_key, layer_names, stacked)
            logger.debug("[Connector] Sent raw KV for %s (%d layers)",
                         rid, len(layer_names))

        self._transport.wait_for_sent()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    # ── Scheduler-side ─────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            return 0, False
        num_external = (len(request.prompt_token_ids) - 1
                        - num_computed_tokens)
        if num_external <= 0:
            return 0, False
        return num_external, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = CompressedKVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
            else:
                if new_req.req_id in self._requests_need_load:
                    _, block_ids = self._requests_need_load.pop(new_req.req_id)
                    meta.add_request(
                        request_id=new_req.req_id,
                        token_ids=new_req.prompt_token_ids,
                        block_ids=block_ids,
                        block_size=self._block_size,
                    )

        # Clear stale entries (cancelled/preempted requests never scheduled).
        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return False, None

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_compressor(self):
        """Lazily build KVCompressionAdapter on first use."""
        if self._compression_cfg is None:
            return None
        if self._compressor is not None:
            return self._compressor

        from kvserve_v1.compression.manager import KVCompressionAdapter
        self._compressor = KVCompressionAdapter(
            compression_spec=self._compression_cfg,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_size,
            model_name=self._model_name,
        )
        spec = self._compression_cfg
        if spec == "default":
            mode_desc = "default"
        elif isinstance(spec, dict):
            mode_desc = spec.get("mode", "custom")
        else:
            mode_desc = str(type(spec).__name__)
        logger.info(
            "[Connector] KVCompressionAdapter built: mode=%s heads=%d head_size=%d",
            mode_desc, self._num_kv_heads, self._head_size)
        return self._compressor

    @staticmethod
    def _build_transport(cfg, local_rank: int = 0):
        from kvserve_v1.transport.nccl_transport import NcclTransport
        return NcclTransport(
            is_sender=cfg.is_kv_producer,
            host=cfg.kv_ip,
            port=cfg.kv_port,
            local_rank=local_rank,
        )
