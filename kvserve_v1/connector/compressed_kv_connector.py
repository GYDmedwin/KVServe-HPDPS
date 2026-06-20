"""CompressedKVConnector — KVConnectorBase_V1 implementation for PD separation.

Phase 1: NCCL transport, identity compression (no-op).
Phase 2: Real compression via KVCompressionAdapter (kvserve pipelines).

Key design decisions (learned from conversation log):
- LLM (sync) not AsyncLLM.
- get_num_new_matched_tokens returns (n, False) — no WAITING_FOR_REMOTE_KVS.
- start_load_kv proactively drains transport; WORKER has its own _worker_received_kv.
- NCCL recv is only called on-demand (after ZMQ signal) — safe for CUDA graph capture.
- build_connector_meta must not drop pending decode loads across steps.
- Compression: GPU-resident CompressedWire bundles sent through NCCL.
  Compressed messages are identified by a "__compressed__" sentinel in layer_names.
"""

import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from kvserve_v1.compression.manager import (
    add_sentinel, is_compressed_layer_names, strip_sentinel)
from kvserve_v1.compression.wire import (
    CompressedWire, build_wire, restore_from_wire)
from kvserve_v1.utils.kv_utils import (
    extract_kv_from_layer_by_blocks, inject_kv_into_layer_by_blocks,
    make_slot_mapping)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LOAD_TIMEOUT_S = 60.0
_DEFAULT_MAX_NCCL_CHUNK_BYTES = 512 * 1024 * 1024


def _sorted_rids(rids: list[str] | set[str]) -> list[str]:
    return sorted(rids)


def _write_compression_stats(
    request_id: str,
    transfer_id: str,
    original_bytes: int,
    compressed_bytes: int,
    compress_ms: float = 0.0,
    quant_ms: float = 0.0,
    codec_ms: float = 0.0,
    prefill_span_ms: float = 0.0,
) -> None:
    stats_path = os.environ.get("KVSERVE_COMPRESSION_STATS_PATH")
    if not stats_path or original_bytes <= 0 or compressed_bytes <= 0:
        return
    row = {
        "request_id": request_id,
        "transfer_id": transfer_id,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compress_ms": compress_ms,
        # breakdown for overlap analysis:
        "quant_ms": quant_ms,            # per-layer transform+quantize (pipelineable)
        "codec_ms": codec_ms,            # monolithic codec (currently not pipelined)
        "prefill_span_ms": prefill_span_ms,  # first->last layer save = hide budget
    }
    try:
        with open(stats_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as e:
        logger.warning_once(
            "[Connector] Failed to write compression stats to %s: %s",
            stats_path, e)


def _write_decode_stats(
    request_id: str,
    transfer_id: str,
    decompress_ms: float,
    codec_decode_ms: float = 0.0,
    dequant_ms: float = 0.0,
) -> None:
    stats_path = os.environ.get("KVSERVE_DECODE_STATS_PATH")
    if not stats_path:
        return
    row = {
        "request_id": request_id,
        "transfer_id": transfer_id,
        "decompress_ms": decompress_ms,
        # breakdown for overlap analysis:
        "codec_decode_ms": codec_decode_ms,  # monolithic codec decode
        "dequant_ms": dequant_ms,            # per-layer dequant+inverse (pipelineable)
    }
    try:
        with open(stats_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError as e:
        logger.warning_once(
            "[Connector] Failed to write decode stats to %s: %s",
            stats_path, e)


_LAYER_IDX_RE = re.compile(r"layers?\.(\d+)")


def _layer_index(layer_name: str) -> int:
    """Parse the physical transformer layer index from a layer name (e.g.
    'model.layers.10.self_attn.attn' -> 10). Both the overlap and non-overlap
    paths key the per-layer quantization config (compress_v3 layer_id ->
    per-layer bit allocation) on this, so they stay consistent and physically
    correct — instead of lexical sort order ('layers.10' < 'layers.2') which
    misassigns each layer's precision."""
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else 0


def _tl(tag: str) -> None:
    """Timeline marker (epoch) for diagnosing where the cross-node critical path
    spends time (producer compress tail vs consumer decompress/fill). Written to
    a FILE (KVSERVE_TIMELINE_PATH) because worker-process logger output is not
    captured in the node log. Epochs are cross-machine comparable (NTP)."""
    path = os.environ.get("KVSERVE_TIMELINE_PATH")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write("%s %.4f\n" % (tag, time.time()))
    except OSError:
        pass


def _max_nccl_chunk_bytes() -> int:
    raw = os.environ.get("KVSERVE_MAX_NCCL_CHUNK_BYTES")
    if not raw:
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning_once(
            "Invalid KVSERVE_MAX_NCCL_CHUNK_BYTES=%r; using default %d",
            raw, _DEFAULT_MAX_NCCL_CHUNK_BYTES)
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    if value <= 0:
        logger.warning_once(
            "KVSERVE_MAX_NCCL_CHUNK_BYTES must be positive; using default %d",
            _DEFAULT_MAX_NCCL_CHUNK_BYTES)
        return _DEFAULT_MAX_NCCL_CHUNK_BYTES
    return value


@dataclass
class ReqMeta:
    request_id: str
    transfer_id: str
    token_ids: list[int]
    block_ids: list[int]
    slot_mapping: torch.Tensor  # CPU LongTensor [num_tokens]


@dataclass
class CompressedKVConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, request_id: str, transfer_id: str, token_ids: list[int],
                    block_ids: list[int], block_size: int) -> None:
        self.requests.append(ReqMeta(
            request_id=request_id,
            transfer_id=transfer_id,
            token_ids=token_ids,
            block_ids=block_ids,
            slot_mapping=make_slot_mapping(token_ids, block_ids, block_size),
        ))


class CompressedKVConnector(KVConnectorBase_V1):
    """
    KVConnectorBase_V1 implementation.
    Transport: NcclTransport (ZMQ control + NCCL data).
    Compression: configurable via kv_connector_extra_config["compression"].
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        cfg = vllm_config.kv_transfer_config
        self.is_producer = cfg.is_kv_producer
        self._block_size = vllm_config.cache_config.block_size

        # SCHEDULER side: consumer tracks which requests need KV load
        self._requests_need_load: dict[str, tuple["Request", list[int]]] = {}
        # request_id -> transfer_id (shared between P/D). If not provided by
        # upstream router, falls back to request_id.
        self._request_transfer_ids: dict[str, str] = {}
        # SCHEDULER side: producer chunked-prefill accumulation state.
        # req_id -> (accumulated block_ids, full prompt_token_ids)
        self.chunked_prefill: dict[str, tuple[list[int], list[int]]] = {}

        # WORKER side: received buffers keyed by transfer_id.
        self._worker_received_kv: dict[
            str, deque[tuple[list[str], Any]]
        ] = defaultdict(deque)

        # WORKER side: producer accumulates per-layer KV before sending
        self._layer_buffers: dict[str, dict[str, torch.Tensor]] = {}
        # WORKER side: monotonic timestamp of each per-layer save (producer).
        # The span (last-first) ≈ prefill forward time = the budget available to
        # hide per-layer compression behind, for the overlap optimization.
        self._layer_save_times: dict[str, list[float]] = defaultdict(list)

        # KV shape info — read from vllm_config so both producer and consumer have it
        self._num_kv_heads: int = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        self._head_size: int = vllm_config.model_config.get_head_size()
        # Total transformer layers — the consumer streams KV until it has this
        # many layers (overlap=stream sends one layer per message).
        try:
            self._num_layers: int = int(vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config))
        except Exception:  # noqa: BLE001
            self._num_layers = int(getattr(
                vllm_config.model_config.hf_text_config, "num_hidden_layers", 0))
        # Basename of the model path (e.g. "Qwen2.5-7B-Instruct") used to patch
        # the quantizer's model_name when "default" compression mode is selected.
        self._model_name: str = os.path.basename(
            vllm_config.model_config.model.rstrip("/"))
        self._tp_rank: int = 0
        self._tp_size: int = 1

        # Compression spec: None | "default" | custom-dict | controller-dict
        # Built lazily on first use to avoid import overhead at init time.
        self._compressor: Optional[Any] = None
        self._compression_cfg = cfg.kv_connector_extra_config.get("compression")

        # Overlap switch: "off" (default) | "tq" (transform+quant on side stream,
        # codec batched) | "full" (per-layer transform+quant+codec). Env var
        # KVSERVE_OVERLAP_MODE overrides the spec. Only the TileLang backend
        # supports overlap; otherwise it falls back to "off".
        self._overlap_mode = "off"
        if isinstance(self._compression_cfg, dict):
            self._overlap_mode = self._compression_cfg.get("overlap", "off")
        self._overlap_mode = os.environ.get(
            "KVSERVE_OVERLAP_MODE", self._overlap_mode)
        # overlap=stream "group" variant: quant each request independently
        # (lossless, per-request scale) but run ONE codec over the whole layer and
        # send ONE message per layer (cuts codec/.item()/send from per-request to
        # per-layer). Validated LOSSLESS + faster (Qwen 7049→6660, Llama
        # 14131→13680). DEFAULT ON; set KVSERVE_STREAM_GROUP=0 to disable.
        self._stream_group = os.environ.get("KVSERVE_STREAM_GROUP", "1") == "1"
        # Compress-offload: move the per-layer extract+quant+codec+send OFF the
        # forward host thread onto a background thread (the forward host is the
        # bottleneck — save_host 10949→14ms). DEFAULT ON; requires stream+group;
        # set KVSERVE_COMPRESS_OFFLOAD=0 to disable.
        self._compress_offload = os.environ.get("KVSERVE_COMPRESS_OFFLOAD", "1") == "1"
        self._offload_queue: Optional[Any] = None
        self._offload_thread: Optional[Any] = None
        self._offload_stream: Optional[Any] = None
        # Async send: with grp+offload, don't block each prefill step on NCCL
        # send completion (wait_for_sent). Instead pipeline sends across steps and
        # report completion via get_finished (request_finished delays block free).
        # Gated to grp+offload+stream+producer (other modes keep the sync barrier).
        self._async_send = (self.is_producer and self._overlap_mode == "stream"
                            and self._stream_group and self._compress_offload
                            and os.environ.get("KVSERVE_ASYNC_SEND", "1") == "1")
        self._rid2tid: dict[str, str] = {}        # request_id -> transfer_id (worker)
        self._pending_send_rids: set[str] = set()  # finished, awaiting send done
        # Async compress (experimental): also skip the per-step compress barrier
        # (offload_queue.join + offload_stream.synchronize) in wait_for_save, so
        # compress+send of step N fully pipeline into step N+1's forward. Safe ONLY
        # with async_send: block-free is delayed via request_finished→True and
        # gated by get_finished (sent_layers>=num_layers ⇒ compress+extract done).
        self._async_compress = (self._async_send
                                and os.environ.get("KVSERVE_ASYNC_COMPRESS", "1") == "1")
        # Side-stream priority for overlap (lower = higher priority; CUDA range
        # is typically [-1 high, 0 low]). Spec key "side_stream_priority" or env
        # KVSERVE_SIDE_PRIORITY. Default 0 (same as the forward's default stream).
        _prio = 0
        if isinstance(self._compression_cfg, dict):
            _prio = int(self._compression_cfg.get("side_stream_priority", 0))
        self._side_priority = int(os.environ.get("KVSERVE_SIDE_PRIORITY", _prio))
        # Producer side-stream overlap state.
        self._side_stream: Optional[Any] = None
        self._q_layers: dict[str, list] = defaultdict(list)   # rid -> [q_perm]
        self._q_metas: dict[str, list] = defaultdict(list)    # rid -> [canon_meta]
        self._q_layer_names: dict[str, list] = defaultdict(list)  # rid -> [layer_name]

        if role == KVConnectorRole.WORKER:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
                get_world_group,
            )
            local_rank = get_world_group().local_rank
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            self._tp_rank = tp_rank
            self._tp_size = tp_size
            self._transport = self._build_transport(
                cfg, local_rank=local_rank, channel_rank=tp_rank)
            logger.info(
                "[CompressedKVConnector] WORKER init: is_producer=%s "
                "local_rank=%d tp_rank=%d tp_size=%d compression=%s",
                self.is_producer, local_rank, tp_rank, tp_size,
                "enabled" if self._compression_cfg else "disabled")

            # Eagerly build + warm up the compressor at engine init so any JIT
            # kernel compilation (TileLang compress_v3) happens here, not on the
            # first request's critical path.
            if self._compression_cfg is not None:
                compressor = self._get_compressor()
                if compressor is not None and hasattr(compressor, "warmup"):
                    compressor.warmup(self._block_size)
                # PRODUCER: enable the multi-stream side stream only if the
                # backend supports per-layer compression; else fall back to off.
                # CONSUMER: keep _overlap_mode as-is (no side stream needed) — it
                # must stay "stream" to drive per-layer streaming receive.
                if self.is_producer and self._overlap_mode != "off":
                    if (compressor is not None
                            and getattr(compressor, "supports_overlap", lambda: False)()):
                        self._side_stream = torch.cuda.Stream(
                            priority=self._side_priority)
                        logger.info("[CompressedKVConnector] overlap=%s enabled "
                                    "(producer side stream, priority=%d)",
                                    self._overlap_mode, self._side_priority)
                        # Compress-offload: run extract+quant+codec+send on a
                        # background thread so the forward host thread is free of
                        # all compression work (it's host-bound, see save_host).
                        if (self._compress_offload and self._overlap_mode == "stream"
                                and self._stream_group):
                            import queue as _queue
                            import threading as _threading
                            self._offload_queue = _queue.Queue()
                            self._offload_stream = torch.cuda.Stream(
                                priority=self._side_priority)
                            self._offload_thread = _threading.Thread(
                                target=self._offload_loop, daemon=True)
                            self._offload_thread.start()
                            logger.info("[CompressedKVConnector] compress-offload "
                                        "ON (background thread)")
                    else:
                        logger.warning("[CompressedKVConnector] overlap=%s requested "
                                       "but backend lacks per-layer support; using off",
                                       self._overlap_mode)
                        self._overlap_mode = "off"
                elif not self.is_producer and self._overlap_mode != "off":
                    logger.info("[CompressedKVConnector] consumer overlap=%s "
                                "(streaming receive)", self._overlap_mode)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default KV cache layout.")
            return None
        if vllm_config.model_config.use_mla:
            logger.warning_once(
                "CompressedKVConnector has not validated MLA KV cache layout; "
                "falling back to vLLM default layout.")
            return None
        logger.info_once(
            "CompressedKVConnector setting KV cache layout to NHD "
            "for the validated compressed KV transfer path.")
        return "NHD"

    # ── Worker-side ────────────────────────────────────────────────────────

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.is_producer:
            return

        _tl("D load_enter")
        newly_recv = self._transport.drain_received()
        if newly_recv:
            logger.info(
                "[Connector][RID][RECV] drained from transport: %s",
                _sorted_rids(list(newly_recv.keys())),
            )
            for request_id, payloads in newly_recv.items():
                for payload in payloads:
                    self._worker_received_kv[request_id].append(payload)
            self._expand_grouped()

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        expected_rids = [req_meta.request_id for req_meta in meta.requests]
        logger.info(
            "[Connector][RID][RECV] expected this step: %s; buffered: %s",
            _sorted_rids(expected_rids),
            _sorted_rids(list(self._worker_received_kv.keys())),
        )

        # In stream mode KV arrives as N per-layer messages; otherwise one
        # message carries all layers. Either way, drain + inject (decompressing
        # each message as it arrives — receive/decompress overlap) until the
        # request has all its layers, or until timeout.
        streaming = (self._overlap_mode == "stream") and self._num_layers > 0
        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_id = req_meta.transfer_id
            injected = 0
            deadline = time.monotonic() + _LOAD_TIMEOUT_S
            while injected < (self._num_layers if streaming else 1):
                while not self._worker_received_kv.get(transfer_id):
                    newly = self._transport.drain_received()
                    for key, payloads in (newly or {}).items():
                        for p in payloads:
                            self._worker_received_kv[key].append(p)
                    if newly:
                        self._expand_grouped()
                    if self._worker_received_kv.get(transfer_id):
                        break
                    if time.monotonic() > deadline:
                        logger.warning("[Connector] Timeout KV rid=%s transfer_id=%s "
                                       "(%d/%d layers)", rid, transfer_id, injected,
                                       self._num_layers)
                        break
                    time.sleep(0.002)
                if not self._worker_received_kv.get(transfer_id):
                    break
                layer_names, payload = self._worker_received_kv[transfer_id].popleft()
                if not self._worker_received_kv[transfer_id]:
                    self._worker_received_kv.pop(transfer_id, None)
                n = self._inject_payload(layer_names, payload, req_meta, forward_context)
                injected += n if n > 0 else (self._num_layers if streaming else 1)
                if time.monotonic() > deadline:
                    break
        _tl("D load_exit")

    def _offload_loop(self) -> None:
        """Background producer thread: drains layer tasks and runs the full
        extract+quant+group-codec+send off the forward host thread."""
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                self._process_offload_task(task)
            except Exception as e:  # noqa: BLE001
                logger.error("[CompressedKVConnector] offload task failed: %s", e)
            finally:
                self._offload_queue.task_done()

    def _process_offload_task(self, task: dict) -> None:
        comp = self._get_compressor()
        kv_layer = task["kv_layer"]
        layer_id = task["layer_id"]
        with torch.cuda.stream(self._offload_stream):
            self._offload_stream.wait_event(task["event"])  # layer KV written
            q_perms, cmetas, tids, nblocks = [], [], [], []
            for tid, block_ids in task["members"]:
                extracted = extract_kv_from_layer_by_blocks(kv_layer, block_ids)
                extracted.record_stream(self._offload_stream)
                q_perm, cmeta = comp.compress_quant_layer(extracted, layer_id)
                q_perms.append(q_perm)
                cmetas.append(cmeta)
                tids.append(tid)
                nblocks.append(int(q_perm.shape[2]))
            if q_perms:
                cd = comp.finalize_layer_group(q_perms, cmetas, tids, nblocks,
                                               layer_id)
                wire = build_wire(cd, _max_nccl_chunk_bytes())
                self._transport.send_bundle(
                    tids[0], add_sentinel([task["layer_name"]]),
                    wire.meta, wire.body_chunks, wire.aux_tensors,
                    member_tids=(tids if self._async_send else None))

    def _as_grouped(self, layer_names, payload):
        """If this received message is a grouped per-layer bundle, restore it to a
        CompressedKVData; else None (left for the normal per-request path)."""
        if not is_compressed_layer_names(layer_names):
            return None
        if not isinstance(payload, dict) or not payload.get("__bundle__"):
            return None
        wire = CompressedWire(meta=payload["meta"],
                              body_chunks=payload["body_chunks"],
                              aux_tensors=payload["aux_tensors"])
        cd = restore_from_wire(wire)
        return cd if cd.metadata.get("grouped") else None

    def _expand_grouped(self) -> None:
        """Grouped-stream consumer: a layer arrives as ONE message covering all
        requests. Codec-decode it ONCE, split per member, dequant each with its
        own meta, and push the resulting per-(request,layer) uncompressed KV into
        each member's queue — so the per-request injection loop is unchanged."""
        if not self._stream_group:
            return
        expansions = []  # (member_tid, [layer_name], kv_1layer)
        for key in list(self._worker_received_kv.keys()):
            kept: deque = deque()
            for layer_names, payload in self._worker_received_kv[key]:
                cd = self._as_grouped(layer_names, payload)
                if cd is None:
                    kept.append((layer_names, payload))
                    continue
                layer_name = strip_sentinel(layer_names)[0]
                members = self._get_compressor().decompress_layer_group(cd)
                for tid, kv in members:
                    expansions.append((tid, [layer_name], kv.unsqueeze(0)))
            self._worker_received_kv[key] = kept
        for tid, lns, kv1 in expansions:
            self._worker_received_kv[tid].append((lns, kv1))

    def _inject_payload(self, layer_names, payload, req_meta,
                        forward_context) -> int:
        """Decompress (if needed) one received message and inject its layers.
        Returns the number of layers injected (0 on failure)."""
        rid = req_meta.request_id
        transfer_id = req_meta.transfer_id
        if payload is None:
            logger.error("[Connector][RECV] transport failure rid=%s", rid)
            return 0
        if is_compressed_layer_names(layer_names):
            layer_names = strip_sentinel(layer_names)
            compressor = self._get_compressor()
            if compressor is None or not isinstance(payload, dict) \
                    or not payload.get("__bundle__"):
                logger.error("[Connector] Invalid compressed payload for %s", rid)
                return 0
            wire = CompressedWire(meta=payload["meta"],
                                  body_chunks=payload["body_chunks"],
                                  aux_tensors=payload["aux_tensors"])
            compressed = restore_from_wire(wire)
            t_dec = time.monotonic()
            stacked_kv = compressor.decompress(compressed)
            dec = compressor.get_last_decompress_timing() or {}
            _write_decode_stats(rid, transfer_id, (time.monotonic() - t_dec) * 1e3,
                                codec_decode_ms=float(dec.get("t_codec_decode_ms", 0.0)),
                                dequant_ms=float(dec.get("t_dequant_ms", 0.0)))
            if stacked_kv is None:
                logger.error("[Connector] Decompression failed for %s", rid)
                return 0
        else:
            stacked_kv = payload  # GPU tensor from NCCL (uncompressed)
        injected = 0
        for i, layer_name in enumerate(layer_names):
            layer = forward_context.no_compile_layers.get(layer_name)
            if layer is None:
                continue
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None:
                continue
            kv_cache_layer = kv_cache[forward_context.virtual_engine]
            inject_kv_into_layer_by_blocks(
                kv_cache_layer, stacked_kv[i], req_meta.block_ids, rid)
            injected += 1
        return injected

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        if (self._side_stream is not None and self._overlap_mode == "stream"
                and self._stream_group):
            # Grouped stream: quant each request independently (lossless), then
            # ONE codec + ONE send for the whole layer. .item()/codec/send drop
            # from per-request to per-layer.
            layer_id = _layer_index(layer_name)
            ready = torch.cuda.current_stream().record_event()
            if self._compress_offload and self._offload_queue is not None:
                # Hand the whole layer to the background thread; the forward host
                # only enqueues (extract+quant+codec+send all run off-thread).
                if self._async_send:
                    for rm in meta.requests:
                        self._rid2tid[rm.request_id] = rm.transfer_id
                self._offload_queue.put({
                    "layer_name": layer_name, "layer_id": layer_id,
                    "kv_layer": kv_layer, "event": ready,
                    "members": [(rm.transfer_id, rm.block_ids)
                                for rm in meta.requests],
                })
                for req_meta in meta.requests:
                    self._layer_save_times[req_meta.request_id].append(
                        time.monotonic())
                return
            comp = self._get_compressor()
            q_perms, cmetas, tids, nblocks = [], [], [], []
            with torch.cuda.stream(self._side_stream):
                self._side_stream.wait_event(ready)
                for req_meta in meta.requests:
                    extracted = extract_kv_from_layer_by_blocks(
                        kv_layer, req_meta.block_ids)
                    extracted.record_stream(self._side_stream)
                    q_perm, cmeta = comp.compress_quant_layer(extracted, layer_id)
                    q_perms.append(q_perm)
                    cmetas.append(cmeta)
                    tids.append(req_meta.transfer_id)
                    nblocks.append(int(q_perm.shape[2]))
                if q_perms:
                    cd = comp.finalize_layer_group(q_perms, cmetas, tids, nblocks,
                                                   layer_id)
                    wire = build_wire(cd, _max_nccl_chunk_bytes())
                    self._transport.send_bundle(
                        tids[0], add_sentinel([layer_name]),
                        wire.meta, wire.body_chunks, wire.aux_tensors)
            for req_meta in meta.requests:
                self._layer_save_times[req_meta.request_id].append(time.monotonic())
            return

        for req_meta in meta.requests:
            rid = req_meta.request_id
            extracted = extract_kv_from_layer_by_blocks(kv_layer, req_meta.block_ids)

            if self._side_stream is not None and self._overlap_mode != "off":
                # Overlap: transform+quantize this layer on the side stream,
                # concurrently with the next layer's forward. layer_id = the
                # per-rid call order (== forward layer order).
                layer_id = _layer_index(layer_name)  # physical layer index
                ready = torch.cuda.current_stream().record_event()
                comp = self._get_compressor()
                if self._overlap_mode == "stream":
                    # Compress this layer AND send it immediately, on the side
                    # stream → the per-layer NCCL transfer overlaps the remaining
                    # forward. The .item() inside the codec stalls the forward
                    # host thread briefly, but that's the price of overlapping the
                    # send DURING the forward — measured net-positive (deferring
                    # the send loses the overlap; a delayed-tick async variant was
                    # tried and REGRESSED, see git history).
                    with torch.cuda.stream(self._side_stream):
                        self._side_stream.wait_event(ready)
                        c, cmeta = comp.compress_layer_full(extracted, layer_id)
                        _, h, b, s, d = cmeta["codec_shape"]
                        cd = comp.finalize_chunked([c], [cmeta], req_meta.transfer_id,
                                                   "bfloat16", [1, 2, b, s, h, d])
                        wire = build_wire(cd, _max_nccl_chunk_bytes())
                        self._transport.send_bundle(
                            req_meta.transfer_id, add_sentinel([layer_name]),
                            wire.meta, wire.body_chunks, wire.aux_tensors)
                    extracted.record_stream(self._side_stream)
                else:
                    # "tq"/"full": compress on the side stream, send at step end.
                    with torch.cuda.stream(self._side_stream):
                        self._side_stream.wait_event(ready)
                        if self._overlap_mode == "full":
                            payload, cmeta = comp.compress_layer_full(extracted, layer_id)
                        else:  # "tq"
                            payload, cmeta = comp.compress_quant_layer(extracted, layer_id)
                    extracted.record_stream(self._side_stream)
                    self._q_layers[rid].append(payload)
                    self._q_metas[rid].append(cmeta)
                    self._q_layer_names[rid].append(layer_name)
            else:
                if rid not in self._layer_buffers:
                    self._layer_buffers[rid] = {}
                self._layer_buffers[rid][layer_name] = extracted
            self._layer_save_times[rid].append(time.monotonic())

    def wait_for_save(self) -> None:
        if not self.is_producer:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, CompressedKVConnectorMetadata):
            return

        _tl("P save_enter")
        current_rids = {req_meta.request_id for req_meta in meta.requests}
        logger.info(
            "[Connector][RID][SEND] scheduled this step: %s",
            _sorted_rids(current_rids),
        )

        stale = (set(self._layer_buffers) | set(self._q_layers)) - current_rids
        if stale:
            logger.warning("[Connector] Dropping stale layer buffers: %s", stale)
            for rid in stale:
                self._layer_buffers.pop(rid, None)
                self._layer_save_times.pop(rid, None)
                self._q_layers.pop(rid, None)
                self._q_metas.pop(rid, None)
                self._q_layer_names.pop(rid, None)

        # ── Stream mode: each layer was compressed AND sent in save_kv_layer;
        # just block until all per-layer NCCL sends complete. ──────────────────
        if self._side_stream is not None and self._overlap_mode == "stream":
            if self._async_compress:
                # Don't block on compress either — let compress+send of this step
                # pipeline into the next step's forward. Block-free stays safe via
                # request_finished(True) + get_finished (sent⇒compress done).
                for rid in list(current_rids):
                    self._layer_save_times.pop(rid, None)
                _tl("P save_exit")
                return
            if self._compress_offload and self._offload_queue is not None:
                self._offload_queue.join()  # all layers compressed+enqueued (bg)
                if self._offload_stream is not None:
                    self._offload_stream.synchronize()
            else:
                self._side_stream.synchronize()  # per-layer compress+enqueue done
            # Async send: compress is done (kv blocks safe) but DON'T block on NCCL
            # send completion — let sends pipeline across steps; get_finished +
            # request_finished(delay-free) handle completion/block-release.
            if not self._async_send:
                self._transport.wait_for_sent()
            for rid in list(current_rids):
                self._layer_save_times.pop(rid, None)
            _tl("P save_exit")
            return

        # ── Overlap path: transform+quant already ran on the side stream; here
        # we just sync, run the codec, and send. ───────────────────────────────
        if self._side_stream is not None and self._overlap_mode != "off":
            torch.cuda.current_stream().wait_stream(self._side_stream)
            self._side_stream.synchronize()
            for req_meta in meta.requests:
                rid = req_meta.request_id
                transfer_id = req_meta.transfer_id
                q_layers = self._q_layers.pop(rid, [])
                q_metas = self._q_metas.pop(rid, [])
                q_names = self._q_layer_names.pop(rid, [])
                save_times = self._layer_save_times.pop(rid, [])
                prefill_span_ms = ((max(save_times) - min(save_times)) * 1e3
                                   if len(save_times) >= 2 else 0.0)
                if not q_layers:
                    continue
                # Sort by physical layer index so layer_id (= position in the
                # stacked buffer, used by decompress_v3) matches the layer_id
                # passed to compress_quant_layer, and consumer injection order.
                order = sorted(range(len(q_names)), key=lambda i: _layer_index(q_names[i]))
                q_layers = [q_layers[i] for i in order]
                q_metas = [q_metas[i] for i in order]
                q_names = [q_names[i] for i in order]
                # q_metas[0] is a canon_meta (tq) or chunk_meta (full); derive
                # shape from the codec_shape recorded for full, or q_perm for tq.
                if self._overlap_mode == "full":
                    _, heads, blocks, bs, dim = q_metas[0]["codec_shape"]
                else:
                    _, heads, blocks, bs, dim = q_layers[0].shape
                orig_shape = [len(q_layers), 2, blocks, bs, heads, dim]
                orig_bytes = len(q_layers) * 2 * blocks * bs * heads * dim * 2
                compressor = self._get_compressor()
                t0 = time.monotonic()
                if self._overlap_mode == "full":
                    compressed = compressor.finalize_chunked(
                        q_layers, q_metas, rid, "bfloat16", orig_shape)
                else:
                    compressed = compressor.finalize_codec(
                        q_layers, q_metas, rid, "bfloat16", orig_shape)
                codec_ms = (time.monotonic() - t0) * 1e3
                wire = build_wire(compressed, _max_nccl_chunk_bytes())
                send_names = add_sentinel(q_names)
                logger.info(
                    "[Connector][RID][SEND][overlap] rid=%s transfer_id=%s "
                    "layers=%d payload_bytes=%d codec_ms=%.1f",
                    rid, transfer_id, len(q_names), wire.nbytes, codec_ms)
                self._transport.send_bundle(
                    transfer_id, send_names, wire.meta, wire.body_chunks,
                    wire.aux_tensors)
                _write_compression_stats(
                    request_id=rid, transfer_id=transfer_id,
                    original_bytes=orig_bytes, compressed_bytes=wire.nbytes,
                    compress_ms=codec_ms, quant_ms=0.0, codec_ms=codec_ms,
                    prefill_span_ms=prefill_span_ms)
            self._transport.wait_for_sent()
            _tl("P save_exit")
            return

        for req_meta in meta.requests:
            rid = req_meta.request_id
            transfer_id = req_meta.transfer_id
            if rid not in self._layer_buffers:
                continue
            layer_kv = self._layer_buffers.pop(rid)
            save_times = self._layer_save_times.pop(rid, [])
            prefill_span_ms = ((max(save_times) - min(save_times)) * 1e3
                               if len(save_times) >= 2 else 0.0)
            if not layer_kv:
                continue

            # Order by PHYSICAL layer index (not lexical) so compress_all_layers'
            # layer_id 0..N-1 maps each layer to its own quantization config.
            layer_names = sorted(layer_kv.keys(), key=_layer_index)
            stacked = torch.stack([layer_kv[n] for n in layer_names], dim=0)
            # stacked: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_size]

            compressor = self._get_compressor()
            if compressor is not None:
                t0 = time.monotonic()
                compressed = compressor.compress(stacked, rid)
                compress_ms = (time.monotonic() - t0) * 1e3
                if compressed is not None:
                    wire = build_wire(compressed, _max_nccl_chunk_bytes())
                    send_names = add_sentinel(layer_names)
                    logger.info(
                        "[Connector][RID][SEND] sending compressed rid=%s transfer_id=%s "
                        "layers=%d body_chunks=%d aux_tensors=%d payload_bytes=%d",
                        rid, transfer_id, len(layer_names), len(wire.body_chunks),
                        len(wire.aux_tensors), wire.nbytes,
                    )
                    self._transport.send_bundle(
                        transfer_id, send_names, wire.meta, wire.body_chunks,
                        wire.aux_tensors)
                    _write_compression_stats(
                        request_id=rid,
                        transfer_id=transfer_id,
                        original_bytes=stacked.numel() * stacked.element_size(),
                        compressed_bytes=wire.nbytes,
                        compress_ms=compress_ms,
                        quant_ms=float(compressed.metadata.get("t_quant_ms", 0.0)),
                        codec_ms=float(compressed.metadata.get("t_codec_ms", 0.0)),
                        prefill_span_ms=prefill_span_ms,
                    )
                    elapsed_ms = (time.monotonic() - t0) * 1e3
                    compressor.update_controller(rid, elapsed_ms)
                    logger.debug(
                        "[Connector] Sent compressed KV for %s "
                        "(%d layers, %.2f MB → %.2f MB)",
                        rid, len(layer_names),
                        stacked.numel() * stacked.element_size() / 1e6,
                        wire.nbytes / 1e6)
                    continue
                logger.warning(
                    "[Connector] Compression returned None for %s, "
                    "falling back to raw send", rid)

            logger.info(
                "[Connector][RID][SEND] sending raw rid=%s transfer_id=%s layers=%d shape=%s",
                rid, transfer_id, len(layer_names), list(stacked.shape),
            )
            self._transport.send(transfer_id, layer_names, stacked)
            logger.debug("[Connector] Sent raw KV for %s (%d layers)",
                         rid, len(layer_names))

        self._transport.wait_for_sent()
        _tl("P save_exit")

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Async send (grp+offload): report producer requests whose KV is fully
        # sent (all num_layers grouped messages done), so vLLM can free their
        # blocks (they were delay-freed via request_finished → True).
        if not self._async_send:
            return None, None
        self._pending_send_rids |= set(finished_req_ids or ())
        done: set[str] = set()
        for rid in list(self._pending_send_rids):
            tid = self._rid2tid.get(rid)
            if tid is None:  # no KV tracked (shouldn't happen) → don't hang
                done.add(rid)
                continue
            if self._transport.sent_layers(tid) >= self._num_layers:
                done.add(rid)
        self._pending_send_rids -= done
        for rid in done:
            tid = self._rid2tid.pop(rid, None)
            if tid is not None:
                self._transport.clear_sent(tid)
        return (done or None), None

    # ── Scheduler-side ─────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            return 0, False
        # vLLM requires at least one local token to be scheduled for the
        # request; external KV can cover the prompt prefix before that token.
        num_external = (len(request.prompt_token_ids) - 1
                        - num_computed_tokens)
        if num_external <= 0:
            return 0, False
        return num_external, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        # Capture transfer_id as early as possible for both producer/consumer.
        self._resolve_transfer_id(request.request_id, request)
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = CompressedKVConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                    new_req.req_id
                ]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                prompt_token_ids = new_req.prompt_token_ids or []
                # Chunked prefill: defer transfer until full prompt KV exists.
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], prompt_token_ids
                    )
                    continue
                meta.add_request(
                    request_id=new_req.req_id,
                    transfer_id=self._resolve_transfer_id(new_req.req_id),
                    token_ids=prompt_token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
            else:
                if new_req.req_id in self._requests_need_load:
                    meta.add_request(
                        request_id=new_req.req_id,
                        transfer_id=self._resolve_transfer_id(new_req.req_id),
                        token_ids=new_req.prompt_token_ids or [],
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                    )
                    self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                assert req_id in self.chunked_prefill
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = self.chunked_prefill[req_id][0] + block_ids
                prompt_token_ids = self.chunked_prefill[req_id][1]

                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue

                meta.add_request(
                    request_id=req_id,
                    transfer_id=self._resolve_transfer_id(req_id),
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # Resumed preempted requests are first N in cached_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                meta.add_request(
                    request_id=req_id,
                    transfer_id=self._resolve_transfer_id(req_id, request),
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        self.chunked_prefill.pop(request.request_id, None)
        self._request_transfer_ids.pop(request.request_id, None)
        if not self.is_producer:
            self._requests_need_load.pop(request.request_id, None)
            return False, None
        # Async send: delay block free until the request's KV is fully sent
        # (reported via get_finished). Other modes free immediately.
        if self._async_send:
            return True, None
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
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
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

    def _resolve_transfer_id(
        self, request_id: str, request: "Request | None" = None
    ) -> str:
        if request is not None:
            params = getattr(request, "kv_transfer_params", None)
            if params and params.get("transfer_id"):
                self._request_transfer_ids[request_id] = str(
                    params["transfer_id"])
            else:
                logger.warning_once(
                    "Missing transfer_id in kv_transfer_params from router; "
                    "falling back to local request_id. This is only safe for "
                    "single-process tests and should not be used in PD serving.")
        return self._request_transfer_ids.get(request_id, request_id)

    @staticmethod
    def _build_transport(cfg, local_rank: int = 0, channel_rank: int = 0):
        from kvserve_v1.transport.nccl_transport import NcclTransport
        return NcclTransport(
            is_sender=cfg.is_kv_producer,
            host=cfg.kv_ip,
            port=cfg.kv_port,
            local_rank=local_rank,
            channel_rank=channel_rank,
        )
