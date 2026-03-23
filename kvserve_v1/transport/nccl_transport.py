"""NCCL data transport with ZMQ control signaling.

Architecture (mirrors P2pNcclEngine's PUT_ASYNC approach):
  Consumer (receiver): ZMQ ROUTER binds on kv_port
  Producer (sender):   ZMQ DEALER connects to kv_port

Protocol:
  Init (startup):
    Producer  → INIT{unique_id} → Consumer
    Both call ncclCommInitRank (producer=rank 0, consumer=rank 1)
    ncclCommInitRank is collective → blocks until both sides call it.

  Per-transfer (PUT mode):
    Producer  → PUT{request_id, layer_names, shape, dtype}
    Consumer allocates GPU tensor
    Consumer  → ACK b"0"
    Producer  ncclSend on send_stream  (background thread, PUT_ASYNC)
    Consumer  ncclRecv on recv_stream  (listener thread, only on-demand)
    Consumer stores tensor in _received

Key: ncclRecv is NEVER blocking in the background — it only runs after a ZMQ
signal arrives.  Between requests the recv_stream has no pending ops, so
torch.cuda.synchronize() during CUDA graph capture is not blocked.
"""

import ctypes
import threading
from collections import deque
from typing import Optional

import msgpack
import torch
import zmq

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary, buffer_type, cudaStream_t, ncclDataTypeEnum, ncclUniqueId)
from vllm.logger import init_logger

logger = init_logger(__name__)

_NCCL_ACK_OK = b"0"
_NCCL_ACK_OOM = b"1"


class NcclTransport:
    """ZMQ signaling + NCCL data transfer.

    Producer (is_sender=True): DEALER connects, ncclSend in background thread.
    Consumer (is_sender=False): ROUTER binds, listener thread handles all msgs.
    """

    def __init__(self, is_sender: bool, host: str, port: int,
                 local_rank: int = 0):
        self.is_sender = is_sender
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.nccl = NCCLLibrary()

        self._ctx = zmq.Context()

        if is_sender:
            self._sock = self._ctx.socket(zmq.DEALER)
            self._sock.setsockopt_string(zmq.IDENTITY, f"{host}:{port}")
            self._sock.connect(f"tcp://{host}:{port}")
            logger.info("[NcclTransport] Producer DEALER connected to %s:%d",
                        host, port)

            # Get unique_id, send to consumer, then both call ncclCommInitRank
            unique_id = self.nccl.ncclGetUniqueId()
            self._sock.send(msgpack.dumps(
                {"cmd": "INIT", "unique_id": bytes(unique_id.internal)}))

            with torch.cuda.device(self.device):
                self._comm = self.nccl.ncclCommInitRank(2, unique_id, 0)
            logger.info("[NcclTransport] Producer NCCL comm established")

            self._send_stream = torch.cuda.Stream(device=self.device)

            # PUT_ASYNC: background thread does actual ncclSend
            self._send_queue_cv = threading.Condition()
            self._send_queue: deque = deque()
            self._send_thread = threading.Thread(
                target=self._send_loop, daemon=True, name="nccl-send")
            self._send_thread.start()

        else:
            self._sock = self._ctx.socket(zmq.ROUTER)
            self._sock.bind(f"tcp://*:{port}")
            logger.info("[NcclTransport] Consumer ROUTER bound on port %d",
                        port)

            self._lock = threading.Lock()
            self._received: dict[str, tuple[list[str], torch.Tensor]] = {}
            self._recv_stream = torch.cuda.Stream(device=self.device)

            # Listener handles both INIT and DATA messages
            self._listener = threading.Thread(
                target=self._listen_loop, daemon=True, name="nccl-listen")
            self._listener.start()

    # ── Producer-side ──────────────────────────────────────────────────────

    def send(self, request_id: str, layer_names: list[str],
             stacked_kv: torch.Tensor) -> None:
        """Queue KV tensor for NCCL send (PUT_ASYNC: non-blocking)."""
        assert self.is_sender
        tensor = stacked_kv.to(self.device).contiguous()
        with self._send_queue_cv:
            self._send_queue.append((request_id, layer_names, tensor))
            self._send_queue_cv.notify()

    def wait_for_sent(self) -> None:
        """Block until all queued sends have been dispatched."""
        assert self.is_sender
        with self._send_queue_cv:
            while self._send_queue:
                self._send_queue_cv.wait()

    def _send_loop(self) -> None:
        while True:
            with self._send_queue_cv:
                while not self._send_queue:
                    self._send_queue_cv.wait()
                item = self._send_queue.popleft()
                if not self._send_queue:
                    self._send_queue_cv.notify()
            self._send_one(*item)

    def _send_one(self, request_id: str, layer_names: list[str],
                  tensor: torch.Tensor) -> None:
        meta = {
            "cmd": "PUT",
            "request_id": request_id,
            "layer_names": layer_names,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        self._sock.send(msgpack.dumps(meta))

        ack = self._sock.recv()
        if ack != _NCCL_ACK_OK:
            logger.error("[NcclTransport] Consumer OOM for %s", request_id)
            return

        with torch.cuda.stream(self._send_stream):
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                1,  # consumer is rank 1
                self._comm,
                cudaStream_t(self._send_stream.cuda_stream),
            )
        self._send_stream.synchronize()
        logger.debug("[NcclTransport] Sent KV for %s shape=%s",
                     request_id, list(tensor.shape))

    # ── Consumer-side ──────────────────────────────────────────────────────

    def drain_received(self) -> dict[str, tuple[list[str], torch.Tensor]]:
        """Atomically drain all newly received KV tensors (GPU tensors)."""
        assert not self.is_sender
        with self._lock:
            result = dict(self._received)
            self._received.clear()
        return result

    def _listen_loop(self) -> None:
        """Handle INIT and PUT messages from the producer."""
        comm: Optional[object] = None

        while True:
            try:
                frames = self._sock.recv_multipart()
                # ROUTER frame layout: [identity, data]
                identity, raw = frames[0], frames[1]
                msg = msgpack.loads(raw)
                cmd = msg["cmd"]

                if cmd == "INIT":
                    uid = ncclUniqueId()
                    uid_bytes = msg["unique_id"]
                    ctypes.memmove(uid.internal, uid_bytes, len(uid_bytes))
                    with torch.cuda.device(self.device):
                        comm = self.nccl.ncclCommInitRank(2, uid, 1)
                    logger.info(
                        "[NcclTransport] Consumer NCCL comm established")

                elif cmd == "PUT":
                    if comm is None:
                        logger.error("[NcclTransport] PUT before INIT")
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        continue

                    shape = tuple(msg["shape"])
                    dtype = getattr(torch, msg["dtype"])
                    try:
                        tensor = torch.empty(shape, dtype=dtype,
                                             device=self.device)
                        self._sock.send_multipart([identity, _NCCL_ACK_OK])
                    except torch.cuda.OutOfMemoryError:
                        logger.error("[NcclTransport] OOM for %s",
                                     msg["request_id"])
                        self._sock.send_multipart([identity, _NCCL_ACK_OOM])
                        continue

                    with torch.cuda.stream(self._recv_stream):
                        self.nccl.ncclRecv(
                            buffer_type(tensor.data_ptr()),
                            tensor.numel(),
                            ncclDataTypeEnum.from_torch(dtype),
                            0,  # producer is rank 0
                            comm,
                            cudaStream_t(self._recv_stream.cuda_stream),
                        )
                    self._recv_stream.synchronize()

                    rid = msg["request_id"]
                    layer_names = msg["layer_names"]
                    logger.debug("[NcclTransport] Received KV for %s shape=%s",
                                 rid, list(tensor.shape))
                    with self._lock:
                        self._received[rid] = (layer_names, tensor)

            except Exception as e:
                logger.error("[NcclTransport] listen_loop error: %s", e)
