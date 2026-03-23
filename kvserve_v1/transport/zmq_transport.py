"""ZMQ PUSH/PULL transport for KV cache transfer between P and D instances.

Consumer (PULL) binds; Producer (PUSH) connects.
Background recv thread on consumer side is CPU-only — no CUDA stream conflict.
"""

import json
import pickle
import threading
from typing import Optional

import torch
import zmq

from vllm.logger import init_logger

logger = init_logger(__name__)


class ZmqTransport:
    """
    Producer: PUSH socket, connects to consumer's endpoint.
    Consumer: PULL socket, binds and listens; background thread receives.

    Message format (multipart):
        frame 0: JSON header bytes  {request_id, layer_names}
        frame 1: pickle'd CPU KV tensor  [num_layers, 2, num_tokens, kv_dim]
    """

    def __init__(self, is_sender: bool, host: str, port: int):
        self.is_sender = is_sender
        self._ctx = zmq.Context()

        if is_sender:
            self._socket = self._ctx.socket(zmq.PUSH)
            self._socket.connect(f"tcp://{host}:{port}")
            logger.info("[ZmqTransport] Producer PUSH connected to %s:%d", host, port)
        else:
            self._socket = self._ctx.socket(zmq.PULL)
            self._socket.bind(f"tcp://*:{port}")
            logger.info("[ZmqTransport] Consumer PULL bound on port %d", port)

            self._lock = threading.Lock()
            self._received: dict[str, tuple[list[str], torch.Tensor]] = {}

            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True, name="zmq-recv")
            self._recv_thread.start()

    def send(self, request_id: str, layer_names: list[str],
             stacked_kv: torch.Tensor) -> None:
        """Send stacked KV tensor for a request.

        Args:
            request_id: the request id.
            layer_names: ordered list of layer names (matches stacked_kv dim 0).
            stacked_kv: [num_layers, 2, num_tokens, kv_dim] on any device.
        """
        assert self.is_sender
        header = json.dumps({"request_id": request_id,
                             "layer_names": layer_names}).encode()
        kv_bytes = pickle.dumps(stacked_kv.cpu())
        self._socket.send_multipart([header, kv_bytes])
        logger.debug("[ZmqTransport] Sent KV for %s, layers=%d, shape=%s",
                     request_id, len(layer_names), list(stacked_kv.shape))

    def wait_for_sent(self) -> None:
        """No-op: ZMQ send is synchronous (copies to OS buffer immediately)."""
        assert self.is_sender

    def drain_received(self) -> dict[str, tuple[list[str], torch.Tensor]]:
        """Atomically drain all newly received KV tensors.

        Returns dict mapping request_id → (layer_names, stacked_kv CPU tensor).
        """
        assert not self.is_sender
        with self._lock:
            result = dict(self._received)
            self._received.clear()
        return result

    def _recv_loop(self) -> None:
        while True:
            try:
                header_bytes, kv_bytes = self._socket.recv_multipart()
                header = json.loads(header_bytes)
                rid = header["request_id"]
                layer_names = header["layer_names"]
                stacked_kv: torch.Tensor = pickle.loads(kv_bytes)
                logger.debug("[ZmqTransport] Received KV for %s, shape=%s",
                             rid, list(stacked_kv.shape))
                with self._lock:
                    self._received[rid] = (layer_names, stacked_kv)
            except Exception as e:
                logger.error("[ZmqTransport] recv_loop error: %s", e)
