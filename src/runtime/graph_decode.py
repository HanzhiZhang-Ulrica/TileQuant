"""Fixed-shape CUDA-graph greedy decoding for decoder-only causal language models.

Prefill and one-token decode bodies are captured once per (batch, prompt,
output) shape.  Graph replay launches exactly the kernels of the captured
eager forward, so the model arithmetic is unchanged; only host launch work is
removed.  ``close()`` detaches the decoder so one model can serve several shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, StaticCache
from transformers.masking_utils import AttentionMaskInterface


_CUDA_GRAPH_ATTENTION = "tilequant_cuda_graph"
# Prefills above this many tokens run eagerly into the static cache (their GPU
# time dwarfs launch overhead, and a captured prefill would pin all of its
# activations in the graph pool).
MAX_CAPTURED_PREFILL_TOKENS = 16384
_CUDA_GRAPH_DECODER_ATTRIBUTE = "_tilequant_cuda_graph_decoder"


def _cuda_graph_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **_: object,
) -> tuple[torch.Tensor, None]:
    """Static-cache attention with a valid-K prefix, captured per decoder."""
    decoder = getattr(module, _CUDA_GRAPH_DECODER_ATTRIBUTE)
    if decoder._attention_is_prefill:
        if query.shape[2] != decoder.prompt_tokens:
            raise RuntimeError("captured prefill query length changed")
        output = F.scaled_dot_product_attention(
            query,
            key[:, :, : decoder.prompt_tokens, :],
            value[:, :, : decoder.prompt_tokens, :],
            dropout_p=dropout,
            is_causal=True,
            scale=scaling,
            enable_gqa=True,
        )
    else:
        if query.shape[2] != 1:
            raise RuntimeError("captured decode requires one query token")
        # Cache storage is token-major for B>1: these are views, not a full
        # cache repack per decode token. Each sequence has its own fixed span.
        batch, heads, _, width = query.shape
        output = torch.ops.aten._flash_attention_forward.default(
            query.transpose(1, 2).reshape(batch, heads, width),
            key.transpose(1, 2).flatten(0, 1),
            value.transpose(1, 2).flatten(0, 1),
            decoder._decode_cu_seq_q,
            decoder._decode_cu_seq_k,
            1,
            decoder.max_cache_len,
            dropout,
            False,
            False,
            scale=scaling,
            seqused_k=decoder._decode_seqused_k,
        )[0].view(batch, 1, heads, width).transpose(1, 2)
    return output.transpose(1, 2).contiguous(), None


def _no_attention_mask(*_: object, **__: object) -> None:
    return None


@dataclass(frozen=True)
class GreedyGenerationTiming:
    """Host timestamps surrounding a synchronized fixed-length request."""

    start_s: float
    first_token_s: float
    end_s: float

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_s - self.start_s) * 1_000

    @property
    def decode_ms(self) -> float:
        return (self.end_s - self.first_token_s) * 1_000


class CudaGraphGreedyDecoder:
    """Reusable fixed-batch, fixed-bound greedy decoder for ``StaticCache``.

    ``prepare`` captures fixed prefill and one-token decode bodies outside
    request timing. Every later ``prefill`` copies a request into static CUDA
    storage then replays the reset-and-prefill graph; ``decode`` replays its
    separate graph. The decode graph owns its token and device position, so no
    scalar leaves the GPU while it advances.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        prompt_tokens: int,
        max_new_tokens: int,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        prefill_only: bool = False,
    ) -> None:
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Normal capture needs room for the first token from prefill. A decode
        # graph is only necessary when there is a following token to produce.
        if max_new_tokens < (0 if prefill_only else 1):
            minimum = 0 if prefill_only else 1
            raise ValueError(f"max_new_tokens must be at least {minimum}")
        self.model = model
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaGraphGreedyDecoder requires a CUDA device")
        text_config = model.config.get_text_config(decoder=True)
        if getattr(text_config, "_attn_implementation", None) != "sdpa":
            raise ValueError("model must use the SDPA attention implementation")
        if model.training or getattr(text_config, "attention_dropout", 0.0) != 0.0:
            raise ValueError("CudaGraphGreedyDecoder requires eval-mode zero-dropout attention")
        if getattr(text_config, "sliding_window", None) is not None:
            raise ValueError("CudaGraphGreedyDecoder requires full causal attention")
        if text_config.num_attention_heads % text_config.num_key_value_heads:
            raise ValueError("CudaGraphGreedyDecoder requires grouped-query attention")
        expected_device = self.device.index
        if expected_device is None:
            expected_device = torch.cuda.current_device()
        parameter_devices = {
            (parameter.device.type, parameter.device.index) for parameter in model.parameters()
        }
        if parameter_devices != {("cuda", expected_device)}:
            raise ValueError("CudaGraphGreedyDecoder requires a single-device CUDA model")

        self.prompt_tokens = prompt_tokens
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_cache_len = prompt_tokens if prefill_only else prompt_tokens + max_new_tokens
        self._prefill_only = prefill_only
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_cache_len)
        if batch_size > 1:
            self._allocate_token_major_cache(text_config)
        self._attention_is_prefill = True

        # These storage addresses are inputs/outputs of the captured graph and
        # must never be replaced.  ``position`` is deliberately a CUDA tensor:
        self._prompt = torch.empty((batch_size, prompt_tokens), dtype=torch.long, device=self.device)
        self._prefill_positions = torch.arange(prompt_tokens, dtype=torch.long, device=self.device)
        self._token: torch.Tensor | None = None
        self._position: torch.Tensor | None = None
        self._next_token: torch.Tensor | None = None
        self._generated: torch.Tensor | None = None
        self._decode_cu_seq_q: torch.Tensor | None = None
        self._decode_cu_seq_k: torch.Tensor | None = None
        self._decode_seqused_k: torch.Tensor | None = None
        if not prefill_only:
            # They are used both for RoPE/cache indexing and the FlashAttention
            # valid-K prefix, without exposing future static-cache slots.
            self._token = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
            self._position = torch.empty((1,), dtype=torch.long, device=self.device)
            self._next_token = torch.empty_like(self._token)
            self._generated = torch.empty(
                (batch_size, max_new_tokens), dtype=torch.long, device=self.device
            )
            self._decode_cu_seq_q = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
            self._decode_cu_seq_k = self._decode_cu_seq_q * self.max_cache_len
            self._decode_seqused_k = torch.empty((batch_size,), dtype=torch.int32, device=self.device)
        self._install_attention_backend()
        self._prefill_graph: torch.cuda.CUDAGraph | None = None
        self._prefill_logits: torch.Tensor | None = None
        self._decode_logits: torch.Tensor | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self.last_logits: torch.Tensor | None = None
        self.prefill_captured = False
        self._prepared = False

    def _allocate_token_major_cache(self, text_config) -> None:
        """Allocate every layer's K/V once in token-major storage.

        The decode attention reads (B * L, H, D) views, so batched caches must
        be token-major.  Allocating this layout up front (instead of
        re-laying out the lazily created head-major cache) avoids copying and
        fragmenting large caches.
        """
        dtype = next(self.model.parameters()).dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.bfloat16
        heads = text_config.num_key_value_heads
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        for layer in self.cache.layers:
            layer.dtype, layer.device = dtype, self.device
            layer.max_batch_size, layer.num_heads = self.batch_size, heads
            layer.k_head_dim = layer.v_head_dim = head_dim
            for name in ("keys", "values"):
                tensor = torch.zeros(
                    (self.batch_size, self.max_cache_len, heads, head_dim), dtype=dtype, device=self.device
                ).transpose(1, 2)
                torch._dynamo.mark_static_address(tensor)
                setattr(layer, name, tensor)
            layer.is_initialized = True

    def close(self, attn_implementation: str = "sdpa") -> None:
        """Detach from the model and release graphs and the static cache."""
        for module in self.model.modules():
            if getattr(module, _CUDA_GRAPH_DECODER_ATTRIBUTE, None) is self:
                delattr(module, _CUDA_GRAPH_DECODER_ATTRIBUTE)
        self.model.set_attn_implementation(attn_implementation)
        self._prefill_graph = None
        self._graph = None
        self.cache = None
        self._prefill_logits = None
        self._decode_logits = None
        self.last_logits = None

    def _install_attention_backend(self) -> None:
        """Bind this decoder's fixed cache state to its model attention modules."""
        AttentionInterface.register(_CUDA_GRAPH_ATTENTION, _cuda_graph_attention)
        # The backend never reads a mask (causal SDPA for prefill, valid-K
        # prefix for decode).  transformers 5.1 would otherwise build a default
        # mask whose checks synchronize the host, which graph capture forbids.
        AttentionMaskInterface.register(_CUDA_GRAPH_ATTENTION, _no_attention_mask)
        modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "num_key_value_groups") and hasattr(module, "layer_idx")
        ]
        if not modules:
            raise ValueError("model has no compatible attention modules")
        if any(hasattr(module, _CUDA_GRAPH_DECODER_ATTRIBUTE) for module in modules):
            raise RuntimeError("model already has a CudaGraphGreedyDecoder attached")
        for module in modules:
            setattr(module, _CUDA_GRAPH_DECODER_ATTRIBUTE, self)
        self.model.set_attn_implementation(_CUDA_GRAPH_ATTENTION)

    @classmethod
    @torch.inference_mode()
    def prepare(
        cls,
        model: torch.nn.Module,
        *,
        prompt_tokens: int = 128,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        prefill_only: bool = False,
    ) -> "CudaGraphGreedyDecoder":
        """Allocate, warm up, and capture fixed prefill and decode graphs.

        Capture setup is intentionally separate from request timing. The eager
        run establishes StaticCache storage and lazy kernel state. Real
        requests always start by replaying the reset-and-prefill graph. With
        ``prefill_only=True``, only that graph is captured.
        """
        decoder = cls(
            model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            prefill_only=prefill_only,
        )
        try:
            decoder._capture()
        except BaseException:
            # Restore the model (attention backend, decoder binding) so a failed
            # capture, e.g. out of memory, does not poison the next request shape.
            decoder.close()
            raise
        return decoder

    @torch.inference_mode()
    def _capture(self) -> None:
        decoder = self
        batch_size = self.batch_size
        prompt_tokens = self.prompt_tokens
        prefill_only = self._prefill_only
        max_new_tokens = self.max_new_tokens

        # The cache has to be initialized before CUDA graph capture. Warm each
        # captured body once so capture cannot discover a lazy allocation.
        decoder._prompt.zero_()
        decoder._prefill_body()
        if not prefill_only and max_new_tokens > 1:
            decoder._decode_body()
        torch.cuda.synchronize(decoder.device)

        # Return the eager warm-up's freed blocks to the driver; the graph pool
        # cannot reuse memory the caching allocator keeps reserved.
        torch.cuda.empty_cache()
        decoder.prefill_captured = batch_size * prompt_tokens <= MAX_CAPTURED_PREFILL_TOKENS
        if decoder.prefill_captured:
            prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(prefill_graph):
                decoder._prefill_body()
            decoder._prefill_graph = prefill_graph
        else:
            # The decode capture below expects the cache state after a prefill.
            decoder._prefill_body()

        if not prefill_only and max_new_tokens > 1:
            # The captured prefill populated cache positions [0, prompt_tokens),
            # precisely the cache state expected while capturing one decode token.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                decoder._decode_body()
            decoder._graph = graph
        torch.cuda.synchronize(decoder.device)
        decoder._prepared = True

    @torch.inference_mode()
    def _prefill_body(self) -> None:
        # Reset is captured too, which makes every replay an independent
        # request without CPU cache-clear launches. The registered attention
        # backend slices K/V to the prompt and uses SDPA's causal fast path.
        self._attention_is_prefill = True
        self.cache.reset()
        if not self._prefill_only:
            assert self._generated is not None and self._position is not None
            self._generated.zero_()
            self._position.fill_(self.prompt_tokens)
        output = self.model(
            input_ids=self._prompt,
            attention_mask=None,
            position_ids=self._prefill_positions.view(1, -1),
            past_key_values=self.cache,
            cache_position=self._prefill_positions,
            use_cache=True,
            logits_to_keep=1,
        )
        self._prefill_logits = output.logits
        if not self._prefill_only:
            assert self._token is not None and self._generated is not None
            self._token.copy_(output.logits.argmax(dim=-1))
            self._generated[:, :1].copy_(self._token)

    @torch.inference_mode()
    def _decode_body(self) -> None:
        # The fixed cache keeps its backing allocation, while FlashAttention
        # reads only [0, position + 1) through CUDA-resident prefix metadata.
        assert (
            self._position is not None
            and self._token is not None
            and self._next_token is not None
            and self._generated is not None
            and self._decode_cu_seq_q is not None
            and self._decode_cu_seq_k is not None
            and self._decode_seqused_k is not None
        )
        self._attention_is_prefill = False
        if self.batch_size == 1:
            self._decode_cu_seq_k[:1].zero_()
            self._decode_cu_seq_k[1:].copy_(self._position + 1)
        self._decode_seqused_k.copy_(self._position + 1)
        output = self.model(
            input_ids=self._token,
            attention_mask=None,
            position_ids=self._position.view(1, 1),
            past_key_values=self.cache,
            cache_position=self._position,
            use_cache=True,
            logits_to_keep=1,
        )
        self._decode_logits = output.logits
        self._next_token.copy_(output.logits.argmax(dim=-1))
        # Position P consumes the previous generated token and produces the
        # following token at generated index P - prompt_tokens + 1.
        self._generated.scatter_(
            1, (self._position - self.prompt_tokens + 1).view(1, 1).expand(self.batch_size, 1), self._next_token
        )
        self._token.copy_(self._next_token)
        self._position.add_(1)

    def _prompt_from_host(self, prompt_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if isinstance(prompt_ids, torch.Tensor):
            prompt = prompt_ids.to(dtype=torch.long)
        else:
            prompt = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        if tuple(prompt.shape) != (self.batch_size, self.prompt_tokens):
            raise ValueError(
                f"expected {(self.batch_size, self.prompt_tokens)}, got {tuple(prompt.shape)}"
            )
        return prompt

    @torch.inference_mode()
    def prefill(self, prompt_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Transfer the fixed prompt then replay the reset-and-prefill graph.

        Ordinary capture returns the persistent first generated token. With
        ``prepare(prefill_only=True)``, it instead returns the final-token
        logits from the full prompt forward; that mode has no argmax or
        generated-token storage.
        """
        if self._prefill_graph is None and not self._prepared:
            raise RuntimeError("call CudaGraphGreedyDecoder.prepare() before prefill")
        prompt = self._prompt_from_host(prompt_ids)
        self._prompt.copy_(
            prompt, non_blocking=prompt.device.type == "cpu" and prompt.is_pinned()
        )
        if self._prefill_graph is not None:
            self._prefill_graph.replay()
        else:
            self._prefill_body()
        if self._prefill_logits is None:
            raise RuntimeError("prefill graph did not retain logits")
        self.last_logits = self._prefill_logits
        if self._prefill_only:
            return self._prefill_logits
        assert self._token is not None
        return self._token

    @torch.inference_mode()
    def decode(self, steps: int | None = None) -> torch.Tensor:
        """Replay up to the prepared bound and return a GPU view of generated ids.

        The view aliases reusable decoder storage and is overwritten by the
        next :meth:`prefill` call.
        """
        if not self._prepared:
            raise RuntimeError("call CudaGraphGreedyDecoder.prepare() before decoding")
        if steps is None:
            steps = self.max_new_tokens - 1
        if self._prefill_only:
            raise RuntimeError("decode() is unavailable after prepare(prefill_only=True)")
        if not 0 <= steps <= self.max_new_tokens - 1:
            raise ValueError(f"steps must be in [0, {self.max_new_tokens - 1}]")
        if steps and self._graph is None:
            raise RuntimeError("call CudaGraphGreedyDecoder.prepare() before decoding")
        for _ in range(steps):
            assert self._graph is not None
            self._graph.replay()
        if steps:
            if self._decode_logits is None:
                raise RuntimeError("decode graph did not retain logits")
            self.last_logits = self._decode_logits
        assert self._generated is not None
        return self._generated[:, : steps + 1]

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor | Sequence[int], *, new_tokens: int | None = None) -> torch.Tensor:
        """Run one request and return its greedy ids on CUDA without synchronizing."""
        if new_tokens is None:
            new_tokens = self.max_new_tokens
        if not 1 <= new_tokens <= self.max_new_tokens:
            raise ValueError(f"new_tokens must be in [1, {self.max_new_tokens}]")
        self.prefill(prompt_ids)
        return self.decode(new_tokens - 1)

    @torch.inference_mode()
    def generate_timed(
        self, prompt_ids: torch.Tensor | Sequence[int], *, new_tokens: int | None = None
    ) -> tuple[torch.Tensor, GreedyGenerationTiming]:
        """Generate with synchronized timestamps matching the benchmark protocol."""
        if new_tokens is None:
            new_tokens = self.max_new_tokens
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self.prefill(prompt_ids)
        torch.cuda.synchronize(self.device)
        first_token = time.perf_counter()
        generated = self.decode(new_tokens - 1)
        torch.cuda.synchronize(self.device)
        end = time.perf_counter()
        return generated, GreedyGenerationTiming(start, first_token, end)
