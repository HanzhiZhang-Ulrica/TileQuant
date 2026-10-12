from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from triton import knobs


def all_aligned(*tensors: torch.Tensor) -> bool:
    """True when every pointer has the 16-byte alignment Triton specializes on."""

    for tensor in tensors:
        if tensor.data_ptr() & 15:
            return False
    return True


class CachedLauncher:
    """Launch a Triton kernel through its compiled handle on repeat calls.

    A decode step launches a few hundred small kernels and Triton's per-call
    binder/specialization dominates their host cost.  The first call for a
    key goes through the normal JIT path and returns the compiled kernel; later
    calls with the same key call the compiled launcher directly.

    The caller's ``key`` must determine every property Triton specializes on:
    argument dtypes, 16-byte pointer alignment, the exact integer arguments
    (==1 / divisible by 16 / 32-bit range), all constexprs and launch options.
    Pass ``key=None`` to take the JIT path (misaligned pointers, or integer
    arguments that vary freely such as prefill lengths).

    ``args`` lists every kernel parameter in declaration order, constexprs
    included; the trailing ones are named by ``constexpr_names``.
    """

    def __init__(self, fn, constexpr_names: Sequence[str]):
        self.fn = fn
        self.constexpr_names = tuple(constexpr_names)
        self.cache: Dict[Tuple[Any, ...], Any] = {}

    def __call__(
        self,
        grid: Tuple[int, int, int],
        args: Sequence[Any],
        num_warps: int,
        key: Optional[Tuple[Any, ...]],
        device_index: int,
    ) -> None:
        enter_hook = knobs.runtime.launch_enter_hook
        # Launch hooks (profilers) need Triton's launch metadata: use the JIT path.
        if key is None or getattr(enter_hook, "calls", enter_hook):
            self._jit_launch(grid, args, num_warps)
            return
        key = (device_index, num_warps) + key
        kernel = self.cache.get(key)
        if kernel is None:
            kernel = self._jit_launch(grid, args, num_warps)
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            self.cache[key] = kernel
            return
        exit_hook = knobs.runtime.launch_exit_hook
        kernel.run(
            grid[0],
            grid[1],
            grid[2],
            torch._C._cuda_getCurrentRawStream(device_index),
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            exit_hook if getattr(exit_hook, "calls", exit_hook) else None,
            *args,
        )

    def _jit_launch(self, grid, args, num_warps: int):
        n_runtime = len(args) - len(self.constexpr_names)
        constexprs = dict(zip(self.constexpr_names, args[n_runtime:]))
        return self.fn[grid](*args[:n_runtime], **constexprs, num_warps=num_warps)
