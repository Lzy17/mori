# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Python wrapper for the LocalExpertCount kernel.

Uses the shared two-step JIT pattern from ``mori.ops._jit_loader``:
  1. ``ensure_compiled("ep_local_expert_count")``  — compile ``.hip`` → ``.hsaco``
  2. ``load_hip_module("ep_local_expert_count")``   — load into HIP runtime,
     register kernel in ``KernelRegistry`` via ``shmem_module_init``

The loaded module is cached globally in ``_jit_loader._loaded_modules`` and
shared with any other ops that load the same kernel name.

Imports are deferred to function bodies to avoid the circular-import that
arises from ``mori.cpp`` ↔ ``mori.ops`` mutual loading during package init.
"""

_KERNEL_NAME = "ep_local_expert_count"


def _ensure_kernel():
    """Compile and load the LocalExpertCount kernel on first call."""
    from mori.ops._jit_loader import ensure_compiled, load_hip_module

    ensure_compiled(_KERNEL_NAME)
    load_hip_module(_KERNEL_NAME)


def launch_local_expert_count(
    config,
    indices_ptr,
    total_recv_token_num_ptr,
    local_expert_count_ptr,
    block_num=-1,
    warp_per_block=-1,
    stream=0,
):
    """Count how many tokens are routed to each local expert.

    Args:
        config: ``EpDispatchCombineConfig`` (pybind object) with ``rank``,
            ``num_experts_per_rank``, ``num_experts_per_token``,
            ``warp_num_per_block``, and ``block_num`` set.
        indices_ptr: Device pointer (int) to a flat ``int32`` array of expert
            indices, shape ``[total_recv_tokens, num_experts_per_token]``.
        total_recv_token_num_ptr: Device pointer (int) to a 1-element ``int32``
            tensor holding the number of received tokens.
        local_expert_count_ptr: Device pointer (int) to an ``int32`` output
            array of length ``num_experts_per_rank``.  Zeroed before counting.
        block_num: Grid size override; uses ``config.block_num`` when -1.
        warp_per_block: Warps-per-block override; uses ``config.warp_num_per_block`` when -1.
        stream: HIP stream handle (raw ``hipStream_t`` as int).
    """
    import sys

    from mori.ops._jit_loader import _loaded_modules

    _ensure_kernel()
    if _KERNEL_NAME not in _loaded_modules:
        raise RuntimeError("[mori] LocalExpertCountKernel failed to load")

    sys.modules["libmori_pybinds"].launch_local_expert_count(
        config,
        indices_ptr,
        total_recv_token_num_ptr,
        local_expert_count_ptr,
        block_num,
        warp_per_block,
        stream,
    )
