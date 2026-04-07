// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

#include "mori/collective/intra_node/kernels/vec_type.cuh"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

constexpr int kRSMaxBlocks = 80;

// SdmaReduceScatterKernel pipelines SDMA gather slices with local reduce.  Stage count
// is adaptive (see kernel): small per-rank partitions use 1 slice to avoid sync tax;
// larger ones target ~kSdmaRsPipelineMinChunkBytes per slice, capped at kSdmaRsPipelineChunks.
constexpr int kSdmaRsPipelineChunks = 8;
constexpr size_t kSdmaRsPipelineMinChunkBytes = 4ull * 1024ull * 1024ull;

// Legacy per-block barrier for ReduceScatterKernel / standalone Allreduce_sdma.
struct alignas(128) RSBarrierSignal {
  uint32_t sync[kRSMaxBlocks][8];
  alignas(128) uint32_t flag[kRSMaxBlocks];
};

// Lightweight barrier for SdmaReduceScatterKernel.
// Block 0 does the SDMA scatter + wait, then device-scope broadcasts to
// all other blocks.  Device-side generation counter → graph-safe.
struct alignas(128) CrossPeBarrier {
  alignas(128) uint32_t flag;
};

inline int getDeviceMaxBlocks() {
  int dev = 0;
  (void)hipGetDevice(&dev);
  hipDeviceProp_t prop;
  (void)hipGetDeviceProperties(&prop, dev);
  return (prop.multiProcessorCount > 0) ? prop.multiProcessorCount : 80;
}

// ============================================================================
// ReduceScatterKernel (LEGACY) — IPC reads with per-block start_sync barrier
//
// Before reading peerPtrs, every block executes a start_sync barrier
// (system-scope atomic store + device-scope atomic load) identical to
// cross_device_reduce_2stage in kernel_impl.cuh.
// This replaces the previous hipStreamSynchronize — no host blocking needed.
// ============================================================================
template <typename T>
__global__ void ReduceScatterKernel(int myPe, int npes, const application::SymmMemObjPtr srcMemObj,
                                    const application::SymmMemObjPtr dstMemObj,
                                    const application::SymmMemObjPtr barrierObj,
                                    size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t packedPerRank = elementCountPerRank / pack_size;

  if (elementCountPerRank == 0) {
    return;
  }

  // --- start_sync barrier (same as kernel_impl.cuh) --------------------------
  {
    RSBarrierSignal* self_sg = reinterpret_cast<RSBarrierSignal*>(barrierObj->localPtr);
    uint32_t next_flag = self_sg->flag[blockIdx.x] + 1;

    if (threadIdx.x < static_cast<unsigned>(npes)) {
      RSBarrierSignal* remote_sg =
          reinterpret_cast<RSBarrierSignal*>(barrierObj->peerPtrs[threadIdx.x]);

      __scoped_atomic_store_n(&remote_sg->sync[blockIdx.x][myPe], next_flag, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_SYSTEM);

      while (__scoped_atomic_load_n(&self_sg->sync[blockIdx.x][threadIdx.x], __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE) < next_flag);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      self_sg->flag[blockIdx.x] = next_flag;
    }
  }
  // --- barrier done ----------------------------------------------------------

  const size_t totalPacked = static_cast<size_t>(npes) * packedPerRank;
  const size_t start = static_cast<size_t>(myPe) * packedPerRank;
  const size_t end = (myPe == npes - 1) ? totalPacked : start + packedPerRank;

  P* __restrict__ result = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = result + start;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t idx = start + threadLinearId; idx < end; idx += threadsPerGrid) {
    const P* p0 = reinterpret_cast<const P*>(srcMemObj->peerPtrs[0]);
    A add_reg = upcast_v<typename P::type, pack_size>(p0[idx]);
    for (int pe = 1; pe < npes; ++pe) {
      const P* pp = reinterpret_cast<const P*>(srcMemObj->peerPtrs[pe]);
      packed_assign_add(add_reg, upcast_v<typename P::type, pack_size>(pp[idx]));
    }
    myDst[idx - start] = downcast_v<typename P::type, pack_size>(add_reg);
  }
}

// ---- SdmaReduceScatter subroutines (device) --------------------------------
// Reduce-scatter “gather” via SDMA: each rank sends one packed slice of every
// peer’s partition to that peer’s gather buffer (not the final AllGather).
namespace sdmars_detail {

__device__ inline bool RsPackedSliceBounds(unsigned chunk_index, size_t chunk_packed,
                                           size_t packedPerRank, size_t& k0, size_t& k1) {
  k0 = static_cast<size_t>(chunk_index) * chunk_packed;
  if (k0 >= packedPerRank) {
    return false;
  }
  k1 = k0 + chunk_packed;
  if (k1 > packedPerRank) {
    k1 = packedPerRank;
  }
  return true;
}

__host__ __device__ inline unsigned RsPipelineNumChunks(size_t chunk_bytes, size_t packedPerRank) {
  unsigned num_chunks = 1u;
  if (chunk_bytes >= kSdmaRsPipelineMinChunkBytes * 2u) {
    size_t n = (chunk_bytes + kSdmaRsPipelineMinChunkBytes - 1u) / kSdmaRsPipelineMinChunkBytes;
    if (n > static_cast<size_t>(kSdmaRsPipelineChunks)) {
      n = static_cast<size_t>(kSdmaRsPipelineChunks);
    }
    if (n < 2u) {
      n = 2u;
    }
    unsigned nc = static_cast<unsigned>(n);
    unsigned max_by_packed = static_cast<unsigned>(packedPerRank);
    if (max_by_packed < 1u) {
      max_by_packed = 1u;
    }
    if (nc > max_by_packed) {
      nc = max_by_packed;
    }
    num_chunks = (nc >= 2u) ? nc : 1u;
  }
  return num_chunks;
}

__host__ __device__ inline size_t RsPipelineChunkPacked(size_t packedPerRank, unsigned num_chunks) {
  return (packedPerRank + static_cast<size_t>(num_chunks) - 1u) / static_cast<size_t>(num_chunks);
}

__device__ inline void SignalReduceSliceDone(unsigned int* __restrict__ slice_done_counters,
                                             unsigned c) {
  if (threadIdx.x == 0) {
    atomicAdd(&slice_done_counters[c], 1u);
  }
}

__device__ inline void WaitReduceSliceDone(const unsigned int* __restrict__ slice_done_counters,
                                           unsigned c, unsigned need) {
  if (threadIdx.x == 0) {
    while (__scoped_atomic_load_n(&slice_done_counters[c], __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE) < need) {
    }
  }
}

// Block 0 only: SDMA PUT slice + peer completion flags + bump CrossPeBarrier for
// this slice.  Entire block must call; contains __syncthreads.
template <typename T>
__device__ void SdmaReduceScatter_GatherSlice_Block0(
    int myPe, int npes, const T* __restrict__ input, application::SymmMemObjPtr dstMemObj,
    application::SymmMemObjPtr flagsMemObj, CrossPeBarrier* __restrict__ barrier,
    size_t chunk_bytes_full_rank, size_t byte_offset_in_rank, size_t sub_chunk_bytes,
    uint64_t peer_flag_token, uint32_t barrier_generation, unsigned chunk_index_for_diag) {
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  const int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize;

  if (warpId < npes && laneId == 0) {
    const int destPe = warpId;

    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(const_cast<T*>(input)) +
                      static_cast<size_t>(destPe) * chunk_bytes_full_rank + byte_offset_in_rank;

    uint8_t* remoteDst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[destPe]) +
                         static_cast<size_t>(myPe) * chunk_bytes_full_rank + byte_offset_in_rank;

    anvil::SdmaQueueDeviceHandle** dh =
        dstMemObj->deviceHandles_d + destPe * dstMemObj->sdmaNumQueue;
    HSAuint64* sig = dstMemObj->signalPtrs + destPe * dstMemObj->sdmaNumQueue;
    HSAuint64* esig = dstMemObj->expectSignalsPtr + destPe * dstMemObj->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, remoteDst, sub_chunk_bytes, dh, sig, esig, dstMemObj->sdmaNumQueue,
                        0);
  }

  if (warpId < npes && laneId == 0) {
    const int destPe = warpId;
    shmem::ShmemQuietThread(destPe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &peer_flag_token, 8,
        core::atomicType::AMO_SET, destPe, 0);
  }
  __syncthreads();

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) continue;
    if (threadIdx.x == 0) {
      int spin = 0;
      bool warned = false;
      while (core::AtomicLoadRelaxed(flags + sender) < peer_flag_token) {
        if (++spin > 100000000 && !warned) {
          printf("PE %d: SdmaScatter timeout peer %d chunk %u\n", myPe, sender,
                 chunk_index_for_diag);
          warned = true;
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    __scoped_atomic_store_n(&barrier->flag, barrier_generation, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
  }
  __syncthreads();
}

// Blocks != 0: wait until block 0 has finished gather for slice c.  Thread 0
// spins; caller should __syncthreads after.
__device__ void SdmaReduceScatter_WaitGatherSliceReady_NonzeroBlock(
    CrossPeBarrier* __restrict__ barrier, uint32_t scatter_gen_base, unsigned c) {
  if (threadIdx.x == 0) {
    while (__scoped_atomic_load_n(&barrier->flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
           scatter_gen_base + c);
  }
}

// L2 coherence fix for slot[myPe] + reduce across PE slots for [k0, k1).
template <typename T>
__device__ void SdmaReduceScatter_LocalReduceSlice(int myPe, int npes,
                                                 const T* __restrict__ input,
                                                 typename packed_t<T>::P* __restrict__ buf,
                                                 typename packed_t<T>::P* __restrict__ myDst,
                                                 size_t packed_per_rank, size_t k0, size_t k1,
                                                 size_t tid, size_t stride) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack_size = P::size;

  const P* __restrict__ input_slot =
      reinterpret_cast<const P*>(input) + static_cast<size_t>(myPe) * packed_per_rank;
  for (size_t k = k0 + tid; k < k1; k += stride) {
    myDst[k] = input_slot[k];
  }
  for (size_t k = k0 + tid; k < k1; k += stride) {
    A acc = upcast_v<typename P::type, pack_size>(buf[k]);
    for (int pe = 1; pe < npes; ++pe) {
      packed_assign_add(acc, upcast_v<typename P::type, pack_size>(
                                 buf[static_cast<size_t>(pe) * packed_per_rank + k]));
    }
    myDst[k] = downcast_v<typename P::type, pack_size>(acc);
  }
}

// Single-block AllGather SDMA for one byte slice (sync reduce+AG pipeline, block 0).
template <typename T>
__device__ void SdmaAllgatherSlice_SingleBlock(int myPe, int npes,
                                               application::SymmMemObjPtr dstMemObj,
                                               application::SymmMemObjPtr flagsMemObj,
                                               size_t element_count_per_rank, size_t bytes_per_element,
                                               size_t byte_offset_in_rank, size_t sub_bytes,
                                               uint64_t peer_flag_token) {
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  const int warpId = static_cast<int>(threadIdx.x) / warpSize;
  const int laneId = static_cast<int>(threadIdx.x) % warpSize;

  uint8_t* ag_src = reinterpret_cast<uint8_t*>(dstMemObj->localPtr) +
                    static_cast<size_t>(myPe) * element_count_per_rank * bytes_per_element +
                    byte_offset_in_rank;

  if (warpId < npes && laneId == 0) {
    const int remotePe = warpId;
    uint8_t* ag_dst = reinterpret_cast<uint8_t*>(dstMemObj->peerPtrs[remotePe]) +
                     static_cast<size_t>(myPe) * element_count_per_rank * bytes_per_element +
                     byte_offset_in_rank;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dstMemObj->deviceHandles_d + remotePe * dstMemObj->sdmaNumQueue;
    HSAuint64* signals = dstMemObj->signalPtrs + remotePe * dstMemObj->sdmaNumQueue;
    HSAuint64* expectedSignals = dstMemObj->expectSignalsPtr + remotePe * dstMemObj->sdmaNumQueue;
    core::SdmaPutThread(ag_src, ag_dst, sub_bytes, devicehandles, signals, expectedSignals,
                        dstMemObj->sdmaNumQueue, 0);
  }

  if (warpId < npes && laneId == 0) {
    const int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &peer_flag_token, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) continue;
    if (threadIdx.x == 0) {
      int spinCount = 0;
      bool warned = false;
      while (core::AtomicLoadRelaxed(flags + sender) < peer_flag_token) {
        ++spinCount;
        if (spinCount > 100000000 && !warned) {
          printf("PE %d: AllGather slice timeout peer %d token %llu\n", myPe, sender,
                 static_cast<unsigned long long>(peer_flag_token));
          warned = true;
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace sdmars_detail

// ============================================================================
// SdmaReduceScatterGatherKernel — phase-1 only: pipelined RS gather (SDMA slices)
//
// Launch <<<1, 512>>>.  Leaves CrossPeBarrier at scatter_gen_base + num_chunks - 1
// and optional buffer_wbl2 so a follow-up kernel may reduce + AllGather-pipeline.
// ============================================================================
template <typename T>
__global__ void SdmaReduceScatterGatherKernel(int myPe, int npes, const T* __restrict__ input,
                                              const application::SymmMemObjPtr dstMemObj,
                                              const application::SymmMemObjPtr flagsMemObj,
                                              CrossPeBarrier* __restrict__ barrier,
                                              size_t elementCount) {
  if (blockIdx.x != 0) return;
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  if (elementCountPerRank == 0) return;

  __shared__ uint32_t s_start_flag;
  if (threadIdx.x == 0) {
    s_start_flag = barrier->flag;
  }
  __syncthreads();
  const uint32_t scatter_gen_base = s_start_flag + 1u;

  const unsigned num_chunks = sdmars_detail::RsPipelineNumChunks(chunkBytes, packedPerRank);
  const size_t chunk_packed = sdmars_detail::RsPipelineChunkPacked(packedPerRank, num_chunks);

  for (unsigned c = 0; c < num_chunks; ++c) {
    size_t k0 = 0;
    size_t k1 = 0;
    if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
      break;
    }
    const size_t byte_offset_in_rank = k0 * static_cast<size_t>(pack_size) * bytesPerElement;
    const size_t sub_chunk_bytes = (k1 - k0) * static_cast<size_t>(pack_size) * bytesPerElement;
    const uint64_t peer_flag_token = static_cast<uint64_t>(scatter_gen_base + c);
    sdmars_detail::SdmaReduceScatter_GatherSlice_Block0(
        myPe, npes, input, dstMemObj, flagsMemObj, barrier, chunkBytes, byte_offset_in_rank,
        sub_chunk_bytes, peer_flag_token, scatter_gen_base + c, c);
  }

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

// ============================================================================
// SdmaReduceScatterKernel — pipelined SDMA gather + local reduce in ONE launch
//
// Implementation is split into device helpers above (gather slice vs reduce
// slice) for readability; behavior is unchanged from the previous monolithic
// kernel.
//
// packedPerRank is split into num_chunks slices (adaptive, capped by
// kSdmaRsPipelineChunks).  Cross-PE completion uses flag tokens scatter_gen_base + c.
//
// Block 0 runs all SDMA slices back-to-back (quiet + peer wait each slice),
// bumping barrier->flag after each slice so other blocks can start the L2 fix
// and local reduce for that slice.  While blocks 1.. are reducing slice c,
// block 0 can be issuing slice c+1 on the SDMA engine — true comm/compute overlap.
// Block 0 does not run LocalReduceSlice when gridDim.x > 1; blocks 1.. cover all
// packed indices with stride (gridDim.x - 1) * blockDim.x.  If gridDim.x == 1,
// only block 0 exists and it must perform the reduce after the gather loop.
//
// num_chunks==1 matches the legacy protocol: barrier->flag goes from F0 to F0+1.
//
// Requirement: gridDim.x <= multiProcessorCount (co-resident blocks).
// ============================================================================
template <typename T>
__global__ void SdmaReduceScatterKernel(int myPe, int npes, const T* __restrict__ input,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj,
                                        CrossPeBarrier* __restrict__ barrier, size_t elementCount) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  if (elementCountPerRank == 0) return;

  __shared__ uint32_t s_start_flag;
  if (threadIdx.x == 0) {
    s_start_flag = barrier->flag;
  }
  __syncthreads();
  const uint32_t scatter_gen_base = s_start_flag + 1u;

  const unsigned num_chunks = sdmars_detail::RsPipelineNumChunks(chunkBytes, packedPerRank);
  const size_t chunk_packed = sdmars_detail::RsPipelineChunkPacked(packedPerRank, num_chunks);

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank;

  const unsigned int gdim = gridDim.x;

  if (blockIdx.x == 0) {
    for (unsigned c = 0; c < num_chunks; ++c) {
      size_t k0 = 0;
      size_t k1 = 0;
      if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
        break;
      }
      const size_t byte_offset_in_rank = k0 * static_cast<size_t>(pack_size) * bytesPerElement;
      const size_t sub_chunk_bytes = (k1 - k0) * static_cast<size_t>(pack_size) * bytesPerElement;
      const uint64_t peer_flag_token = static_cast<uint64_t>(scatter_gen_base + c);
      sdmars_detail::SdmaReduceScatter_GatherSlice_Block0(
          myPe, npes, input, dstMemObj, flagsMemObj, barrier, chunkBytes, byte_offset_in_rank,
          sub_chunk_bytes, peer_flag_token, scatter_gen_base + c, c);
    }

    if (gdim == 1u) {
      const size_t reduce_tid = threadIdx.x;
      const size_t reduce_stride = blockDim.x;
      for (unsigned c = 0; c < num_chunks; ++c) {
        size_t k0 = 0;
        size_t k1 = 0;
        if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
          break;
        }
        sdmars_detail::SdmaReduceScatter_LocalReduceSlice<T>(myPe, npes, input, buf, myDst,
                                                             packedPerRank, k0, k1, reduce_tid,
                                                             reduce_stride);
        __syncthreads();
      }
    }
  } else {
    const size_t reduce_tid =
        (static_cast<size_t>(blockIdx.x) - 1u) * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t reduce_stride =
        (static_cast<size_t>(gdim) - 1u) * static_cast<size_t>(blockDim.x);

    for (unsigned c = 0; c < num_chunks; ++c) {
      size_t k0 = 0;
      size_t k1 = 0;
      if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
        break;
      }
      sdmars_detail::SdmaReduceScatter_WaitGatherSliceReady_NonzeroBlock(barrier, scatter_gen_base,
                                                                           c);
      __syncthreads();
      sdmars_detail::SdmaReduceScatter_LocalReduceSlice<T>(myPe, npes, input, buf, myDst,
                                                           packedPerRank, k0, k1, reduce_tid,
                                                           reduce_stride);
      __syncthreads();
    }
  }

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

// ============================================================================
// SdmaReduceAllgatherPipelineKernel — phase 2: local reduce + pipelined AllGather slices
//
// Run after SdmaReduceScatterGatherKernel on the same stream.  barrier->flag must equal
// rs_end = (initial scatter_gen_base + num_chunks - 1) from the gather phase.  AG slice c
// uses peer token uint64_t(rs_end) + 1 + c; final barrier->flag must advance to the max
// AG token of this launch to keep cross-launch flag generations strictly monotonic.
// block 0 issues SDMA AG slices when gridDim > 1.
// slice_done_counters[c] (host-zeroed, length kSdmaRsPipelineChunks): non-zero blocks
// atomic-increment after each slice reduce+wbl2; block 0 spins until count == gridDim.x - 1
// before SDMA for that slice (cross-block ordering __syncthreads cannot provide).
// Launch blockDim.x >= 512 for warp-per-PE AG.
// ============================================================================
template <typename T>
__global__ void SdmaReduceAllgatherPipelineKernel(int myPe, int npes, const T* __restrict__ input,
                                                  const application::SymmMemObjPtr dstMemObj,
                                                  const application::SymmMemObjPtr flagsMemObj,
                                                  CrossPeBarrier* __restrict__ barrier,
                                                  size_t elementCount,
                                                  unsigned int* __restrict__ slice_done_counters) {
  if (elementCount == 0 || npes <= 0) return;

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;
  const size_t bytesPerElement = sizeof(T);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;
  const size_t packedPerRank = elementCountPerRank / pack_size;
  if (elementCountPerRank == 0) return;

  __shared__ uint32_t s_rs_end;
  if (threadIdx.x == 0) {
    s_rs_end = barrier->flag;
  }
  __syncthreads();
  const uint32_t rs_end = s_rs_end;

  const unsigned num_chunks = sdmars_detail::RsPipelineNumChunks(chunkBytes, packedPerRank);
  const size_t chunk_packed = sdmars_detail::RsPipelineChunkPacked(packedPerRank, num_chunks);

  const uint32_t pipeline_done_flag = rs_end + num_chunks;

  P* __restrict__ buf = reinterpret_cast<P*>(dstMemObj->localPtr);
  P* __restrict__ myDst = buf + static_cast<size_t>(myPe) * packedPerRank;

  const unsigned int gdim = gridDim.x;
  if (gdim > 1u && slice_done_counters == nullptr) return;

  if (blockIdx.x == 0 && gdim == 1u) {
    const size_t reduce_tid = threadIdx.x;
    const size_t reduce_stride = blockDim.x;
    for (unsigned c = 0; c < num_chunks; ++c) {
      size_t k0 = 0;
      size_t k1 = 0;
      if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
        break;
      }
      sdmars_detail::SdmaReduceScatter_LocalReduceSlice<T>(myPe, npes, input, buf, myDst,
                                                           packedPerRank, k0, k1, reduce_tid,
                                                           reduce_stride);
      __syncthreads();
    }
    __syncthreads();
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    if (threadIdx.x == 0) {
      asm volatile("buffer_wbl2" ::: "memory");
    }
    __syncthreads();
#endif
    const uint64_t ag_peer = static_cast<uint64_t>(rs_end) + 1ull;
    sdmars_detail::SdmaAllgatherSlice_SingleBlock<T>(
        myPe, npes, dstMemObj, flagsMemObj, elementCountPerRank, bytesPerElement, 0u, chunkBytes,
        ag_peer);
    if (threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag, pipeline_done_flag, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_DEVICE);
    }
  } else {
    for (unsigned c = 0; c < num_chunks; ++c) {
      size_t k0 = 0;
      size_t k1 = 0;
      if (!sdmars_detail::RsPackedSliceBounds(c, chunk_packed, packedPerRank, k0, k1)) {
        break;
      }
      const size_t byte_offset_in_rank = k0 * static_cast<size_t>(pack_size) * bytesPerElement;
      const size_t sub_chunk_bytes = (k1 - k0) * static_cast<size_t>(pack_size) * bytesPerElement;

      if (blockIdx.x != 0) {
        const size_t reduce_tid =
            (static_cast<size_t>(blockIdx.x) - 1u) * static_cast<size_t>(blockDim.x) + threadIdx.x;
        const size_t reduce_stride =
            (static_cast<size_t>(gdim) - 1u) * static_cast<size_t>(blockDim.x);
        sdmars_detail::SdmaReduceScatter_LocalReduceSlice<T>(myPe, npes, input, buf, myDst,
                                                             packedPerRank, k0, k1, reduce_tid,
                                                             reduce_stride);
      }
      __syncthreads();
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
      __syncthreads();
      if (blockIdx.x != 0 && threadIdx.x == 0) {
        asm volatile("buffer_wbl2" ::: "memory");
      }
#endif
      __syncthreads();
      if (blockIdx.x != 0) {
        sdmars_detail::SignalReduceSliceDone(slice_done_counters, c);
      }
      __syncthreads();
      if (blockIdx.x == 0) {
        const unsigned int need = gdim - 1u;
        sdmars_detail::WaitReduceSliceDone(slice_done_counters, c, need);
        __syncthreads();
        const uint64_t ag_peer = static_cast<uint64_t>(rs_end) + 1ull + static_cast<uint64_t>(c);
        sdmars_detail::SdmaAllgatherSlice_SingleBlock<T>(
            myPe, npes, dstMemObj, flagsMemObj, elementCountPerRank, bytesPerElement,
            byte_offset_in_rank, sub_chunk_bytes, ag_peer);
      }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __scoped_atomic_store_n(&barrier->flag, pipeline_done_flag, __ATOMIC_RELAXED,
                              __MEMORY_SCOPE_DEVICE);
    }
  }

  while (__scoped_atomic_load_n(&barrier->flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
         pipeline_done_flag) {
  }
  __syncthreads();
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
}

// ============================================================================
// AllGatherSdmaKernel — AllGather via SDMA
//
// Each rank sends its reduced shard (at dstMemObj->localPtr + myPe * stride)
// to every rank via SDMA put, then waits for all peers to finish.
// ============================================================================
template <typename T>
__global__ void AllGatherSdmaKernel(int myPe, int npes, const application::SymmMemObjPtr dstMemObj,
                                    const application::SymmMemObjPtr flagsMemObj,
                                    CrossPeBarrier* __restrict__ barrier, size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  using P = typename packed_t<T>::P;
  constexpr int pack_size = P::size;

  const size_t elementCountPerRank =
      ((elementCount / npes + pack_size - 1) / pack_size) * pack_size;

  if (elementCountPerRank == 0) {
    return;
  }

  const size_t bytesPerElement = sizeof(T);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  __shared__ uint64_t ag_token;
  if (threadIdx.x == 0) {
    ag_token = static_cast<uint64_t>(barrier->flag) + 1ULL;
    barrier->flag = static_cast<uint32_t>(ag_token);
  }
  __syncthreads();
  uint64_t flag_val = ag_token;

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // --- SDMA put: send my reduced shard to every rank -------------------------
  uint8_t* agSrcPtr = reinterpret_cast<uint8_t*>(dstMemObj->localPtr) +
                      static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;
  size_t agSendBytes = elementCountPerRank * bytesPerElement;

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    application::SymmMemObjPtr dest = dstMemObj;

    uint8_t* agDstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) +
                        static_cast<size_t>(myPe) * elementCountPerRank * bytesPerElement;

    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(agSrcPtr, agDstPtr, agSendBytes, devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
  }

  // --- Notify remote PEs that our data is in place ---------------------------
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // --- Wait for all peers to finish AllGather --------------------------------
  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }
    if (threadLinearId == 0) {
      int spinCount = 0;
      bool warned = false;
      while (core::AtomicLoadRelaxed(flags + sender) < flag_val) {
        ++spinCount;
        if (spinCount > 10000000 && !warned) {
          printf("PE %d: AllGather timeout waiting for peer %d\n", myPe, sender);
          warned = true;
        }
      }
    }
    __syncthreads();
  }

  // Flags are monotonic generation tokens (AMO_SET), so no reset is needed.
}

}  // namespace collective
}  // namespace mori
