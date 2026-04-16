# MORI Benchmark Reproduction: MI355X + AINIC (Pollara / ionic)

## Goal

Reproduce the MORI-EP and MORI-IO benchmarks documented in
[MORI-EP-BENCHMARK.md](MORI-EP-BENCHMARK.md) and
[MORI-IO-BENCHMARK.md](MORI-IO-BENCHMARK.md)
on AMD Instinct MI355X GPUs with AINIC (Pollara / ionic) RoCE NICs.

## Environment

| Item | Value |
|------|-------|
| Nodes | `smci355-ccs-aus-g12-06` (node 0), `smci355-ccs-aus-g12-14` (node 1) |
| Docker Image | `rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260331` |
| Docker Entry | `run_docker.sh` (see below) |
| MORI Git SHA (installed) | `13ec475c` |
| MORI Git SHA (latest main) | `241461c0` |
| ROCm | 7.0.0 |
| PyTorch | 2.9.0a0+git7bcbafe (HIP 7.0.51831-a3e329ad8) |
| Python | 3.10.12 |
| GPU | AMD Instinct MI355X x 8 per node |
| NIC | AINIC (Pollara / ionic) — 9 ports per node |
| RDMA Devices | `ionic_0` through `ionic_8` (RoCE over Ethernet) |

### NIC/Interface Mapping (Node 0)

| RDMA Device | Network Interface | IP | MTU | Role |
|-------------|------------------|----|-----|------|
| ionic_0 | enp9s0 | 172.1.1.0/31 | 9144 | Data |
| ionic_1 | enp25s0 | 172.225.1.0/31 | 9144 | Data |
| ionic_2 | enp105s0 | 172.161.1.0/31 | 9144 | Data |
| ionic_3 | enp121s0 | 172.193.1.0/31 | 9144 | Data |
| ionic_4 | enp137s0 | 172.129.1.0/31 | 9144 | Data |
| ionic_5 | enp153s0 | 172.97.1.0/31 | 9144 | Data |
| ionic_6 | enp196s0 | 10.235.24.254/22 | 1500 | **Management** |
| ionic_7 | enp233s0 | 172.33.1.0/31 | 9144 | Data |
| ionic_8 | enp249s0 | 172.65.1.0/31 | 9144 | Data |

> **Note:** On node 1, `ionic_2` and `ionic_3` are `PORT_DOWN`.

### Docker Entry

```bash
sudo docker run -it \
  --network=host --pid=host --ipc=host \
  --ulimit memlock=-1 \
  --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --shm-size=32g \
  --group-add video --group-add rdma \
  -e SGLANG_USE_AITER=1 \
  -v /data:/data -v /home/amd:/home/amd \
  -v /dev/infiniband:/dev/infiniband \
  rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260331 bash
```

### GPU/NIC Discovery Commands

```bash
# GPU info
rocm-smi --showid --showproductname
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# RDMA devices
ibv_devinfo | grep -E "hca_id|state|link_layer"
ls /sys/class/infiniband/

# Map RDMA device -> netdev
for dev in /sys/class/infiniband/ionic_*; do
    echo "$(basename $dev) -> $(ls $dev/device/net/)"
done

# Find management interface (for GLOO_SOCKET_IFNAME)
ip route get 8.8.8.8  # shows default route interface
```

---

## Original Doc Commands vs. Working Commands

### MORI-EP Intra-node

**Original (unchanged — works as-is):**

```bash
cd /path/to/mori
export PYTHONPATH=/path/to/mori:$PYTHONPATH
python3 tests/python/ops/bench_dispatch_combine.py
```

**Working command on MI355X:**

```bash
cd /sgl-workspace/mori
export PYTHONPATH=/sgl-workspace/mori:$PYTHONPATH
python3 tests/python/ops/bench_dispatch_combine.py
```

No environment variable changes needed. This benchmark is purely intra-node (XGMI).

### MORI-EP Inter-node

**Original doc command:**

```bash
export GLOO_SOCKET_IFNAME=ens14np0
export MORI_RDMA_DEVICES=^mlx5_0,mlx5_1

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr="10.194.129.65" --master_port=1234 \
    examples/ops/dispatch_combine/test_dispatch_combine_internode.py --bench
```

**What had to be changed:**

| Change | Original | MI355X + AINIC | Why |
|--------|----------|----------------|-----|
| `GLOO_SOCKET_IFNAME` | `ens14np0` | `enp196s0` | Different management NIC name |
| `MORI_RDMA_DEVICES` | `^mlx5_0,mlx5_1` | `ionic_0,ionic_1,ionic_4,ionic_5,ionic_7,ionic_8` (include mode) | ionic devices, not mlx5; must exclude DOWN ports and management NIC |
| `--nproc_per_node` | `8` | `1` | Script internally spawns 8 GPU processes via `torch.multiprocessing.spawn`; `WORLD_SIZE` from torchrun = number of nodes, not GPUs |
| `--bench` | `--bench` | `--cmd bench` | Script uses `--cmd` argument with choices: test, bench, stress, sweep_bench, profile |
| `MORI_DISABLE_TOPO` | (not set) | `1` | Topology detection (`MatchGpuAndNic`) does not support ionic devices |
| `--master_addr` | `10.194.129.65` | `10.235.24.254` | Node 0 management IP |

**Adapted command (node 0):**

```bash
export GLOO_SOCKET_IFNAME=enp196s0
export MORI_RDMA_DEVICES=ionic_0,ionic_1,ionic_4,ionic_5,ionic_7,ionic_8
export MORI_DISABLE_TOPO=1

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.235.24.254" --master_port=1234 \
    examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench
```

**Status: SUCCESS** — with `--num-qp 1` and single NIC (`MORI_RDMA_DEVICES=ionic_0`).

> **Key fix:** The original hang was caused by too many concurrent QPs.
> With 6 NICs × 16 ranks × 4 QPs (default), the ionic driver could not handle the QP AllToAll exchange.
> Reducing to 1 NIC and `--num-qp 1` resolves the hang.

**Working command (node 0):**

```bash
export GLOO_SOCKET_IFNAME=enp196s0
export MORI_RDMA_DEVICES=ionic_0
export MORI_DISABLE_TOPO=1

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.235.24.254" --master_port=1241 \
    examples/ops/dispatch_combine/test_dispatch_combine_internode.py --cmd bench --num-qp 1
```

### MORI-IO

**Original doc command:**

```bash
export GLOO_SOCKET_IFNAME=ens14np0

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.194.129.65" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.194.129.65"
```

**What had to be changed:**

| Change | Original | MI355X + AINIC | Why |
|--------|----------|----------------|-----|
| `GLOO_SOCKET_IFNAME` | `ens14np0` | `enp196s0` | Different management NIC name |
| `--master_addr` | `10.194.129.65` | `10.235.24.254` | Node 0 management IP |
| `--host` | Same IP on both nodes | **Each node uses its own IP** | Each node's IOEngine creates a TCP control-plane listener that must bind to a local address; using a remote IP causes "Cannot assign requested address" |
| Missing dep | (assumed installed) | `pip install prettytable` | Not in Docker image |

**Working command (node 0):**

```bash
export GLOO_SOCKET_IFNAME=enp196s0

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr="10.235.24.254" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.235.24.254" \
    --all --enable-sess --enable-batch-transfer
```

**Working command (node 1):**

```bash
export GLOO_SOCKET_IFNAME=enp196s0

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
    --master_addr="10.235.24.254" --master_port=1234 \
    tests/python/io/benchmark.py --host="10.235.58.242" \
    --all --enable-sess --enable-batch-transfer
```

> **Key insight:** `--host` must be set to **each node's own routable IP**, not the peer's IP.

---

## Observed Results

### MORI-EP Intra-node (SUCCESS)

| Metric | Value |
|--------|-------|
| Dispatch BW (avg across 10 rounds) | **369.6 GB/s** |
| Combine BW (avg across 10 rounds) | **395.1 GB/s** |
| Bytes per round | ~296 MB |
| Config | 8 GPUs, 4096 tokens, bf16, 32 experts/rank, topk=8 |

### MORI-IO RDMA Read Sweep (SUCCESS)

1 initiator + 1 target device, session enabled, batch transfer (256), default QPs.

| MsgSize (B) | Max BW (GB/s) | Avg BW (GB/s) | Min Lat (us) | Avg Lat (us) |
|-------------|---------------|---------------|--------------|--------------|
| 8 | 0.07 | 0.07 | 28.37 | 29.15 |
| 64 | 0.56 | 0.55 | 29.09 | 30.01 |
| 1024 | 7.09 | 6.73 | 36.95 | 38.94 |
| 8192 | 21.20 | 19.97 | 98.94 | 104.99 |
| 65536 | 27.88 | 27.43 | 601.77 | 611.58 |
| 262144 | 28.80 | 28.61 | 2330.54 | 2345.73 |
| 1048576 | 28.93 | 28.81 | 9278.77 | 9315.96 |

**Peak RDMA read bandwidth: 28.95 GB/s** (at 524 KB message size, 1 NIC).

### MORI-EP Inter-node (SUCCESS — single NIC, --num-qp 1)

| Metric | Dispatch | Combine |
|--------|----------|---------|
| RDMA BW | 4 GB/s | 4 GB/s |
| XGMI BW | 13-14 GB/s | 14 GB/s |
| Total BW (LL) | 16-17 GB/s | 17 GB/s |
| Avg Latency | 27,459 us | 26,711 us |
| Best Latency | 27,075 us | 26,091 us |

Config: 2 nodes × 8 GPUs = 16 ranks, 4096 tokens/rank, bf16, `--num-qp 1`, `MORI_RDMA_DEVICES=ionic_0`.

> **Note:** Hangs with 6 NICs or default `--num-qp` (4). See [Open Issues](#open-issues--limitations).

### MORI-IO RDMA Read — 6 NICs (SUCCESS)

6 initiator + 6 target devices, session enabled, batch transfer (256).

| NIC (Rank) | Peak BW @ 1MB (GB/s) | Avg BW @ 1MB (GB/s) |
|------------|---------------------|---------------------|
| ionic_0 (0) | 28.30 | 28.24 |
| ionic_1 (1) | 28.65 | 28.58 |
| ionic_4 (2) | 17.74 | 17.48 |
| ionic_5 (3) | 17.74 | 17.49 |
| ionic_7 (4) | 18.65 | 18.60 |
| ionic_8 (5) | 18.89 | 18.87 |
| **Aggregate** | **129.97** | **129.26** |

**Aggregate 6-NIC peak RDMA read bandwidth: ~130 GB/s**

> **Note:** NICs ionic_0 and ionic_1 achieve ~28.5 GB/s each (near line rate),
> while ionic_4/5/7/8 achieve ~17.5-18.9 GB/s. This may reflect PCIe topology
> or NUMA placement differences.

### Result Files

- Excel workbook: `results/mori_benchmark_results.xlsx`
- Raw logs: `results/logs/`
  - `ep_intranode.log` — EP intra-node benchmark
  - `ep_internode_success.log` — EP inter-node benchmark (1 NIC, --num-qp 1)
  - `io_node0.log` — IO RDMA Read, 1 NIC
  - `io_6nic_node0.log` — IO RDMA Read, 6 NICs aggregate

---

## Open Issues / Limitations

### 1. EP Inter-node Hangs with Multiple NICs on AINIC

The MORI-EP inter-node benchmark hangs during RDMA endpoint creation
(`context.cpp:276`, `bootNet.AllToAll`) when using multiple NICs or the default QP count.

**What works:**
- `--num-qp 1` + `MORI_RDMA_DEVICES=ionic_0` (single NIC) — **benchmark completes successfully**
- Gloo connects all 16 ranks (2 nodes x 8 GPUs) in all configurations
- `MORI_DISABLE_TOPO=1` + explicit `MORI_RDMA_DEVICES` allows static NIC matching
- MORI-IO RDMA works with 6 NICs simultaneously

**What still fails:**
- 6 NICs with `--num-qp 1`: hangs during QP exchange
- 1 NIC with default `--num-qp` (4): hangs during QP exchange
- Any configuration with > 1 NIC or > 1 QP

**Root cause:** The EP context creates many QPs (num_nics × 16 ranks × num_qp). With 6 NICs × 16 ranks × 4 QPs = 384 QPs per rank, the ionic direct-verbs path cannot handle this many concurrent QP state transitions. Even 1 NIC × 16 ranks × 4 QPs = 64 QPs is too many. Only 1 NIC × 16 ranks × 1 QP = 16 QPs works.

**Recommendation:** File an issue with ROCm/mori team to investigate ionic QP scaling limits for EP inter-node. The IO path handles QPs fine because it creates far fewer (1 per connection).

### 2. `--host` Parameter in MORI-IO is Ambiguous

The current docs show a single `--host` IP. In practice, each node must bind its own TCP listener for the IOEngine control plane. This means `--host` must be set to **each node's own IP**.

### 3. Missing `prettytable` Dependency

The IO benchmark requires `prettytable` which is not included in the Docker image.

### 4. Node 1 Has DOWN Ports

`ionic_2` and `ionic_3` are `PORT_DOWN` on node 1. These must be excluded from `MORI_RDMA_DEVICES` to avoid hangs.

---

## Suggested Doc Improvements for Upstream

1. **MORI-EP-BENCHMARK.md:**
   - Add AINIC / `ionic_*` example alongside `mlx5_*`
   - Clarify that `--nproc_per_node=1` with torchrun (not 8), since the script handles GPU parallelism internally
   - Fix `--bench` to `--cmd bench` in the inter-node example
   - Document `MORI_DISABLE_TOPO=1` requirement for non-Mellanox NICs
   - Add guidance on discovering the correct `GLOO_SOCKET_IFNAME`

2. **MORI-IO-BENCHMARK.md:**
   - Clarify that `--host` must be each node's own routable IP
   - Show separate commands for each node with different `--host` values
   - Add `prettytable` to prerequisites

3. **General:**
   - Add a "Hardware Discovery" section showing how to identify RDMA devices, map them to interfaces, and choose the right `GLOO_SOCKET_IFNAME`
   - Document which NIC vendors are supported (Mellanox, Broadcom, Pensando/ionic)
   - Note that topology detection only works for certain NIC types; others need `MORI_DISABLE_TOPO=1`
