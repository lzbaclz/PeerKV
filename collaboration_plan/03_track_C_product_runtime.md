# Track C — vLLM 多卡长上下文服务运行时产品化（6–12 个月）

> 本文件是 Track C 的**师弟可独立执行**计划。配套文件：
> - `README.md`（总览 / 三大决策 / 里程碑）
> - `01_track_A_readport_iccd.md`（跨卡 KV 传输测量论文，提供 §P3 要落地的 HBM-footprint 代价律 + do-no-harm 放置;**方向无效**，不要落地 always-PUSH）
> - `02_track_B_system_paper.md`（系统论文，**复用 Track C 的 P1–P2 作为实验章节**）
> - `04_cross_cutting.md`（CI / do-no-harm 不变式门禁 / 标定环境 / 风险登记册）
>
> **三大决策约束（本文件必须与之一致）：**
> - **D1**：Track C 的 **P1（真实 vLLM 后端）+ P2（do-no-harm 在线选择器）就是 Track B 的实验章节**。一次把真系统做出来 → 同时拿到产品和论文。
> - **D2**：唯一产品主线 = **vLLM + CUDA + A100/H100 NVLink 多卡**。MLX / GH200 / UMA / CUDA managed-memory 一律推迟到研究性 **Track D / P4**。
> - **D3**：**"Do no harm" 是写进代码 + CI 测试的硬不变式（不是文档注释）**：单卡可放下的请求**永不**变慢；peer 在跑计算 ⇒ 禁用 CFK；NVLink 带宽 < PCIe ⇒ 路由回 host。这同时是论文的 R1 红线。

---

## 0. 起点**不是零**（先读这一节，禁止重建已有件）

Track C 已有一批**已验证存在**的真实代码件。任何人**不得重写**下列文件，只能在其上演进。路径全部相对仓库根 `/home/lzq/codes/PeerKV`：

| 文件 | 现状（已核对） | 演进归属 |
|---|---|---|
| `umallm/vllm_integration/gh200_connector.py` | 真实 `class UMAGraceHopperConnector(KVConnectorBase_V1)`，已实现全套生命周期方法：`register_kv_caches` / `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` / `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` / `request_finished` / `get_stats`；内部有 `_Worker`(residency/搬运) + `_Scheduler`(计划/外部KV复用) + `UMAGraceHopperMetadata(KVConnectorMetadata)`。**这是和 MoRIIO 同一个接口的真实连接器实现** | 区域 2 的**主骨架**。要从 GH200/Grace 语义抽象出 NVLink-peer 语义（见 §区域2） |
| `umallm/vllm_integration/peerkv_register.py` | `register_peerkv()`：monkeypatch `vllm.v1.attention.backends.flash_attn` 的 `get_impl_cls()` → `PeerKVFlashAttentionImpl`。**已核对 monkeypatch 目标在 vLLM 0.8.5 存在**；import-safe；幂等；有 fallback 重绑模块级类 | 区域 2 要**去 monkeypatch 化**（脆弱，禁止进产品） |
| `umallm/vllm_integration/peerkv_attn.py` | `class PeerKVFlashAttentionImpl(FlashAttentionImpl)`，覆写 `forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None)`，有 `set_peer_kv()`、`_with_block_table()`、env 驱动的 `PeerKVLayout`。对齐 vLLM 0.8.5 源 `forward(...:481)` | 区域 2 / 区域 4 的 attention 注入点 |
| `csrc/peer_fused_decoder.cu` | **已实现 standalone bench（commit `56eafab`，534 行）**，未接入 `setup.py`。`attn_split`(split-K flash decode)/`attn_combine`/`merge2`；fork/join 用 `cudaEventRecord/StreamWaitEvent` 跨设备；**e28：跨设备 CUDA-graph capture 失败已注明**。构建：`nvcc ... -o build/peer_fused_decoder_bench`。导出符号 `peer_fused_decoder_step(...)` | 区域 4 的 kernel 起点；待接入打包 |
| `csrc/peer_fused_bench.cu` | **已工作**的多设备 fork-join 微基准（measured **1.34–1.98×**）；证明 split-K + 跨设备 event 同步可行 | 区域 4 的**正确性/性能黄金参照** |
| `umallm/elastic_policy.py` | CPU 离线选择器：`OperatingPoint{SINGLE,CFK,COPYBACK,TP,HOST,INFEASIBLE}`、`Geometry`(`llama2_7b_mha`/`gqa_8kv`)、`PeerState`、`Deployment`(`tp_capacity_tokens=253536`)、`DecodeStepModel`(`_kv_read_ms`/`_copyback_ms`/`predict_ms`/`calibrate`)、`admissible_points()`、`select_point()`（deadline-gated argmin）、`link_degraded()`（`nvlink_bw_gbps < BETA_PCIE_GBPS`）。**关键诚实点**：weight-read 非对称项是注释推导**未执行**；TP 在无实测 TPOT 时 `predict_ms` 返回 `math.inf`（admissibility-only） | 区域 5 选择器的**算法核**；要从离线 numpy 搬进在线热路径 |
| `scripts/activate_nvlink.sh` | 关 MIG → 本地双 A100 NVLink 复活（**NV12，273 GB/s，11.3× PCIe**） | 区域 9 标定/CI 前置 |
| 测试 | 仓库现有 **23 个 test 文件 / ~159 个 `test_` 函数**（含 research 测试，brief 记为 "146 research tests"） | 区域 9 在此之上加 multi-GPU/版本矩阵/perf 回归 |

> **核对命令（师弟先跑一遍确认未漂移）：**
> ```bash
> grep -n "class UMAGraceHopperConnector\|def build_connector_meta\|def start_load_kv\|def save_kv_layer" umallm/vllm_integration/gh200_connector.py
> grep -n "def register_peerkv\|get_impl_cls" umallm/vllm_integration/peerkv_register.py
> grep -n "class PeerKVFlashAttentionImpl\|def forward\|def set_peer_kv" umallm/vllm_integration/peerkv_attn.py
> grep -n "def select_point\|def admissible_points\|class DecodeStepModel\|math.inf\|def link_degraded" umallm/elastic_policy.py
> grep -n "attn_split\|peer_fused_decoder_step\|cudaEventRecord" csrc/peer_fused_decoder.cu
> ```

**结论：区域 2/4/5 都是"把已有原型产品化/在线化"，不是从零写。**

---

## 1. 编号 / 标签约定

- **任务编号**：`C<区域>.<序号>`，如 `C2.1`。
- **标签**（每个任务都打）：
  - `[论文-needed]` = Track B 论文必须依赖（属于 D1 的实验系统），优先级最高。
  - `[product-only]` = 仅产品需要，论文不依赖，**论文 freeze 后再做**。
  - `[论文最小深度]` vs `[产品完整深度]` = 同一能力的两档深度（见 §4 phase 表）。
- **里程碑**：M2（formal connector + 8B GQA decode 正确，+6wk）、M3（在线 do-no-harm 选择器 + 标定闭环，+10wk）、M4（fused peer-partial kernel + overlap，+14wk）、M5（Track B 论文用 M2–M4，+4–6mo）、M6（产品 P2–P4，+6–12mo）。
- **负责人占位**：`@师弟-runtime`（vLLM/Python 主力）、`@师弟-cuda`（CUDA/kernel）、`@导师`（gate 评审）。具体人名 [待确认]。

---

## 2. 十大产品化区域（全展开 · 每区给到能动手的 HOW）

> 区域 1–5 是论文主线（D1），写到 command/class/method 粒度。区域 6–10 是 `[product-only]`，**论文 freeze（M5）后**才铺开，这里只给骨架 + gate，避免论文期被产品化拖死。

### 区域 1 — 收敛 scope（converge scope）`[论文-needed]`

- **WHAT**：把当前散在 `umallm/vllm_integration/`（GH200/Grace 语义）+ `elastic_policy.py`（离线）+ `csrc/`（skeleton）的三块，收敛成**唯一主线**：`vLLM + CUDA + A100/H100 NVLink 双卡 peer-KV`。砍掉/隔离 GH200/UMA/MLX/managed-memory 叙事。
- **HOW（到能动手）**：
  1. 新建包目录 `umallm/peerkv/`（产品主线），把 `gh200_connector.py` 中**与 fabric 无关**的逻辑（`_Scheduler`、外部 KV 复用、metadata 计划）抽到 `umallm/peerkv/connector_core.py`；GH200/Grace 专有的 `to_grace`/`to_compressed` 等留在原文件并打 `# Track D / P4 only` banner。
  2. 写 `umallm/peerkv/README.md`：一句话 scope = "single product mainline (D2)"，列出**被 park 的清单**（MLX/GH200/UMA/managed-memory → Track D）。
  3. 在 `pyproject.toml` 增加 extras：`peerkv = ["vllm>=0.8.5", "torch>=2.4"]`（当前是 `vllm>=0.8`；serve 实验用 0.8.5，**锁 0.8.5 为下限**）。
- **交付物**：`umallm/peerkv/` 包骨架 + `connector_core.py`（仅抽取，不改行为）+ scope README。
- **验收 gate**：`import umallm.peerkv` 成功；`pytest -k "connector_core"` 现有行为不回归（对 `gh200_connector` 旧测试做 import 重定向兼容层）。
- **依赖**：无（最先做）。**工具**：`@师弟-runtime`。**风险**：抽取时破坏 `_Scheduler` 内部状态 → 用"先移动后改名、每步跑测试"的小步法。

### 区域 2 — 正式 vLLM 后端（de-monkeypatch → formal connector）`[论文-needed]` ★M2 核心

- **WHAT**：去掉 `peerkv_register.py` 的 monkeypatch，改为 vLLM 官方路径：通过 `--kv-transfer-config`（MoRIIO 同款）注册一个**正式 KV connector**，并让 attention 注入走官方 backend 选择而非改 `get_impl_cls`。补齐/打磨全套 `KVConnectorBase_V1` 生命周期方法。
- **HOW（到能动手）**：
  1. **注册路径**：实现 `umallm/peerkv/connector.py::PeerKVConnector(KVConnectorBase_V1)`（从 `UMAGraceHopperConnector` 重命名抽象而来）。通过 vLLM `KVTransferConfig` 注册：启动用
     ```
     vllm serve <model> \
       --kv-transfer-config '{"kv_connector":"PeerKVConnector","kv_role":"kv_both","kv_connector_module_path":"umallm.peerkv.connector"}'
     ```
     （字段名以本机 vLLM 0.8.5 的 `KVTransferConfig` 为准 [待确认实测]，参照 MoRIIO connector 的注册写法。）
  2. **生命周期方法**（逐个对齐 vLLM 0.8.5 签名，已存在于 `gh200_connector.py`，需打磨成 peer 语义）：
     - `get_num_new_matched_tokens(request, num_computed_tokens)` → 返回可从 peer/host 复用的 token 数（async=True）。
     - `update_state_after_alloc(request, blocks, num_external_tokens)` → 记账已分配 block 的 owner。
     - `build_connector_meta(scheduler_output) -> KVConnectorMetadata` → 产出本 step 的搬运计划（peer→local / local→peer）。
     - `start_load_kv(forward_context, **kwargs)` / `wait_for_layer_load(layer_name)` → 触发 + 等待 peer-KV 加载。
     - `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)` / `wait_for_save()` → 写出（对接 P3 放置律;方向 push/pull 任选，非性能项）。
     - `request_finished(request, block_ids)` / `get_stats()`。
  3. **去 monkeypatch attention**：保留 `PeerKVFlashAttentionImpl(FlashAttentionImpl)` 类，但**不再 monkeypatch `get_impl_cls`**。改为：connector 在 `start_load_kv` 把 peer-KV 通过 `forward_context` / connector-metadata 旁路传入，attention impl 通过官方注册的 backend 名（环境变量 `VLLM_ATTENTION_BACKEND` + 官方 backend 注册点）选择。若 0.8.5 无干净注册点，则把 peer-partial 合并**完全下放到 connector 的 load 阶段**（先把 peer-KV 物化到 local paged cache，再走原生 FlashAttention），把 fused-kernel 留给区域 4 做"快路径"，**慢路径不依赖 monkeypatch**。
  4. 保留 `register_peerkv()` 仅作**dev/debug** 开关，文件头加 `# NOT FOR PRODUCTION — use --kv-transfer-config` banner。
- **交付物**：`umallm/peerkv/connector.py`（正式 connector）；`docs/route_c/run_connector.md`（启动命令 + 字段说明）；一条 `vllm serve` 能起、能跑 8B GQA decode。
- **验收 gate（= M2 主 gate）**：
  - `vllm serve` 用 `--kv-transfer-config` 起 connector，**不打任何 monkeypatch**，8B GQA（Llama-3-8B）decode 数值正确（对照纯 vLLM 单卡输出 token-by-token，或 logits MSE < 阈值）。
  - 单卡可放下的请求：connector 走 no-op（do-no-harm，见区域 5）。
- **依赖**：区域 1。**工具**：`@师弟-runtime`，参照 `gh200_connector.py` 现成方法体 + MoRIIO 注册写法。**风险**：0.8.5 connector 接口与更高版本漂移 → 在区域 9 建版本矩阵，主线锁 0.8.5。

### 区域 3 — 生产级 KV placement manager 状态机 `[论文-needed]`

- **WHAT**：把 block 的位置/归属做成显式状态机，支持失败回退。
- **HOW**：
  1. `umallm/peerkv/placement.py`：每个 KV block 一个 `BlockState ∈ {local, peer, host, compressed, evicted}` + `owner: gpu_id` + `migratable: bool`。用 dataclass + dict 索引 `(req_id, layer, block_id) -> BlockMeta`。
  2. **状态转移**：`local→peer`(借出) / `peer→local`(copy-back) / `local→host`(offload) / `*→evicted`(请求结束)。每条转移调用区域 2 connector 的搬运原语，并写审计日志（区域 8）。
  3. **peer-state poller**：`umallm/peerkv/peer_poll.py`，周期读 peer GPU 的 `compute_busy`（SM 利用率/有无在跑 decode）、`hbm_free`、`nvlink_bw`（用 NVML / `pynvml`）。`PeerState` 复用 `elastic_policy.py` 的同名结构。
  4. **failure fallback**：peer-reset / P2P-disable / OOM 时，状态机把受影响 block 一律降级到 `host` 或 `local`，并标记请求为 "degraded but correct"（绝不丢正确性）。
- **交付物**：`placement.py` + `peer_poll.py` + 状态转移图（文字版，放 `docs/route_c/placement_fsm.md`）。
- **验收 gate**：故障注入（P2P-disable / 模拟 peer busy）下，状态机把请求降级且**输出仍数值正确**；poller 1Hz 不引入可见 TPOT 抖动（< 2%）。
- **依赖**：区域 2。**工具**：`@师弟-runtime` + `pynvml`。**风险**：poller 采样开销污染热路径 → 放独立线程 + 节流。

### 区域 4 — kernel / graph / 异步热路径 `[论文最小深度: 论文-needed][产品完整深度: product-only]`

- **WHAT**：把 `peer_fused_decoder.cu` skeleton 实现为可调用 kernel：FlashInfer/Triton paged peer-partial attention，CUDA streams/events/P2P 重叠，处理跨设备 CUDA-graph 限制。
- **HOW**：
  1. **kernel 落地**：基于 `peer_fused_bench.cu`（已工作，1.34–1.98×）的 `attn_split`(split-K) + `attn_combine` + `merge2`，把 `peer_fused_decoder_step(...)` 写完：local shard 在 cuda:0 算 partial，peer shard 在 cuda:1 算 partial，`cudaEventRecord/StreamWaitEvent` 跨设备 fork-join，`cudaMemcpyAsync(...Default...)` 把 peer 的 `(O1,lse1)`（~KB 级）拉回，`merge2` 做 log-sum-exp 合并。
  2. **接口**：`csrc/` 出 `peer_fused_decoder.so`，Python 侧 `umallm/peerkv/fused_kernel.py` 用 `torch.utils.cpp_extension` / `ctypes` 绑定；接到区域 2 的快路径。
  3. **paged 化**：把 bench 的连续 KV 改成 vLLM paged block_table 索引（参照 `peerkv_attn.py::_with_block_table`）。优先 **FlashInfer** 的 paged decode 接口；若集成成本高，先 **Triton** paged split-K 版本兜底。
  4. **CUDA-graph caveat（硬约束，写进代码注释 + 文档）**：跨设备 capture 会失败（**e28 / cudagraph.json: "capture failed"**，已在 skeleton 注明）。因此 peer-partial 路径**禁用 cudagraph**（`enforce_eager` 或对该 attention 段 graph-exclude）；单卡快路径仍可用 graph。论文里把这条作为"已知工程边界"诚实写出。
  5. **overlap**：peer-partial 的 compute 与 `(O,lse)` 回传用不同 stream + event 重叠（M4 目标）。
- **交付物**：`peer_fused_decoder.so` + `fused_kernel.py` 绑定 + 一份 microbench（对 `peer_fused_bench.cu` 复现 1.34–1.98×）。
- **验收 gate（= M4 主 gate）**：fused peer-partial decode 数值对齐慢路径（区域 2）；端到端 TPOT 相对慢路径有提升；cudagraph 禁用路径稳定不崩。
- **依赖**：区域 2（快路径挂载点）。**论文最小深度**：能跑出 1 个正确 + 有加速的 fused 点即可入论文。**产品完整深度**：FlashInfer 全量 paged + 多 stream overlap + 多 head_dim/dtype = product-only。**工具**：`@师弟-cuda`，nvcc + FlashInfer/Triton。**风险**：FlashInfer 版本与 vLLM 0.8.5 不兼容 → Triton 兜底。

### 区域 5 — do-no-harm 在线策略 `[论文-needed]` ★M3 核心（D3 的代码落地点）

- **WHAT**：把 `elastic_policy.py` 的离线 numpy 选择器搬成**在线**：启动标定 → per-box cost profile → 在线 predicted-vs-actual 闭环 + 自动降级。**D3 硬规则写进代码 + CI**。
- **HOW**：
  1. **启动标定**（`umallm/peerkv/calibrate.py`）：服务起来后跑一次微标定，填 `DecodeStepModel.calibrate(geom, single_pts, cfk_pt, copyback_pt, ...)`：实测 `single` 各上下文点、`CFK` 点、`copyback` 点的 ms/tok；测 `nvlink_bw_gbps`、`tp_capacity_tokens`。落 `~/.peerkv/box_profile.json`。
  2. **在线选择器**：把 `select_point(ctx_tokens, geom, peer, deploy, deadline_ms)` 接到 connector 的 `build_connector_meta`，每请求/每批选 corner ∈ `{single, TP, copyback, CFK, host}`。
  3. **D3 硬规则（必须在代码里 assert，不是注释）**：
     - **R1 单卡可放下 ⇒ 永不变慢**：`if ctx_tokens <= single_capacity: return SINGLE`（在选择器最前面短路；**禁止**对可单卡请求选 CFK/copyback）。
     - **peer 在跑计算 ⇒ 禁 CFK**：`admissible_points` 里 `if peer.compute_busy: drop CFK`（已在 `elastic_policy` 有 admissibility 逻辑，要接真实 `peer_poll`）。
     - **NVLink < PCIe ⇒ 路由回 host**：`if link_degraded(peer): route HOST`（`link_degraded` 已实现 `nvlink_bw_gbps < BETA_PCIE_GBPS`）。
     - **TP 无实测 TPOT ⇒ admissibility-only**：保持 `predict_ms(TP)=inf`，TP **永不**作为 latency-oracle 被选中（这是 Track B 的 R3 红线，见 `02_track_B_system_paper.md`）。
  4. **predicted-vs-actual 闭环**：每 N 步比对预测 ms/tok 与实测；偏差超阈 ⇒ `auto-downgrade`（回退到上一档更保守 corner，最终回退 SINGLE/HOST），并重标定该点。
  5. **weight-read 非对称项**：当前是注释未执行；本任务**不**强行执行它（保持诚实），仅在 P3/Track A 放置律接入后再评估是否纳入。
- **交付物**：`calibrate.py` + `box_profile.json` schema + 在线 selector 接入 + `tests/test_do_no_harm.py`（把 R1/peer-busy/link-degraded/TP-inf 四条做成断言）。
- **验收 gate（= M3 主 gate，也是 CI 门禁）**：
  - `pytest tests/test_do_no_harm.py` 四条全过；
  - 端到端 A/B：单卡可放下的请求集，PeerKV-on 的 P50/P99 TPOT 相对纯 vLLM **不退化**（do-no-harm 实测，见区域 9 / `04_cross_cutting.md`）；
  - 注入 peer-busy / 降级 NVLink，选择器实时切到 copyback / host。
- **依赖**：区域 2、区域 3（poller）。**工具**：`@师弟-runtime`。**风险**：标定漂移导致误判 → 闭环 + 自动降级兜底；标定开销 → 仅启动一次 + 增量重标。

---

### 区域 6–10 —`[product-only]`（**论文 M5 freeze 后**铺开，这里给骨架 + gate）

> 这五个区域**不进 Track B 论文**。在 M5 前只维持"不挡论文"的最小桩；M5 后按 P-阶段展开。

#### 区域 6 — 全模型覆盖 `[product-only]`
- **WHAT**：MHA/MQA/GQA；Llama/Qwen/Mixtral/DeepSeek；fp16/bf16/fp8-KV/int4；TP/PP；LoRA；spec-decode；chunked-prefill；prefix-cache；sliding-window；multi-tenant。
- **HOW（骨架）**：建 `umallm/peerkv/model_matrix.yaml` 列能力×模型矩阵；每格状态 ∈ `{supported, partial, blocked}`。论文期只需 **Llama-3-8B GQA / bf16 / 单 box** 一格为 `supported`。
- **gate**：每新增一格 → 区域 9 加一组数值正确性测试再翻 `supported`。
- **风险**：fp8-KV / int4 与 peer-partial merge 的数值合并需重验。

#### 区域 7 — serving plane `[product-only]`
- **WHAT**：OpenAI API、admission control、SLO batching、cancel、streaming、warmup、reload、quota。
- **HOW（骨架）**：复用 vLLM 原生 OpenAI server；PeerKV 只在 connector 层挂钩；admission/SLO 用区域 5 的 deadline-gated 选择器输出做准入。
- **gate**：cancel/stream 不泄漏 peer-borrowed block（接区域 10 的 KV wipe）。

#### 区域 8 — observability `[product-only / 论文期最小桩]`
- **WHAT**：搬运量、corner 命中分布、predicted-vs-actual 偏差、peer-busy 命中率、降级次数。
- **HOW（骨架）**：`get_stats()`（connector 已有）→ Prometheus exporter；论文期最小桩 = 把 corner 选择 + do-no-harm 触发记成结构化日志（**论文画 cross-over map / 选择器命中图要用**，所以这一块**论文期就要最小可用**）。
- **gate**：能导出 corner 直方图与 do-no-harm 触发计数供论文作图。**标签实为 `[论文最小深度]`**：日志足够画图即可。

#### 区域 9 — testing + CI `[论文-needed 部分 + product-only 部分]`
- **WHAT**：multi-GPU 正确性、版本矩阵、perf 回归、fault injection（OOM / peer-reset / P2P-disable / NVLink-degraded）、soak、A/B-vs-vLLM。
- **HOW**：
  1. **`[论文-needed]`**：(a) multi-GPU 数值正确性（peer-partial vs 单卡）；(b) **do-no-harm A/B-vs-vLLM**（区域 5 的 gate）；(c) fault injection 的 P2P-disable / NVLink-degraded（验证 D3 路由回 host）。这三类**纳入 CI 硬门禁**，细则在 `04_cross_cutting.md`。
  2. **`[product-only]`**：版本矩阵（vLLM 0.8.5 + 更高）、soak（24h）、perf 回归基线库。
  3. CI 前置：`scripts/activate_nvlink.sh` 确认 NV12/273GB/s；多卡测试打 `@pytest.mark.multigpu`，无 NVLink 环境自动 skip。
- **gate**：D1 三类测试在双 A100 box 全绿才允许合主线。
- **风险**：共享 box 被同租户占用导致 perf 测试 flaky（见 MEMORY：co-tenant breaks profiling）→ 加 timeout guard + 低 gpu-mem-util + 重试。

#### 区域 10 — packaging `[product-only]`
- **WHAT**：wheel/Docker、硬件支持表、topology probe、rollback、multi-tenant KV memory wipe。
- **HOW（骨架）**：`scripts/topology_probe.py`（NVML 探 NV-link 拓扑，无 NVLink → 自动退化为纯 vLLM，即"零损害降级"）；多租户 KV wipe = 请求结束 `request_finished` 里对 peer-borrowed block 显式清零。
- **gate**：无 NVLink 机器上安装即退化为纯 vLLM，零行为变化（终极 do-no-harm）。

---

## 3. 四个阶段（P1–P4）与里程碑映射

| 阶段 | 目标 | 含区域 | 论文/产品标签 | 里程碑 |
|---|---|---|---|---|
| **P1 enablement** | serve 超过单卡 KV 容量；beat host-offload；90GB 单卡 OOM 场景能服务 | 1,2,3 | `[论文最小深度]` 全程 | **M2**(+6wk) |
| **P2 do-no-harm selector** | 在线选择器 + 标定闭环；D3 四条硬规则；A/B-vs-vLLM 不退化 | 5,(3),(8最小桩),(9论文部分) | `[论文最小深度]` | **M3**(+10wk) |
| **P3 HBM-footprint 放置进 runtime** | 把 Track A 的代价律接进真实 P2P 运行时（peer-over-local-repack + chunk 限尾;方向任选，不特判 push） | 4,2,5 | `[论文最小深度]` | **M4**(+14wk) → 喂 **M5** 论文(+4–6mo) |
| **P4 GH200/UMA** | Grace/UMA/managed-memory 研究线（**Track D**） | 原 `gh200_connector.py` 专有部分 | `[product-only]` / 研究 | **M6**(+6–12mo) |

**各阶段任务清单（带标签）：**

### P1（M2）— enablement
- `C1.1` 收敛 scope 建 `umallm/peerkv/` 包 — `[论文最小深度][论文-needed]`
- `C2.1` 去 monkeypatch，`--kv-transfer-config` 注册正式 connector — `[论文-needed]` ★
- `C2.2` 补齐全套生命周期方法（peer 语义）— `[论文-needed]` ★
- `C3.1` BlockState 状态机 + peer poller — `[论文-needed]`
- **P1 出口 gate**：8B GQA decode 正确（无 monkeypatch）+ 能 serve 超单卡 KV + **beat host-offload**（端到端 TPOT 对比，数据进 Track B 实验）。

### P2（M3）— do-no-harm selector
- `C5.1` 启动标定 → `box_profile.json` — `[论文-needed]`
- `C5.2` 在线 selector 接 `build_connector_meta` — `[论文-needed]` ★
- `C5.3` D3 四条硬规则写进代码 + `test_do_no_harm.py` — `[论文-needed]` ★（D3 红线）
- `C5.4` predicted-vs-actual 闭环 + auto-downgrade — `[论文-needed]`
- `C8.1` observability 最小桩（corner 直方图 / do-no-harm 计数，供论文作图）— `[论文最小深度]`
- `C9.1` CI 论文三件套（multi-GPU 正确性 / A/B-vs-vLLM / P2P-disable 降级）— `[论文-needed]`
- **P2 出口 gate**：cross-over **MAP** 可作图（corner vs ctx_tokens）；do-no-harm A/B 实测不退化；这一套 = **Track B 实验章节主体（D1）**。

### P3（M4 → M5）— read-port push 进 runtime
- `C4.1` 实现 `peer_fused_decoder_step`（复用 bench 的 split-K/merge2）— `[论文最小深度][论文-needed]`
- `C4.2` paged 化 + FlashInfer/Triton 绑定 + cudagraph 禁用注释 — `[论文-needed]`
- `C2.3` `save_kv_layer` 的搬运按 **HBM-footprint 代价律** 选去向（NVLink peer 优先于本地满速 repack;大 handoff 分块限尾;**方向 push/pull 任选**，不做 always-PUSH 特判，见 `01_track_A_readport_iccd.md`）— `[论文-needed]`
- `C4.3` compute/transfer overlap（stream+event）— `[论文最小深度]`
- **P3 出口 gate**：fused 点正确 + 有加速；放置策略在真实运行时生效（peer handoff 对在忙 holder 的 victim ≤ ~9% 复现 Track A 量级;不在忙卡做满速 local repack）→ 数据喂 **M5** 论文。

### P4（M6）— GH200/UMA（Track D，product-only）
- `C6+`/`C7+`/`C10+` 全模型覆盖 / serving plane / packaging — `[product-only]`
- GH200 connector 专有路径回归 — 研究线，**不阻塞论文**。

---

## 4. 关键依赖图（一眼版）

```
区域1(scope) ─► 区域2(formal connector,M2) ─► 区域3(FSM/poller)
                                   │
                                   ├─► 区域5(do-no-harm selector,M3) ─► 区域9(CI论文三件套)
                                   │                                  └─► 区域8(observ最小桩→作图)
                                   └─► 区域4(fused kernel,M4) ─► 区域2.3(save_kv push,接Track A) ─► M5论文
区域6/7/10(product-only) ── 论文M5 freeze 之后 ──► M6
```

---

## 5. 跨文件 / 全局风险（详版在 `04_cross_cutting.md`）

1. **vLLM 接口漂移**：主线锁 **0.8.5**（serve 实验所用）；版本矩阵 product-only。
2. **跨设备 CUDA-graph capture 失败（e28）**：peer-partial 路径强制禁 graph，写进代码 + 论文诚实边界。
3. **共享 box co-tenant 污染 profiling**（见 MEMORY）：perf/标定加 timeout + 低 gpu-mem-util + 重试 + skip 标记。
4. **do-no-harm 是红线**：R1/peer-busy/link-degraded/TP-inf 四条**必须**在 `test_do_no_harm.py` 断言并进 CI；任何让可单卡请求变慢的改动**禁止合并**（D3）。
5. **诚实边界保留**（论文必须可见，来自 Track B）：GQA copy-back margin 仅 1.43×；fitting 单卡比 KV-parallel 快 2.7–3.8×；需 idle peer compute；N=2 单 box 无 NVSwitch。Track C 的实现**不得**用"乐观默认"掩盖这些（如默认开 CFK）。
6. **monkeypatch 残留**：`register_peerkv()` 仅 dev 用，产品路径必须 `--kv-transfer-config`；CI 加一条"产品启动不含 monkeypatch"的断言。

---

## 6. 师弟上手第一周清单（具体命令）

```bash
# 0. 确认环境与起点未漂移
bash scripts/activate_nvlink.sh && nvidia-smi topo -m            # 期望 NV12
grep -n "class UMAGraceHopperConnector\|def build_connector_meta" umallm/vllm_integration/gh200_connector.py
grep -n "def select_point\|math.inf\|def link_degraded" umallm/elastic_policy.py
pytest -q                                                         # 现有 ~159 测试基线

# 1. C1.1 建主线包
mkdir -p umallm/peerkv && : > umallm/peerkv/__init__.py
# 抽 connector_core.py（小步：先移动后改名，每步 pytest）

# 2. C2.1 先把现成 connector 用官方路径起起来（8B GQA）
vllm serve <Llama-3-8B-path-from-/public/model_zoo> \
  --kv-transfer-config '{"kv_connector":"PeerKVConnector","kv_role":"kv_both","kv_connector_module_path":"umallm.peerkv.connector"}' \
  --enforce-eager   # 先禁 cudagraph，规避 e28
```

> 模型从 `/public/model_zoo` 取（HF 走 503-prone 代理，见 MEMORY）。`--kv-transfer-config` 字段以本机 vLLM 0.8.5 `KVTransferConfig` 实测为准 **[待确认]**。

---

**唯一开放决策（与 Track A 共享，见 `01_track_A_readport_iccd.md`）**：Track A 投稿 deadline 真伪（原信 2026-06-10，今天 2026-06-08）。Track C 不受此 deadline 约束（产品 6–12 月线），但 **M5 Track B 论文** 的投稿窗口取决于 Track A 经验与目标venue（USENIX ATC / HPCA / CLUSTER / ICDCS）— 在 `README.md` 统一登记。
