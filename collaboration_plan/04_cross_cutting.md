# 04 · 跨切面工程要求（Track B 论文 × Track C 产品 共用）

> 适用范围：本文件覆盖 **Track B（系统论文）** 与 **Track C（vLLM 运行时产品化）** 共享的工程纪律。
> 核心立场（来自三大决策 D1/D2/D3）：**只建一次真实系统 → 同时拿到产品和论文的实验章节**。
> 因此本文件里写的每一条「不变量 / 指标 / 测试 / 打包」都既是产品的发布门禁，也是论文的可信度红线（R1/R3）。
>
> 阅读顺序与配套文件：
> - `README.md` —— 总览、里程碑 M1–M6、人员/负责人映射。
> - `01_track_A_readport_iccd.md` —— 跨卡 KV 传输测量论文（**方向无效**；本文件 §3.6 复用它的 HBM-footprint 代价律 + do-no-harm 放置，**不再有 always-PUSH**）。
> - `02_track_B_system_paper.md` —— 系统论文的实验设计与 honest boundaries（本文件的 A/B-vs-vLLM、cross-over MAP 就是它的实验数据来源）。
> - `03_track_C_product_runtime.md` —— 10 个产品化领域、4 个 phase；本文件是其中领域 (5)(8)(9)(10) 的**可执行细化**。
>
> 仓库：`/home/lzq/codes/PeerKV`，分支 `peerkv-parallel`。
> 已核实的现状文件（写测试/代码时直接对接，**不要凭空创建新模块**）：
> - `umallm/elastic_policy.py` —— CPU 离线 selector（admissibility + deadline-gated argmin）。`OperatingPoint = {SINGLE, COPYBACK, CFK, TP, HOST, INFEASIBLE}`；`select_point()` / `admissible_points()` 已存在。weight-read 非对称项 **仅注释、未执行**；TP 在没有实测 TPOT 时 `predict_ms`（定义于 ~135 行）返回 `math.inf`（admissibility-only，honest default，TP 的 inf 分支在 159–160 行）。
> - `umallm/vllm_integration/gh200_connector.py` —— 真实 `KVConnectorBase_V1` 实现（MoRIIO 用的同一接口）。
> - `umallm/vllm_integration/peerkv_register.py` —— monkeypatch `vllm.v1.attention.backends.flash_attn` 的 `get_impl_cls`（**脆弱，须去 monkeypatch 化**）。
> - `umallm/vllm_integration/peerkv_attn.py` —— `FlashAttentionImpl` 子类。
> - `scripts/activate_nvlink.sh` —— MIG-off → 重训 NVLink → topo 翻 `NODE→NV12`、peer BW `3.6→273 GB/s`。**本文件 §5.3 把它产品化成运行时自检 probe**。
> - `tests/`（22 个 `test_*.py`，pytest）+ `pyproject.toml` + `setup.py`（已有打包骨架）。
> - 已核实硬件常量（`elastic_policy.py` 顶部）：`BETA_HBM=773 GB/s`，`BETA_NVLINK=273 GB/s`（单向，非 vendor 600 双向），`BETA_PCIE=24 GB/s`，`C_NVLINK=23.6us > C_PCIE=12.6us`，`BORROWER_BW_RETAINED=0.9974`，`LENDER_FLOPS_RETAINED=0.6676`（lender 算力掉 33%）。
>
> vLLM 基线版本：serve 实验用 **vLLM 0.8.5**，要求 `vllm>=0.8`。

---

## §0 本文件的四块 + 一句话定位

| 小节 | 主题 | 它同时是 |
|---|---|---|
| §1 | "Do-no-harm" 硬不变量 + 4 个 CI 测试 | 产品发布门禁 + 论文 R1 红线 |
| §2 | 可观测性（在线问题 → Prometheus 指标集 → 结构化日志） | 运营能力 + 论文实验数据采集管道 |
| §3 | 测试与 CI（research-tests → product-tests 八类） | 防回归 + 论文“可复现”证据 |
| §4 | 打包与运维（wheel/Docker/拓扑探针/回滚/多租户擦除） | 上线交付 + 论文 artifact |

**贯穿全文件的一条铁律（D3）**：
> “不要比基线慢 / 不要做有害的事” 必须是**代码里的断言 + CI 里的测试**，**不能只是文档里的一行注释**。任何把不变量写成“注意：理论上不应…”而没有对应 `assert` 和 `test_*` 的 PR，一律打回。

---

## §0.5 实施进度对照表（2026-06-09；risk 9.C.1 —— 规划→代码单一真理源）

> 本表是**规划与代码的对账**。状态：`[DONE]` 已实施可用 · `[STUB]` 文件存在但未接线 ·
> `[TODO]` 仅规划无代码。文档写了但代码没有的**必须**标 `[TODO]/[STUB]`，禁止默认“已实现”。
> 改动一个交付物时同步改这一行。

| 交付物 | 规划位置 | 状态 | 落点 |
|---|---|---|---|
| pytest markers (gpu/perf/soak) | §3.9 | `[DONE]` | `pyproject.toml` |
| CPU CI（collect 0-error + not-gpu） | §3.9 | `[DONE]` | `.github/workflows/ci.yml` |
| box-idle probe（co-tenant 门） | §5.3 / 9.A.4 | `[DONE]` | `umallm/observability/box_probe.py`（含 `gate_or_skip()`，已接 g1–g10 + `e2e_handoff_stressor.py` 入口；逃生口 `PEERKV_SKIP_IDLE_PROBE=1`） |
| topology probe（MIG/NVLink 自检） | §4.3 | `[DONE]` | `umallm/observability/topology_probe.py` |
| Prometheus 指标集 | §2.2 | `[STUB]` 定义齐、未接热路径 | `umallm/observability/metrics.py` |
| `enforce_do_no_harm` + 4 CI 测试 | §1 | `[TODO]` | `umallm/runtime/selector.py`(stub) |
| 在线 KV 放置状态机 | MASTER_PLAN §3.2 | `[TODO]` | `umallm/runtime/kv_manager.py`(stub) |
| CUDA 在线标定闭环 | MASTER_PLAN §3.3 | `[TODO]` | `umallm/runtime/calibration.py`(stub) |
| 正式 connector（去 monkeypatch） | §3.2 / 9.A.1 | `[TODO]` 现为 monkeypatch | `umallm/vllm_integration/peerkv_register.py` |
| read-vs-write 机制实验（建议③） | 新 | `[DONE]` 脚本就绪，待跑 | `experiments/g5_read_write_split.py` |
| e2e 5-condition（+local，建议①） | 新 | `[DONE]` 脚本就绪，待跑 | `experiments/e2e_run.sh` + `e2e_handoff_stressor.py` |
| onboarding 自动化 | 9.C.3 | `[DONE]` | `scripts/onboard.sh` + `AGENTS.md` |
| GPU CI（self-hosted dual-A100） | §3.9 | `[TODO]` 需 runner | — |
| 回滚开关 `PEERKV_ENABLED` | §4.4 | `[TODO]` | — |
| 多租户 KV 擦除 | §4.5 | `[TODO]` | `tests/test_kv_wipe.py`(未建) |

---

## §1 "Do-no-harm" 硬不变量（代码强制 + CI 测试 + 论文 R1）

### 1.1 WHAT（做什么）
把 D3 的三条规则固化成一个**单点可调用的代码不变量**，并用 4 个 CI 测试钉死它。三条规则（论文里就是 R1 红线）：

1. **R1-fit**：单卡放得下的请求，**永远不能被做慢**。即只要 `ctx_tokens <= single_gpu_capacity_tokens`，selector 必须返回 `SINGLE`（绝不 CFK/COPYBACK/TP）。
2. **R1-busy**：peer 在算（compute-busy）时，**禁止 CFK**（CFK 需要 lender 出算力，而 lender 算力掉 33% → 伤到 peer 的在跑请求）。peer-busy 时只允许“借 HBM+链路”的 COPYBACK 或回退 HOST。
3. **R1-route**：当 `NVLink 有效带宽 < PCIe 有效带宽`（NVLink 降级/未训练/P2P 关闭）时，**禁止走 peer NVLink 路径**，路由回 host/PCIe。

> ⚠️ **更新（Track A 重定位）**：旧的"方向规则（always-PUSH）作为 R1 第四条"已**删除**——Track A 受控实测证明跨卡传输**方向（push/pull）对 victim 与带宽都无影响**，不是 do-no-harm 轴。取而代之的真实放置守则（来自 Track A 的 HBM-footprint 代价律，见 §3.6）：**(a)** 优先 NVLink peer handoff（对在忙 holder do-no-harm，~9%/~3ms）；**(b) 禁止在忙卡上做满速本地 HBM repack**（~127%，真正的 harm 源，例如在线 re-quant/compaction）；**(c)** 大 handoff 分块限尾；传输方向按软件便利任选。

### 1.2 HOW（怎么做，到能动手）

**Step 1 — 落一个单一真理源函数**（放到 `umallm/elastic_policy.py`，与现有 `select_point` 同模块，复用其 `OperatingPoint`/`PeerState`/`Geometry`）：

```python
# umallm/elastic_policy.py  (新增，紧挨 select_point)
class DoNoHarmViolation(RuntimeError):
    """Raised when a chosen OperatingPoint would break a do-no-harm rule.
    This is a HARD invariant: it is asserted in the hot path and tested in CI."""

def enforce_do_no_harm(point: "OperatingPoint",
                       ctx_tokens: int,
                       geom: "Geometry",
                       peer: "PeerState",
                       link: "LinkState") -> "OperatingPoint":
    # R1-fit: 单卡放得下 => 必须 SINGLE
    if ctx_tokens <= geom.single_gpu_capacity_tokens and point is not OperatingPoint.SINGLE:
        raise DoNoHarmViolation(f"R1-fit: fits single ({ctx_tokens}<={geom.single_gpu_capacity_tokens}) but chose {point}")
    # R1-busy: peer 在算 => 禁 CFK
    if point is OperatingPoint.CFK and peer.compute_busy:
        raise DoNoHarmViolation("R1-busy: peer compute-busy, CFK forbidden (lender loses 33% FLOPs)")
    # R1-route: NVLink 有效带宽 < PCIe => 禁 peer-NVLink 路径
    if point in (OperatingPoint.CFK, OperatingPoint.COPYBACK) and link.nvlink_eff_gbps < link.pcie_eff_gbps:
        raise DoNoHarmViolation(f"R1-route: nvlink {link.nvlink_eff_gbps}<pcie {link.pcie_eff_gbps} GB/s, route host")
    # (R1-direction REMOVED: Track A's controlled measurement shows push/pull are
    #  equivalent in bandwidth and victim cost -- direction is NOT a do-no-harm axis.
    #  The real harm axis is HBM-read footprint: do not full-rate local-repack on a
    #  busy GPU; prefer NVLink peer handoff; chunk large handoffs. transfer_dir is a
    #  recorded field, not a constraint.)
    return point
```

- 需要新增的小数据类（同模块）：`LinkState(nvlink_eff_gbps: float, pcie_eff_gbps: float, transfer_dir: str)`；`Geometry` 已有就补 `single_gpu_capacity_tokens` 字段；`PeerState` 已有就补 `compute_busy: bool`（若已有等价字段则复用，**不要重复造**）。
- **关键纪律**：`select_point()` 的返回值在交给运行时之前，**必须**过一遍 `enforce_do_no_harm()`。也就是 selector 的出口只有一条，且这条出口带断言。

**Step 2 — 在 vLLM 热路径上挂同一个断言**：在 `peerkv_attn.py`（`FlashAttentionImpl` 子类）真正决定“本步去哪算/搬哪块 KV”的地方，调用 `enforce_do_no_harm(...)`。production 模式下 violation = 立刻 fallback 到 `SINGLE` 并打 `WARN` 日志 + Prometheus counter `peerkv_do_no_harm_violations_total{rule=...}` 自增（见 §2.2）；CI/test 模式（环境变量 `PEERKV_STRICT=1`）下 violation = 直接抛 `DoNoHarmViolation`（让测试红）。

**Step 3 — 4 个 CI 测试**（新建 `tests/test_no_harm.py`，pytest，CPU 即可跑前三个，第四个走 GPU job 见 §3）：

```
tests/test_no_harm.py
  ├─ test_no_harm_single_fits
  ├─ test_no_harm_peer_busy
  ├─ test_no_harm_nvlink_degraded
  └─ test_no_harm_ab_vs_vllm        # 标记 @pytest.mark.gpu, 进多GPU job
```

### 1.3 四个 CI 测试的精确规格

#### (a) `test_no_harm_single_fits`
- **WHAT**：扫一遍“单卡放得下”的 ctx 区间，断言 selector 永远只回 `SINGLE`。
- **HOW**：参数化 `ctx_tokens ∈ {1, 1024, geom.single_gpu_capacity_tokens-1, geom.single_gpu_capacity_tokens}`，对每个 ctx 配 5 种随机 `PeerState`（含 idle 和 busy），断言 `select_point(...)` 经 `enforce_do_no_harm` 后 `== SINGLE`。再做一个**反向 fuzz**：构造一个故意返回 `CFK` 的 mock selector，断言 `enforce_do_no_harm` 抛 `DoNoHarmViolation`（证明守门有效，不是空守门）。
- **验收 gate**：100% 通过；fuzz 分支必须真的抛异常（用 `pytest.raises`）。

#### (b) `test_no_harm_peer_busy`
- **WHAT**：peer compute-busy 时，selector 输出集合 ∩ `{CFK}` = ∅。
- **HOW**：`PeerState(compute_busy=True)`，`ctx_tokens > single_gpu_capacity_tokens`（强制要 off-card）。断言：(i) `admissible_points(...)` 不含 `CFK`；(ii) `select_point` 落在 `{COPYBACK, HOST, TP, INFEASIBLE}`；(iii) `enforce_do_no_harm(CFK, peer_busy)` 抛异常。再加一条**数值对账**：用 `LENDER_FLOPS_RETAINED=0.6676` 验证 CFK 在 busy 下的 lender 代价被正确判为 inadmissible（保证拦的是真实物理而不是魔法数）。
- **验收 gate**：三条 sub-assert 全过。

#### (c) `test_no_harm_nvlink_degraded`
- **WHAT**：NVLink 有效带宽掉到 < PCIe 时，禁 peer-NVLink，路由 host。
- **HOW**：`LinkState(nvlink_eff_gbps=20.0, pcie_eff_gbps=24.0, ...)`（20<24，模拟链路降级/部分链路 down）。断言：(i) selector 不选 `CFK`/`COPYBACK`（它们要走 NVLink）；(ii) 回退到 `HOST`；(iii) 健康 link（`nvlink=273, pcie=24`）时才允许 `COPYBACK/CFK`。（旧的"`transfer_dir="pull"` 必抛异常"用例**已删**——方向不再是不变量，见 §1/§3.6 的 Track A 重定位。）
- **验收 gate**：降级与健康两种 LinkState 的分支都对，方向 fuzz 抛异常。

#### (d) `test_no_harm_ab_vs_vllm`（GPU job，是论文 R1 的实测证据）
- **WHAT**：真实双 A100 上跑一组**单卡放得下**的请求，对比“原版 vLLM 0.8.5”与“PeerKV-connector 开启”的 **TPOT/TTFT**，断言 PeerKV **不慢于** baseline（允许误差带）。这条把 D3 的“don't be slower than baseline”从文档变成测试。
- **HOW**：
  1. 用 `scripts/activate_nvlink.sh` 自检确保 NV12（见 §5.3 probe），否则 `pytest.skip("NVLink not NV12")`。
  2. baseline：原版 vLLM serve（关 PeerKV connector）跑固定 workload（Llama-3-8B GQA decode，固定 seed、固定 batch、固定输入/输出长度，**全部 ctx ≤ single_gpu_capacity_tokens**），各请求重复 ≥5 次取 median。
  3. treatment：同 workload 同 seed，PeerKV connector 开启（formal backend，**非 monkeypatch**）。
  4. 断言：`tpot_p50_peerkv <= tpot_p50_vllm * (1 + EPS)` 且 `ttft_p50_peerkv <= ttft_p50_vllm * (1 + EPS)`，`EPS = 0.03`（3% 噪声带，[待确认] 用首轮 5 次重复的方差校准这个 EPS）。
  5. 输出 A/B JSON 落到 `tests/_artifacts/ab_vs_vllm.json`，供论文画图（§2/§3.8）。
- **验收 gate**：median TPOT/TTFT 不退化（在 EPS 内）；且 `peerkv_do_no_harm_violations_total == 0`。**这是 M1/M2 的发布门禁，也是论文 §evaluation 的第一张表。**
- **风险**：共享机器有 co-tenant 会污染时延（见 MEMORY：co-tenant breaks vLLM profiling）→ 测试前用 §2 的 `peerkv_peer_compute_busy` 指标 gate，busy 则 skip 并标记 flaky，不算失败。

### 1.4 交付物
- `umallm/elastic_policy.py`：`enforce_do_no_harm()` + `DoNoHarmViolation` + `LinkState`。
- `tests/test_no_harm.py`：4 个测试。
- CI 工作流里 `test_no_harm_*` 列为 **required check**（§3.9）。
- 论文：R1 章节直接引用这 4 个测试名 + `ab_vs_vllm.json` 的表。

### 1.5 依赖 / 工具 / 负责人 / 风险
- **依赖**：M2（formal vLLM connector，去 monkeypatch）才能跑 (d)；(a)(b)(c) 在 M1 就能上（纯 CPU）。
- **工具**：pytest、vLLM 0.8.5、`scripts/activate_nvlink.sh`、Prometheus client。
- **负责人**：师弟（policy + CPU 测试）；GPU 测试 (d) 需占用双 A100 窗口（与 Track A 实验错峰）。
- **风险**：① `enforce_do_no_harm` 若只挂在 selector 出口、热路径绕过它 → 不变量失效。**Mitigation**：热路径的搬运调用也强制过它（Step 2），并在 §3 的 fault-injection 里注入“非法 point”验证拦截。② EPS 带太松会放过真实退化 → 用方差校准并写进 commit message。

---

## §2 可观测性（Observability）

### 2.1 WHAT — 必须在线回答的问题清单
可观测性不是“多打点”，是“能回答下面这些运营 + 论文问题”。每个问题映射到 §2.2 的指标。

| # | 在线问题 | 用到的指标 | 谁关心 |
|---|---|---|---|
| Q1 | 现在每个请求落在哪个 corner？分布如何？ | `peerkv_selected_point_total{point}` | 运营 + 论文(corner 分布) |
| Q2 | do-no-harm 有没有被违反过？哪条规则？ | `peerkv_do_no_harm_violations_total{rule}` | 发布门禁(必须恒为0) |
| Q3 | PeerKV 相对单卡 baseline 的 TPOT/TTFT 是变好还是变差？ | `peerkv_tpot_seconds`,`peerkv_ttft_seconds`(带 `mode` label) | R1 红线 + SLO |
| Q4 | cross-over 点在哪？（ctx 多大时 off-card 才开始划算） | `peerkv_ctx_tokens` histogram × `selected_point` | 论文 cross-over MAP |
| Q5 | peer 当前 busy 吗？借不借得到算力/HBM？ | `peerkv_peer_compute_busy`,`peerkv_peer_hbm_free_bytes` | R1-busy + 调度 |
| Q6 | NVLink 链路健康吗？有效带宽多少？降级了吗？ | `peerkv_link_eff_gbps{fabric}`,`peerkv_nvlink_lanes_up` | R1-route + 探针 |
| Q7 | 跨 GPU 搬了多少字节(HBM-read footprint)？耗时多少？方向只作诊断记录 | `peerkv_xfer_bytes_total{dir}`,`peerkv_xfer_seconds{dir}` | 代价律(bytes)+ 容量规划 |
| Q8 | 单卡 OOM(90GB) 时有没有成功 enable（serve beyond single card）？ | `peerkv_single_card_oom_total`,`peerkv_enabled_beyond_single_total` | P1 enablement KPI |
| Q9 | TP-2 作为 admissibility/upper-bound 时容量够不够？ | `peerkv_tp_capacity_tokens`,`peerkv_admissible_points` | R3(TP 不做时延 oracle) |
| Q10 | SLO（deadline）有没有 miss？哪些请求 miss？ | `peerkv_slo_miss_total`,`peerkv_deadline_slack_seconds` | SLO 批处理 |

### 2.2 HOW — 最小 Prometheus 指标集
用 `prometheus_client`（Python），放到新模块 `umallm/observability/metrics.py`，由 connector / selector / probe 各处 import 同一组单例。**最小集（不要膨胀）**：

```python
# umallm/observability/metrics.py
from prometheus_client import Counter, Gauge, Histogram

# --- selector / do-no-harm ---
SELECTED_POINT = Counter("peerkv_selected_point_total", "corner chosen", ["point"])
DO_NO_HARM_VIOL = Counter("peerkv_do_no_harm_violations_total", "invariant breaches", ["rule"])
ADMISSIBLE_POINTS = Gauge("peerkv_admissible_points", "size of admissible set (last req)")

# --- latency (mode = single|peerkv, point=...) ---
TPOT = Histogram("peerkv_tpot_seconds", "time per output token", ["mode", "point"],
                 buckets=(.005,.01,.02,.04,.08,.16,.32,.64,1.28))
TTFT = Histogram("peerkv_ttft_seconds", "time to first token", ["mode"],
                 buckets=(.05,.1,.2,.4,.8,1.6,3.2,6.4))
CTX_TOKENS = Histogram("peerkv_ctx_tokens", "context length at admission",
                       buckets=(8e3,16e3,32e3,64e3,128e3,253536,5e5))  # 253536 = TP-2 cap

# --- peer / link health ---
PEER_BUSY = Gauge("peerkv_peer_compute_busy", "1 if peer GPU is computing", ["peer"])
PEER_HBM_FREE = Gauge("peerkv_peer_hbm_free_bytes", "peer free HBM", ["peer"])
LINK_EFF_GBPS = Gauge("peerkv_link_eff_gbps", "measured one-way eff bw", ["fabric"])  # nvlink|pcie
NVLINK_LANES_UP = Gauge("peerkv_nvlink_lanes_up", "# NVLink lanes trained (expect 12)")

# --- transfer (dir = push|pull) ---
XFER_BYTES = Counter("peerkv_xfer_bytes_total", "cross-GPU KV bytes", ["dir"])
XFER_SECONDS = Histogram("peerkv_xfer_seconds", "cross-GPU transfer time", ["dir"],
                         buckets=(1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2))

# --- enablement / SLO ---
SINGLE_CARD_OOM = Counter("peerkv_single_card_oom_total", "single-card KV OOM events")
ENABLED_BEYOND = Counter("peerkv_enabled_beyond_single_total", "served because off-card KV worked")
SLO_MISS = Counter("peerkv_slo_miss_total", "requests missing deadline", ["point"])
DEADLINE_SLACK = Histogram("peerkv_deadline_slack_seconds", "deadline - predicted (neg=miss)",
                           buckets=(-1,-.5,-.1,0,.1,.5,1,5))
```

- **暴露方式**：vLLM serve 进程内起 `prometheus_client.start_http_server(PORT)`（默认 `:9400`，可配）。Docker 暴露该端口（§4）。
- **打点位置**（精确）：
  - `SELECTED_POINT` / `ADMISSIBLE_POINTS`：`select_point()` 返回处。
  - `DO_NO_HARM_VIOL`：`enforce_do_no_harm` 的每个 `raise`/production-fallback 分支（label `rule ∈ {fit,busy,route,direction}`）。
  - `TPOT`/`TTFT`/`CTX_TOKENS`：connector 在每步/每请求出口，`mode="peerkv"` 或 `"single"`，`point` 取本步 corner。
  - `LINK_EFF_GBPS`/`NVLINK_LANES_UP`：启动探针（§5.3）+ 周期性后台 refresh（默认 30s）。
  - `XFER_*`：`peerkv_attn.py` 真正发起 P2P DMA 处，`dir` 记录本次实际发起方向（push/pull 任选，非性能项）;`bytes` 才是与 victim 代价相关的量（HBM-read footprint）。

### 2.3 HOW — 结构化日志（structured logging）
- 用 JSON-lines（`logging` + `python-json-logger` 或自带 `json.dumps`），每请求一条 `decision` 记录，字段：
  `{ts, req_id, ctx_tokens, single_capacity, admissible:[...], selected_point, peer_busy, nvlink_eff_gbps, pcie_eff_gbps, transfer_dir, do_no_harm_ok, predicted_tpot_ms, measured_tpot_ms, slo_ms, slo_met}`。
- 每条 violation 一条 `do_no_harm_violation` WARN 记录，字段含 `rule`、`req_id`、`fallback_point`。
- **纪律**：日志里出现的每个判定字段，都要能跟 §2.2 的某个指标对账（指标用于聚合/告警，日志用于单请求复盘 + 论文逐请求数据）。
- **论文用途**：`decision` 日志直接 dump 成 dataframe → 画 cross-over MAP（Q4）、corner 分布（Q1）、predicted-vs-measured 回归（校准闭环 M3）。

### 2.4 交付物 / 验收 gate / 依赖 / 风险
- **交付物**：`umallm/observability/metrics.py`、结构化 logger、一份 `docs/observability.md`（指标→问题映射表，即 §2.1）。
- **验收 gate**：① serve 起来后 `curl :9400/metrics` 能看到全部 13 个指标；② 跑一遍 §1.3(d) 的 A/B，`peerkv_do_no_harm_violations_total == 0`、`peerkv_selected_point_total{point="single"} == 请求数`（因为全是 single-fit）；③ `decision` 日志能被一行 pandas 读成表。
- **依赖**：依赖 §1 的 selector/不变量已落地。
- **风险**：指标 cardinality 爆炸（label 太多）→ 已把 label 控制在 `point/mode/dir/rule/fabric/peer` 这几个低基数维度，**禁止把 req_id 当 label**（req_id 只进日志）。

---

## §3 测试与 CI（research-tests → product-tests）

### 3.0 立场
现有 `tests/` 里 22 个是 **research-tests**（验证“某个想法成不成立”，CPU 模型 + smoke）。产品要再叠一层 **product-tests**（验证“真系统在真硬件上不退化、不出错、能复现”）。**“别比基线慢” = product-test，不是文档行。** 下面八类全部要有对应 `tests/` 文件或 CI job。

### 3.1 多 GPU 正确性（multi-GPU correctness）
- **WHAT**：off-card / CFK / COPYBACK 路径算出的 logits 与单卡参考**逐元素一致**（确定性 exactness）。
- **HOW**：新建 `tests/test_multigpu_correctness.py`（`@pytest.mark.gpu`）。固定 seed，跑一条 ctx > single-capacity 的请求，分别用 (i) 单卡参考（强制把 KV 全放本卡的 oracle 路径）、(ii) COPYBACK、(iii) CFK，断言三者输出 logits `torch.allclose(rtol=0, atol=tol)`。`tol`：fp16 用 `atol=1e-2`（[待确认] 用首轮跑校准，CFK 的 partial-(O,lse) 合并理论上应 bit-近似）。复用 `tests/test_multigpu.py`/`test_peer_parallel.py` 已有 fixture。
- **验收 gate**：三路径在 tol 内一致；任一路径 NaN/Inf 即红。

### 3.2 vLLM 版本矩阵（vLLM version matrix）
- **WHAT**：connector 在多个 vLLM 版本上能 import + 起 serve + 跑通最小请求。
- **HOW**：CI matrix `vllm ∈ {0.8.5(锁定基线), 0.8.x-latest, 0.9.x若存在}`（[待确认] 实际可用版本以发布时为准）。每个版本跑 `tests/test_e2e_runtime.py` 的 smoke。**重点防 monkeypatch 脆性**：`peerkv_register.py` 现在 monkeypatch `flash_attn.get_impl_cls`，版本一变就崩 → 测试矩阵第一个目标就是逼出“去 monkeypatch、走 formal `KVConnectorBase_V1` 注册”（M2 任务）。
- **验收 gate**：锁定版本 0.8.5 必须全绿（required）；其它版本绿/黄都记录，红则开 issue 不阻塞 0.8.5。

### 3.3 CUDA / NCCL / torch 矩阵
- **WHAT**：在不同 CUDA/NCCL/torch 组合下编译 `csrc/*.cu`（peer_fused_decoder）+ P2P 跑通。
- **HOW**：CI matrix `(torch, cuda, nccl)` 至少 2 组（当前生产组合 + 一个 next）。每组 `pip install -e .` 编译 CUDA 扩展 + 跑 `tests/test_multigpu_correctness.py` 的最小 case。记录 `nvidia-smi`、`nvcc --version`、`torch.version.cuda`、`torch.cuda.nccl.version()` 到 artifact。
- **验收 gate**：生产组合编译 + P2P + 正确性全过；矩阵其它组合编译过即可。
- **风险**：本地 HF 走 503-prone 代理（MEMORY）→ CI 模型一律从 `/public/model_zoo` 取，**禁走 HF 网络**。

### 3.4 性能回归（perf-regression）—— “别比基线慢”就是这条
- **WHAT**：固定 workload 的 TPOT/TTFT 不得相对**上一次绿基线**退化超过阈值；且 PeerKV 对 single-fit 请求不得慢于原版 vLLM（= §1.3(d) 的常驻化）。
- **HOW**：新建 `tests/perf/test_perf_regression.py`。基线数值落 `tests/perf/baseline.json`（git 跟踪）。每次跑：固定 seed/batch/len，median(≥5)，与 baseline 比 `(1+EPS)`，`EPS=0.05`（perf job 比 §1 的 A/B 略松，因为它跨更多 workload）。退化即红。基线更新需 PR 显式改 `baseline.json` + 说明原因（防偷偷放水）。
- **验收 gate**：无退化；single-fit ≤ vLLM。
- **依赖**：稳定的双 A100 窗口；§2 的 `peerkv_tpot_seconds` 直接喂数据。
- **风险**：co-tenant 抖动 → 跑前查 `peerkv_peer_compute_busy`，busy 则重试/skip，连续 3 次 busy 才报 infra-flaky（不算 perf 红）。

### 3.5 故障注入（fault injection）
- **WHAT**：对 P1 中列的故障面逐个注入，验证系统**安全降级到 SINGLE/HOST**而非崩溃/出错值。
- **HOW**：新建 `tests/test_fault_injection.py`，每种故障一个 case：
  | 故障 | 注入方法 | 期望行为 |
  |---|---|---|
  | peer OOM | mock `peer_hbm_free_bytes=0` | 不选 COPYBACK/CFK，回 SINGLE/HOST/INFEASIBLE，不崩 |
  | peer-reset | mock peer device 不可达异常 | 捕获 → fallback SINGLE + `WARN` + counter |
  | P2P-disable | 设 `transfer_dir`/p2p 不可用 | R1-route 触发，路由 host |
  | NVLink-degraded | `LinkState(nvlink_eff<pcie)` | R1-route 触发（同 §1.3c）|
  | 非法 corner | 直接喂 `enforce_do_no_harm(CFK, busy)` | 抛 `DoNoHarmViolation` |
- **验收 gate**：每种故障都“安全降级 + 打点 + 不返回错误数值”。

### 3.6 放置代价律常驻测试（复用 Track A 的 HBM-footprint 结论）
- **WHAT**：把 Track A 的代价律钉成产品测试——**(a)** 在忙 holder 上**禁止满速本地 HBM repack**（应改走 peer 或分块）；**(b)** 大 handoff 必须分块限尾；**(c)** 传输方向 push/pull **不影响**正确性与性能（不得有"必须 push"的硬门）。
- **HOW**：`tests/test_direction.py` 已存在 → 改写：断言 selector 在 `peer.compute_busy` 且需要搬运时**不选**满速 local-repack 路径（选 peer-handoff 或 chunked）；断言 `transfer_dir ∈ {push,pull}` 都被接受（**删掉**旧的"pull 必抛异常"用例——那基于已证伪的方向规则）。可选 GPU 验证：复跑 `experiments/g3_placement.py`，断言 peer victim ≪ local victim、且 push≈pull（与 Track A g1–g3 对齐）。
- **验收 gate**：忙卡上无满速 local-repack 进热路径；大 handoff 走分块；方向 push/pull 均不报错且性能等价。

### 3.7 Soak（长稳）
- **WHAT**：连续高负载跑 N 小时无内存泄漏 / 无 fd 泄漏 / 无 metric 异常增长 / do-no-harm 恒 0。
- **HOW**：`scripts/soak.sh`（新建）跑 ≥2h（夜间）混合 workload（single-fit + off-card 混合），周期采集 RSS、GPU mem、`peerkv_*` 指标。断言：RSS/GPU-mem 斜率 ≈0（线性回归斜率 < 阈值），`do_no_harm_violations_total==0`，无未捕获异常。
- **验收 gate**：2h 内无泄漏、无 violation、无崩溃。**M2/M3 发布前必跑一次。**

### 3.8 确定性 exactness + A/B-vs-vLLM
- 确定性：§3.1 已覆盖（固定 seed + allclose）。额外加“**重复两遍同输入 → 输出 bit-相同**”（关闭非确定性 kernel/atomics 时）作为 `test_determinism`。
- A/B-vs-vLLM：§1.3(d) 的 `test_no_harm_ab_vs_vllm` 即此项，产出 `ab_vs_vllm.json` → 论文表 1。

### 3.9 CI 编排（怎么把上面跑起来）
- **CPU job（每次 push 必跑，快）**：现有 22 research-tests + `test_no_harm`(a/b/c) + `test_fault_injection`(mock 部分) + `test_direction`(fuzz)。
- **GPU job（nightly + release-gate，`@pytest.mark.gpu`）**：`test_multigpu_correctness`、`test_no_harm_ab_vs_vllm`、`test_perf_regression`、`test_determinism`、soak（仅 release）。GPU job 先跑 §5.3 拓扑探针，非 NV12 直接 skip（标 infra，不算失败）。
- **required checks（合 PR 必须绿）**：CPU job 全部 + `test_no_harm_*`(a/b/c)。GPU job 红 → 阻塞 release tag，不阻塞普通 PR（除非 PR 改了热路径，标 `needs-gpu` label 时强制 GPU job）。
- **交付物**：`.github/workflows/ci.yml`（或仓库现有 CI 体系，[待确认] 现仓库是否已有 CI 配置）、`pytest` markers（在 `pyproject.toml` 注册 `gpu`/`perf`/`soak`）。

### 3.10 依赖 / 工具 / 负责人 / 风险（§3 汇总）
- **依赖**：M2（formal connector）解锁 3.2/3.4 真实跑；M3（校准闭环）让 3.4 的 predicted-vs-measured 有意义。
- **工具**：pytest(+markers)、CI runner（带双 A100 的 self-hosted runner，[待确认] 是否有专用 runner）、`/public/model_zoo`、prometheus client。
- **负责人**：师弟主写测试；GPU runner 维护 + 错峰调度需与 Track A 协调。
- **风险**：① 没有带 GPU 的 CI runner → GPU job 只能手动在本地双 A100 跑，**那也必须有脚本 + 落 artifact**，不能口头“我跑过了”。② 共享机器 co-tenant → 所有 perf/AB 测试统一走“busy 则 skip/retry”策略，避免 flaky 把 CI 弄成狼来了。

---

## §4 打包与运维（packaging / ops）

### 4.1 wheel / Docker
- **WHAT**：可 `pip install peerkv` 的 wheel（含编译好的 `csrc/*.cu` 扩展）+ 一个可直接 serve 的 Docker 镜像。
- **HOW**：
  - wheel：复用现有 `setup.py`/`pyproject.toml`，把 CUDA 扩展编译纳入 build（`torch.utils.cpp_extension`）。产出 `peerkv-<ver>+cu<XX>-cp3xx-linux_x86_64.whl`，文件名带 CUDA/torch 版本（与 §3.3 矩阵对应）。
  - Docker：`docker/Dockerfile`，base = 与生产匹配的 `nvidia/cuda` + vLLM 0.8.5。`ENTRYPOINT` = vLLM serve + PeerKV connector 注册（formal，非 monkeypatch）。暴露 `:8000`(OpenAI API) + `:9400`(metrics)。镜像启动先跑 §5.3 拓扑探针。
- **验收 gate**：`pip install` 后 `python -c "import umallm; from umallm.elastic_policy import enforce_do_no_harm"` 通过；`docker run` 后 `curl :8000/v1/models` 和 `curl :9400/metrics` 都返回。

### 4.2 硬件支持表（hardware support table）
- **WHAT**：一张明确“支持 / 实验性 / 不支持”的表，写进 README + `docs/hardware_support.md`，**与论文 honest boundaries 完全一致**（不能产品文档吹、论文里又说 N=2）。
- **HOW（已核实事实填表）**：

  | 硬件 | 状态 | 依据 |
  |---|---|---|
  | 2× A100-SXM4-80GB 直连 NVLink（NV12, 273 GB/s 单向, 11.3× PCIe） | **支持（主线）** | 本地盒已验证（MIG-off 后） |
  | 单卡 A100/H100 | 支持（degenerate，恒走 SINGLE） | do-no-harm 保底 |
  | H100 NVLink 多卡 | **已实测(Track A g1–g4)** | dual-H100-SXM NVLink4 NV18 跨代验证完成(`results_h100/`)；产品 runtime 集成仍待 Track C |
  | NVSwitch / N>2 / 多机 | **不支持（未验证）** | 本盒无 NVSwitch，N=2 单盒（honest boundary） |
  | GH200 / UMA / CUDA-managed-memory | **Track D research，phase 4** | D2 明确 park |
  | MLX / Apple UMA | **Track D，park** | D2 |

- **验收 gate**：表中每一行的“支持”都有对应通过的 GPU 测试或被明确标“实验性/不支持”；论文 §limitations 引用同一张表。

### 4.3 启动拓扑探针（startup topology probe = `activate_nvlink.sh` 运行时自检）
> 这是 §1.3(d)/§3.9 GPU job 的前置门，也是 D3 的 R1-route 的“事实来源”。

- **WHAT**：把 `scripts/activate_nvlink.sh` 的诊断逻辑产品化成一个**只读自检 probe**（启动时跑），回答：MIG 是否关、NVLink 是否 NV12、几条 lane up、peer BW、P2P 是否可用；并据此设置运行时模式。
- **HOW**：
  - 新建 `umallm/observability/topology_probe.py`，封装：
    1. `nvidia-smi --query-gpu=mig.mode.current` → 若任一卡 MIG=enabled → **拒绝以 peer 模式启动**，打 ERROR 指向 `scripts/activate_nvlink.sh`（“MIG disables NVLink on A100”，已核实根因），degrade 到单卡 SINGLE-only 模式。
    2. `nvidia-smi topo -m` → 解析是否含 `NV12`/`NV#`；NV12 → 设 `LINK_EFF_GBPS{fabric="nvlink"}=273`、`NVLINK_LANES_UP=12`；`NODE`/无 NV → degrade host-only（R1-route 永远路由 host）。
    3. `nvidia-smi nvlink -s` → 统计 trained lanes 数。
    4. P2P 探测（小 buffer P2P copy）→ 实测一次 `nvlink_eff_gbps` 写进指标（不要只信 vendor 273，用实测）。
    5. 实测 `pcie_eff_gbps`（小 buffer H2D）。
  - **probe 是只读自检**：它**不**像 `activate_nvlink.sh` 那样 `rmmod`/`-mig 0`（那是 root 运维动作）。probe 发现 MIG-on/链路 down 时，**指引运维去跑** `sudo bash scripts/activate_nvlink.sh`，而不是自己改系统。脚本本身保留为“运维修复工具”。
  - 探测结果 → 直接构造 §1 的 `LinkState`，喂给 `enforce_do_no_harm`（probe 是 R1-route 的数据源，闭环）。
- **验收 gate**：① 在 NV12 盒上 probe 报 NV12/12-lanes/~273GB/s 且允许 peer 模式；② 人为把一卡设 MIG-on（或 mock）→ probe 拒绝 peer 模式、degrade SINGLE、ERROR 指向脚本；③ probe 写出的 `LINK_EFF_GBPS` 指标可被 `curl :9400/metrics` 看到。
- **风险**：probe 跑 `nvidia-smi` 子进程在容器里需挂载/权限 → Docker 用 `--gpus all` + NVIDIA runtime，文档写清。

### 4.4 回滚开关（rollback switch）
- **WHAT**：一个环境变量/配置项，**一键把 PeerKV 完全旁路**，退回原版 vLLM 行为（所有请求 SINGLE，connector 不介入）。
- **HOW**：`PEERKV_ENABLED=0`（默认 production 视情况），或 config `peerkv.enabled=false`。关时：connector 注册成 no-op、selector 恒返回 `SINGLE`、不发任何 P2P。**这条让运营在线上一旦发现回归（§3.4 失败的运行版）能秒退**，也是 do-no-harm 的运营兜底。
- **验收 gate**：`PEERKV_ENABLED=0` 时 §1.3(d) 的 A/B 中 treatment == baseline（行为字节级一致）；切换无需重编译、重启进程即可（若做成进程级则文档写明需重启）。
- **依赖**：formal connector（M2）。

### 4.5 多租户 KV 内存擦除（multi-tenant KV memory wipe）
- **WHAT**：跨租户复用 GPU/peer-HBM 的 KV block 之前，**必须清零**，防止租户 A 的 KV 被租户 B 读到（尤其 peer-HBM 借用场景，COPYBACK 把别人 HBM 借走再还）。
- **HOW**：
  - KV block 在归还/重分配前 `memset 0`（`torch.Tensor.zero_()` 或 cudaMemset）；peer-借用的 block 归还 lender 前同样清零。
  - 提供 `peerkv.multi_tenant.wipe_on_free=true`（默认 true，安全优先；可关用于 benchmark）。
  - 测试 `tests/test_kv_wipe.py`：分配 block→写入 sentinel→free→重分配→断言读到全 0（不是 sentinel）。peer 路径同测。
- **验收 gate**：重分配后绝不出现上一租户的 sentinel；peer-借用归还后 lender 侧 block 全 0。
- **风险**：每次 free 都清零有性能开销 → 默认开但允许单租户/benchmark 场景关闭；§3.4 perf 测试用关闭态测上限、用开启态测生产数。

### 4.6 交付物 / 依赖 / 负责人（§4 汇总）
- **交付物**：wheel + `docker/Dockerfile` + `umallm/observability/topology_probe.py` + `docs/hardware_support.md` + 回滚开关 + `tests/test_kv_wipe.py`。
- **依赖**：M2 formal connector（4.1/4.4）；probe 依赖 NV12 盒（4.3）。
- **负责人**：师弟（probe + wipe + 测试），打包/Docker 可与运维协作。
- **风险**：硬件支持表与论文 honest boundaries 漂移 → 把这张表设为**单一真理源**，论文 §limitations 和产品 README 都引用它，改动走同一 PR。

---

## §5 把不变量串成一条闭环（给师弟的执行心智模型）

```
  启动: topology_probe (§4.3) ──► LinkState (nvlink/pcie eff, lanes, MIG)
                                      │
   请求到达 ──► select_point (§1, elastic_policy.py) ──► enforce_do_no_harm (§1)
        │                                                   │ 违反? 
        │                                                   ▼
        │                                        production: fallback SINGLE + counter (§2.2)
        │                                        CI(PEERKV_STRICT=1): raise (§1.3)
        ▼
   热路径搬运: 恒 PUSH (§3.6) ──► 打点 XFER_*/TPOT/TTFT (§2.2) ──► 结构化日志 (§2.3)
        │
        ▼
   CI 门禁: test_no_harm_* + perf_regression + fault_injection + soak (§3)
        │
        ▼
   发布: wheel/Docker + 硬件支持表 + 回滚开关 + KV wipe (§4)
        │
        ▼
   论文: R1 表(AB) + cross-over MAP(日志) + honest boundaries(支持表) ← 同一份真实系统的数据
```

**一句话纪律重申（D3）**：上面每一个箭头上的“规则”，都必须有一个 `assert`/`raise` + 一个 `test_*`。没有测试的不变量等于不存在。

## §6 开放项 / 待确认（写代码前先 close）
- [待确认] CI runner 是否带双 A100（决定 GPU job 是自动还是手动 + artifact）。
- [待确认] §1.3(d) 的 `EPS=0.03` 与 §3.4 的 `EPS=0.05` 用首轮重复方差校准后回填。
- [待确认] §3.1 多 GPU 正确性的 `atol`（CFK partial 合并的数值容差）以首轮实测定。
- [待确认] vLLM 版本矩阵的实际可用上界（0.9.x 是否存在/兼容）。
- H100 NVLink 多卡 Track A 测量已完成（`results_h100/`，跨代 g1–g4）；Track C 产品 runtime 在 H100 上的集成与 CI 覆盖仍 [待确认]。
- [待确认] 仓库是否已有 CI 配置（`.github/workflows/`）还是需新建。
