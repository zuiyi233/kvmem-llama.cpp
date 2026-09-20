# GDN MTP 解码性能分析与优化计划（P0-P2）

- 日期：2026-09-21
- 范围：kvmem-llama.cpp，`llama-kvmem-server`，GDN replay MTP 解码路径
- 模型：Qwen3.8-35B-A3B Q4_K_M（21.7GB，`L:\Qwen3.8\Qwen3.8-35B-A3B-Q4_K_M.gguf`）
- 硬件：RTX 3090 24G / Tesla T10 16G，Threadripper 3960X 24 核，64GB RAM
- 测量：`bench_stream.py PORT medium N`（4k 上下文，think 模式，稳态 decode t/s，剔除 prefill）

## 1. 目标

把 **kvmem 开启 + GDN MTP** 的解码速度从当前 88.6 t/s（3090）优化到接近/达到上游 Prism 纯 GPU 路径的 ~125 t/s。

## 2. 现状数据全景（3090，同一 kvmem 二进制，2026-09-21 实测）

| 配置 | median t/s | 相对 125 |
|---|---|---|
| 上游 Prism 纯 GPU（历史数据，条件未复现） | ~125 | 基准 |
| `--no-kvmem`（关闭 kvmem，无 MTP） | 81.9 | -34% |
| kvmem 开，无 MTP | 51.6~61.8 | -51% |
| **kvmem + GDN MTP n_max=3** | **88.6** | **-29%** |
| kvmem + MTP n_max=4 | 95.0 | -24% |
| kvmem + MTP n_max=5 | 97.9 | -22% |
| kvmem + MTP n_max=3 + `--presence-penalty 0` | 96.4 | -23% |

> 注：同日同机波动约 ±10%（GPU 频率/瞬时负载）；79.3 与 88.6 均为同配置不同时刻测得。

## 3. 每步耗时拆解（KVMEM_VERIFY_SPLIT，n_max=3，39 步 verify，每步产出 4 token）

```
per_step: begin=0.00  decode_tgt=8.54  sync=10.66  process=2.01  sample=16.50 (ms)   合计 ~37.7ms/步
```

| 阶段 | 每步耗时 | 占比 | 性质 |
|---|---|---|---|
| gdn_replay_begin | 0.00ms | 0% | GDN 事务开始，可忽略 |
| decode_tgt（llama_decode 4-token batch） | 8.5ms | 23% | GPU MoE 前向（入队） |
| sync（llama_synchronize 等 GPU 完成） | 10.7ms | 28% | GPU 真实执行主要落点 |
| process（verify 结果处理） | 2.0ms | 5% | CPU |
| **sample（采样 + 接受判定，4 位置）** | **16.5ms** | **44%** | **CPU，全链路最长单项（关键路径）** |

### 3.1 n_max 扫描（2026-09-21 实测）

| n_max | t/s | 每步 token | decode_tgt | sync | process | sample |
|---|---|---|---|---|---|---|
| 3 | 88.6 | 4 | 8.54 | 10.66 | 2.01 | 16.50 |
| 4 | 95.0 | 5 | 14.83 | 10.92 | 1.82 | 20.76 |
| 5 | 97.9 | 6 | 12.04 | 13.16 | 1.99 | 25.33 |

sample 与位置数近似线性（每位 ~4.2ms）；n_max 升高收益递减（固定开销摊薄 vs sample/decode 线性涨）。

### 3.2 决定性实验：penalty 对 sample 的影响

`--presence-penalty 0`（其余不变，n_max=3）：sample 16.50 → 10.96ms/步（-34%），总速 88.6 → 96.4 t/s（+9%）。

采样链实况（日志 `KVMEM_TRACE sampling`）：`temperature=0 top_p=1 top_k=0 min_p=0 presence_penalty=1.5`——
链上实际只有 penalty(1.5) + temp(0)。penalty 采样器每位置全量遍历 15 万 vocab，加上 argmax 全遍历与 logits D2H，构成每位 ~4.2ms。

## 4. 瓶颈定位结论

1. **关键路径在 CPU 侧 sample，不是 GPU、更不是 KV 内存化**。GPU 每步仅忙 ~10.7ms，CPU sample 16.5ms 反超——CPU 采样成为每步串行链路的顶梁。这也是与 125 的第一大差距来源。
2. **MTP 的 4-token batch 是 MoE 的放大器**：4-token verify decode ~10.7ms vs 单 token ~12.2ms（no-kvmem 路径）——batch 权重驻留复用让 MoE 的 decode 效率随 batch 提升；MTP 命中率近乎 100%（日志 n_accept=3/3 持续满档、restore=0 无回滚）。
3. **kvmem 编排税（-37%）在 MTP 档已被 4 倍摊薄**，不再是主要矛盾。

## 5. 与 125 的差距归因

| 差距项 | 大小 | 证据 |
|---|---|---|
| sample 可优化部分（penalty 遍历） | 5.5ms/步 | 实验实测 -34% |
| sample 剩余（argmax + D2H + 链框架） | ~11ms/步 | penalty=0 后仍 10.96ms |
| GPU decode 速度（kernel 版本差） | decode_tgt+sync ~19ms/步 | kvmem 子模块基座 b10759 vs Prism b9591+（未同条件复测，含测量条件差，未完全归因） |
| 每步间编排间隙 | ~2ms/步 | 88.6 vs 理论 106 t/s |

## 6. 优化计划（P0 → P1 → P2）

### P0 采样链精简（最高性价比，预期 +10%~30%）

- 无副作用部分：**penalty 采样器在 penalty 全为零时从链上跳过**（等价语义，上游 llama.cpp 本应如此）。实测 penalty 1.5→0 已省 34% sample。
- 进阶：temp≤0（greedy）且用户接受 greedy 语义时，进一步构建 argmax-only 链（跳过 top_k/top_p/min_p/penalty 遍历）。
- 预期：sample 4 位置 <2ms/步 → 每步 ~24ms → 理论 120+ t/s。

### P1 n_max 调参（零成本，+8~10%）

- n_max 3→4/5：实测 88.6→95→97.9。需在 128k/256k 上下文复测 accept 率不掉。
- 桌面 app「KVMem 一键参数」预设同步升级。

### P2 logits D2H 优化（预期 +3~8%）

- 4 位置 logits 合并单次 D2H（或 GPU 端 logits 处理后再拷），减少 per-position 拷贝与同步。

### 远期（P3+，本计划不实施）

- GPU kernel 版本升级（子模块 b10759 → b9591+）：decode 或降 20-30%，彻底追平 125，风险最高。
- 架构级流水线：draft 生成与 sample/verify 重叠（上游 speculative 同受限）。

## 7. 预期达成路径

P1（n_max=4）→ ~95 t/s → P0（采样精简）→ ~115-125 t/s → P2（D2H）→ 120-135 t/s。

**T10 同样受益**：T10 的 25.9 t/s 中 sample 占比同构，P0+P1 预计再上 15-25%。

## 8. 实施记录

### P0 落地：fast-greedy 采样路径（2026-09-21，提交 1699f74）

**实现**（`tools/kvmem-spec.cpp`）：
- `kvmem_spec_fast_greedy_ok(sparams)`：当 `temp<=0` 且链上所有非温度采样器均处禁用态（top_k<=0、top_p>=1、min_p<=0、penalty 全 0、dry 0、top_n_sigma<0、typ_p>=1、xtc 0、无 grammar、无 reasoning budget）时启用。
- `kvmem_spec_sample_fast_greedy`：一次 `llama_synchronize` + 一次 `llama_get_logits`（全量 outputs 块，一次 D2H），每位置直接对 logits 行 argmax，`common_sampler_accept` 保持采样器状态一致。
- 语义等价性：temp<=0 的 llama.cpp 温度采样器本身就是 argmax（llama-sampler.cpp:270-285），链上其余采样器在禁用态均为 no-op（penalties is_disabled / top_k<=0 empty 等），因此输出与完整链逐 token 一致。
- 原 `common_sampler_sample_and_accept_n` 路径保留（temp>0 / grammar / penalty 场景不受影响）。

**实测（3090，Qwen3.8-35B-A3B Q4_K_M，KVMEM_PROFILE）**：

| 配置 | t/s | sample/步 |
|---|---|---|
| 原链 n_max=3（优化前） | 88.6 | 16.50ms |
| `--presence-penalty 0` n_max=3 | 96.4 | 10.96ms |
| **fast-greedy n_max=3** | **144.3** | **1.40ms** |
| **fast-greedy n_max=4（4k）** | **155.0** | 1.70ms |
| **fast-greedy n_max=4（128k）** | **155.4** | 1.72ms |

- sample 每步 -92%（16.5→1.4ms）；总速超 125 目标 24%。
- 128k 长上下文与 4k 零差距（kvmem 内存化 KV 卖点与 fast-greedy 速度同时兑现）。
- **T10 顺带验证**（ncmoe 20 + fast-greedy + n_max=4）：25.9 → **37.3 t/s（+44%）**，fast_greedy 触发确认；T10 新瓶颈转为 CPU 专家计算与 GPU 执行。
- 限制：fast-greedy 仅覆盖 greedy（temp<=0）+ 无 penalty 场景；thinking 模式（temp=1.0）与 penalty>0 场景仍走原链（sample ~16.5ms，速度 ~89-96 t/s）。

### P1 n_max 调参

- n_max=4 与 fast-greedy 组合实测 155 t/s（4k 与 128k 一致），accept 率满档（committed_rows/verify_calls ≈ 4-5）。桌面预设建议 n_max 3→4。

### P2 状态

- 批量 logits D2H 已隐含在 fast-greedy 路径中（一次 get_logits 替代逐位置）。temp>0 场景的 cur 构造优化（15 万 token_data/位置）未实施，留待需要时。
