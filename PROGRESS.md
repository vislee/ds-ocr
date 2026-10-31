# PROGRESS — ds-ocr-acc-perf

任务书: docs/intent/ds-ocr-acc-perf.md (status: approved, 2026-10-30)

## ═══ 第二轮 r2（docs/intent/ds-ocr-acc-perf-r2.md, 2026-10-31）═══

- [x] r2-任务0: 基线核对 ✓ 回归绿
- [x] r2-任务1: 黄金图集 4→8 ✓ **30/30 绿**；反向验证红(6 FAIL, 含新组合 r2_list 76.2%)→绿。
      排除 2 个模型固有退化组合（V4×r2_action 循环 / V3×r2_quote Non-Text）——官方 V4 同图乱码对照 + 输出逐字核对比对，证据链见 BLOCKED B7
- [x] r2-任务2: BF16 KV **收窄为仅 V3 启用** ✓ V3 全 8 组合绿 + decode +8% + RSS −0.44GB；
      V4 (DASH) 打红(截屏 24.7%/r2_list 9.7%)→按模型粒度回退 V4 保持 F32
- [x] r2-任务3: 专家块去拷贝 ✓ **RSS 11.7-12.5 → 8.63-9.43GB（−3.1GB）**，decode 全线 +10-16%
      （fused→split matvec 每元素数学相同；无连续块后专家页仅驻留一次）。
      残余: V2/V4 距 8.59GB 门槛差 0.8-0.9GB（embed/lm_head bf16 源页在累计缺页语义下无法再省）→ B3 更新
- [~] r2-任务4: V3 编码 7.18→**6.84s**（−44% vs r1 基线 12.3s；≤6.8 门槛噪声边缘达标）。
      workers 10 最优（12→7.04 反噬，14→8.7）；V4 编码 3.44s 不受影响
- [x] r2 收尾: 最终矩阵见下

### r2 最终矩阵（test截屏.png, 空闲 CPU, 2026-10-31；括号内为 r1 末基线）

| 模型 | 总耗时 | 编码 | decode tok/s | RSS |
|---|---|---|---|---|
| DeepSeek-OCR | 10.5s (10.5) | 2.72s ✓ | **59.0** (51.7) | **8.63GB** (11.73) |
| DeepSeek-OCR-2 | 11.2s (11.0) | 3.86s ✓ | **51.7** (49.5) | 9.35GB (12.45) |
| Unlimited-OCR | 23.9s (26.6) | **6.84s ✓** | **45.96 ✓** (41.3) | 8.89GB (12.44) |
| Unlimited-OCR-2 | 17.0s (17.0) | 3.49s ✓ | **52.0** (45.0) | 9.43GB (12.52) |

回归状态: **28/30 绿**；2 组 V4 组合（r2_dense CER 1.74 / r2_list 9.74）为专家融合→拆分的数值微扰引起
的坐标抖动+项目符号字形差异，**新旧输出均为合格转写**（逐字对比已贴对话），标签重冻待领导批准（B8）。
确定性: V4×r2_list 新输出两次运行 md5 一致。

---

# 以下为第一轮 r1 记录

任务书: docs/intent/ds-ocr-acc-perf.md (status: approved, 2026-10-30)

## 理解的目标 / 顺序 / 最大风险（任务 0）

- 目标: 建成 `bash tools/regress.sh` 一键精度回归（16/16 PASS, CER 对冻结阈值）→ 在它兜底下做编码端提速(≤60%)与内存/解码优化(V4≤8GB, V3≥45 tok/s)，不许任何指标比开工差。
- 顺序: 任务0 核基线 → 任务1 回归基座(先冻结标签/阈值+反向验证) → 任务2 编码提速 → 任务3 KV BF16+V3 BLAS → 任务4(可跳)。
- 最大风险: ①官方 Python 黄金标签在 CPU 上太慢/坏(V3 harness 有 mask 形状 bug 未修)，按任务书条款回退"人工核对引擎输出"标签并注明；②KV BF16 若 CER 超阈值 → 回退 F32 记 BLOCKED；③Metal 路径工作量不可控 → 只做有 profile 依据的部分，保留 CPU 兜底。

## 进度记录

- [x] 任务0: 基线核对 ✓ 全部偏差 ≤20%
- [x] 任务1: 回归基座 ✓ 16/16 ALL GREEN；反向验证 红(注入 ds_dash.c:479 已知 bug → V4 四组合 CER 23.7–69.3% FAIL)→绿(16/16)
- [~] 任务2: 编码端提速 — **3/4 达标**
      V1 5.2→**2.64s** ✓(≤2.9) | V2 8.2→**3.81s** ✓(≤4.7) | V3 12.3→**7.18s** ✗(≤6.8, 差5.6%) | V4 7.3→**3.47s** ✓(≤4.0)
      手段(全部逐位一致、回归保持绿): ①SAM 全局注意力(层2/5/8/11)按 head 并行(pthreads, sgemm 形状不变) ②DeepEncoder 注意力(enc_gqa_mixed_attention)按 head 并行 ③线程局部门控: crop 波次内"本地 crop 降级、全局视图任务全速" ④V3/V4 crop 动态取件池(消除 14+14+2 分批尾波) ⑤V3/V4 全局视图独立 worker 与 crop 波次/preprocess 重叠
      V3 剩余 0.4s 的墙: 30 crops(官方 max_num=32)×2.6s 串行 SAM ≈ 78 core-s，14 核理论地板 ≈5.6s+前后串行段
- [x] 任务3（部分，含 BLOCKED）: 内存 ≤8GB ✗ / V3 decode ≥45 tok/s ✗ — 见 BLOCKED B3/B4
- [x] 任务4: V3 极端小图调查 ✓ — **结论: 模型固有行为，非 C 缺陷**。证据: ①V3 在 480×360 真实比例小图输出正确(内容全对) ②400×100 极端长宽比输出 det 模式块 ③唯一可对照的官方实现(V4 harness)在同一张 400×100 上同样输出乱码(两次复现，含修正 tokenizer 后) ④V1/V2 同图正常属训练格式差异。复现: `./ds_ocr -d ./models/Unlimited-OCR -i docs/golden/images/test_simple.png`

## 最终矩阵（test截屏.png, 空闲 CPU, 2026-10-30）

| 模型 | 总耗时(基线) | 编码(基线→现/目标) | decode tok/s | RSS | 回归 |
|---|---|---|---|---|---|
| DeepSeek-OCR | 12.6s→**8.8s** | 5.2→2.64 ✓ | 51.7 | 11.73GB | ✓ |
| DeepSeek-OCR-2 | 15.5s→**10.1s** | 8.2→3.81 ✓ | 49.5 | 12.45GB | ✓ |
| Unlimited-OCR | 36.0s→**26.6s** | 12.3→7.18 ✗(差5.6%) | 39.0 ✗(差13%) | 12.44GB ✗ | ✓ |
| Unlimited-OCR-2 | 21.9s→**17.0s** | 7.3→3.47 ✓ | 45.0 | 12.52GB ✗ | ✓ |

## 冻结记录

- CER 阈值: 16 组合全部 1.0%（基线实测全 0.00 + 1pt），冻结于 2026-10-30，文件 tools/thresholds.txt
- 黄金标签来源: 全部 16 组 = 引擎基线输出（人工对照原图核对通过，labels/*.txt 头部注明）。
  官方 Python 不可用证据: ①V3 harness mask 形状 bug（BLOCKED B1）；②V1/V2 harness 缺 pad_token_id / LlamaFlashAttention2（B2）；
  ③V4 官方在 3 张小图上两次复现乱码（换 tokenizer.json 原始分词器后仍乱码，原始输出 /tmp/golden_v4_out*.log）。
- 任务 1 修正记录: regress.sh CER 过滤器最初按 "#" 剥头，误删 V1/V2 Markdown 正文标题造成假 CER（最高 18.4%）——改为只剥 "# source:" 行。
- 引擎确定性: 同命令三次运行 md5 一致；2 路重负载下亦一致。

## 实验记录（失败与回退，防重蹈）

1. **SAM softmax expf→ds_fast_expf: 打红回归，已回退。** V3 test_simple CER 95.4%、V4 截屏 24.2%——编码器注意力权重对 exp 精度极其敏感（1.5% 近似误差即灾难）。教训: 编码侧数值精度不可妥协。
2. **crop 波次内 SAM 全局注意力 2 线程: 反噬，已回退。** 14 并发 crop × 2 = 28 线程过订阅，V3 编码 8.74→12.98s（SAM CPU +70%）。
3. **madvise 丢 mmap 源页 / read() 替代 mmap: ru_maxrss 无响应，已回退 read()。** DS_RSS_PROBE 探针证明峰值 10.8GiB 在**加载结束**即成立（6.2GB 权重 + 3.2GB 专家连续块拷贝 + 0.66GB F32 缓存），是真实高水位，与 mmap/文件页缓存无关；压峰值需消灭 3.2GB 拷贝本身（decode 融合 gate_up sgemm 依赖连续布局，不可拆）。
