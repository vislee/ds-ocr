# ds-ocr 推理框架：准确率回归体系 + 瓶颈攻坚

slug: ds-ocr-acc-perf
date: 2026-10-30
type: 执行型
status: approved

你是执行者，本文是你唯一任务来源，中途没人可问：拿不准的写 `BLOCKED.md`，跳过继续做别的，随交付提交。断线/换会话先读 `PROGRESS.md` 接着做，别重做；每完成一项立刻更新。

这活为什么干：上轮已修复 V4 DASH 七处 bug，四模型对 test截屏.png 的输出与原图人工核对一致（2026-10-30）。但精度验证全靠手工、编码端（SAM）占总耗时 44-54%、V4 峰值内存 12.9GB。目标：改成「可回归验证」——改任何一行代码，一条命令就知道精度坏没坏、速度内存达标没。
两个要求打架时：算得对 > 做得全 > 做得快。回归红了，速度优化等于白做。
「只允许/不许」是死规矩，违反算失败；「建议」可绕，理由记一行进 PROGRESS.md。

## 我替领导拍的板（发出前可改判）
1. 目标的目标 =「可回归验证的生产级引擎」，先建测量再攻瓶颈，不是刷 benchmark。猜的｜错则多花在测试基建上。
2. 速度目标定「编码端 ≤ 现状 60%」。猜的｜保守 = 可能少赚一半提速。
3. 内存目标 ≤8GB，走 KV F32→BF16，不上 INT8。猜的｜要 ≤4GB 需 INT8+专项精度验证，另立一轮。
4. 黄金图集 = 现有 4 张测试图。猜的｜图少测不出长尾回归，加图需领导给素材。
5. CER 阈值由任务 1 实测基线后冻结（基线 +1pt），冻结后不许放宽。

## 界限
- 只允许改：`ds_*.c/h`、`main.c`、`Makefile`、新建 `tools/**`、`docs/golden/**`；其余只读。
- `models/**` 不许动（权重与 config 都不许）；`tools/regress.sh` 通过验收后冻结。
- 不新增依赖（CER 用 python3 标准库 difflib 算）。
- 顺手活（重构抽象层、升级 stb_image、跑 OmniDocBench）→ 记 BLOCKED.md，不做。

## 现状与任务 0
实测（来源：`./ds_ocr … --profile` 与 `/usr/bin/time -l`，2026-10-30，空闲 CPU，图 test截屏.png 1794×1578）：

| 模型 | 总耗时 | 编码 | 解码 | tok/s | 峰值内存 |
|---|---|---|---|---|---|
| DeepSeek-OCR | 10.8s | 4.8s | 4.6s | 51.1 | 未测 |
| DeepSeek-OCR-2 | 14.1s | 7.8s | 4.6s | 48.5 | 未测 |
| Unlimited-OCR | 31.6s | 11.3s | 15.5s | 39.2 | 未测 |
| Unlimited-OCR-2 | 19.8s | 6.7s | 10.9s | 46.5 | 12.9GB |

已知薄弱：V3 对 400×100 极端小图输出 det 块；多页文档路径未测过；无自动化精度验证。
任务 0：跑一遍任务 1、2、3 的验收命令核对上表数字，偏差 >20% 的条目连证据写进 BLOCKED.md 最上面，只做不受影响的部分；核对无误后把「理解的目标/执行顺序/最大风险」≤10 行写进 PROGRESS.md 再动工。

## 任务 1：回归基座（必须最先做——后面全部速度优化靠它兜底）
一句话：`bash tools/regress.sh` 一条命令判定 4 模型 × 4 图精度 PASS/FAIL。
做法：a) 把 4 张测试图固化进 `docs/golden/images/`（含复制 /tmp/crop_real.png，它在临时目录会丢）；b) 用官方 Python harness（`/tmp/run_official.py`、`/tmp/run_official_v3.py`，已存在，脚本内含本机 transformers 5.12 的兼容 shim）逐图生成黄金输出存 `docs/golden/<model>_<img>.txt`；官方 CPU 单图超 30 分钟或输出乱码（test_simple.png 实测 ĠLessĠLess…）时，改用人工核对过的引擎输出做标签，文件头注明来源；c) regress.sh 对每组合算 CER（1−difflib SequenceMatcher 比率），打印「模型×图 CER% PASS/FAIL」汇总表；d) CER 阈值 = 首轮实测值 +1pt，冻结为 regress.sh 顶部常量并抄进 PROGRESS.md。
死规矩：阈值冻结后不许放宽；黄金标签生成后只读。
验收：`bash tools/regress.sh` 输出 16/16 PASS 且每格 CER ≤ 冻结阈值。
反向验证：把 ds_dash.c:479 `rt->valid[i] = (xi[i] >= 0 && eff[xi[i]]) ? 1 : 0;` 临时改回 `cand[xi[i]]`（上轮已确认的 bug），regress.sh 必须 ≥1 格 FAIL；还原后 16/16。红、绿两段输出都贴对话。

## 任务 2：编码端提速（目标：编码耗时 ≤ 现状 60%）
一句话：编码端压到 V1 ≤2.9s、V2 ≤4.7s、V3 ≤6.8s、V4 ≤4.0s。
从哪下手：先 `--profile` 分解 SAM 12 层/窗口注意力耗时大头（ds_visual_tokenizer.c 已有窗口并行 gate），候选手段按 profile 排序：提高窗口并行度、crop 级 SAM/CLIP 流水线重叠、Metal GPU 路径（ds_metal.m 已有骨架）、消除逐层重复 malloc。不许做无 profile 依据的优化。
死规矩：动 Metal 必须保留 CPU 兜底路径（环境变量可切回）。
验收：`for m in DeepSeek-OCR DeepSeek-OCR-2 Unlimited-OCR Unlimited-OCR-2; do ./ds_ocr -d ./models/$m -i ./docs/golden/images/test截屏.png 2>/tmp/t_$m.log; done; grep Encoding /tmp/t_*.log` 四个数字全部 ≤ 上表阈值，且 `bash tools/regress.sh` 仍 16/16 PASS。两段输出都贴。

## 任务 3：解码与内存（目标：V4 峰值内存 ≤8GB 且 V3 decode ≥45 tok/s）
一句话：KV 缓存 F32→BF16（ds_moe_decoder.c 的 kv_cache_k/v、ds_dash 的读路径与 idx_k_hist 同步改），V3 R-SWA 参考集扫描改 BLAS（cblas_sgemv 走 strided lda）或预打包连续 K/V，消除每步 3328 行×12 层的标量扫读。
死规矩：KV 精度改动必须过任务 1 回归；CER 允许在冻结阈值内浮动。
验收：`/usr/bin/time -l ./ds_ocr -d ./models/Unlimited-OCR-2 -i ./docs/golden/images/test截屏.png 2>&1 | grep "maximum resident"` ≤ 8589934592 字节；V3 同命令 `grep tok/s` ≥ 45；`bash tools/regress.sh` 16/16。三条都贴。
若 BF16 KV 导致 CER 超阈值：回退 F32，BLOCKED.md 如实记「内存目标未达」——不许放宽阈值硬凑。

## 任务 4（可跳过）：V3 极端小图调查
一句话：给结论——V3 对 400×100 图输出 det 块是模型固有行为还是 C 缺陷（用 /tmp/run_official_v3.py 跑官方 BF16 对照）。
验收：≥1 条结论，附复现命令与原始输出；若动代码，regress.sh 保持 16/16。连败 1 次即记 BLOCKED 跳过。

## 规矩
- 防作弊：skip/todo/mock、放宽 CER 阈值、改黄金标签、改 regress.sh 让失败变绿、`|| true`——全算失败；每条验收贴实际命令输出，只说「做完了」不算。
- 同一验收连败 3 次换下一项；任何指标比开工基线差必须回滚并如实报告——「没做成但说清」合格，「做了但更糟」不合格。
- 诊断环境变量 DS_DUMP_ROUTE / DS_V4_NOBIAS / DS_V4_FULLATTN 是交付物一部分，不许删。

## 完成条件
1. 硬指标一：`bash tools/regress.sh` 16/16 PASS（精度守约束）。
2. 硬指标二：任务 2、3 的全部数字验收满足（性能/内存外部判据）。
3. 每条验收在对话里贴实际命令输出，含任务 1 反向验证的红→绿证据；只说做完了不算。
4. PROGRESS.md 逐项有据；BLOCKED.md 随交付提交，空也写「无」。
5. 止损：任务 2、3 各满 3 轮即停，如实汇报卡在哪、还差什么；任何指标低于开工基线——立即回滚收尾。
