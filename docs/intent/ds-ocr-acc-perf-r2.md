# ds-ocr 识别准确率保障 + 剩余效率瓶颈（第二轮）

slug: ds-ocr-acc-perf-r2
date: 2026-10-31
type: 执行型
status: approved（用户指示"拆解+落实"）

你是执行者，本文是唯一任务来源；拿不准写 BLOCKED.md 跳过继续。断线先读 PROGRESS.md（上一轮 r1 同目录）。
为什么干: r1 建成了回归门控（tools/regress.sh，16/16 绿），本轮用它作仲裁者，把 r1 因数值风险被搁置的优化变成有界实验，并补齐准确率保障的覆盖面。
让步顺序: 算得对 > 做得全 > 做得快。回归红 = 一票否决，立即回退。

## 我替领导拍的板
1. 准确率交付 = 黄金图集 4→8 张（新增密集小字/标题区/表格线区域裁剪，人工对照冻结）+ 多页路径列为待裁决不实施（CLI 单图架构，属功能开发非准确率）。猜的｜领导要真多页支持则另立一轮。
2. BF16 KV 只对 V3/V4 启用（8192 槽大户）；V1/V2 的 KV 仅 0.5GB 不动。猜的｜若 V3/V4 也打红则整体回退记 BLOCKED。
3. 实验«红»的判定 = tools/regress.sh 任一组合 FAIL，或 V3/V4 decode 较基线慢 >5%。

## 界限
只允许改: ds_*.c/h、main.c、Makefile、tools/**、docs/golden/**。models/** 只读。
tools/regress.sh 的 CER 判定逻辑与 thresholds.txt 冻结值不许放宽；新增图片按同流程冻结新阈值行。

## 现状与任务 0
r1 末实测（空闲 CPU）: 编码 V1 2.64/V2 3.81/V3 7.18/V4 3.47s；decode 51.7/49.5/39.0/45.0 tok/s；RSS 11.73/12.45/12.44/12.52GB；回归 16/16。
任务 0: 跑 regress.sh 与 2 个模型抽查核对以上数字（±20%），异常写 BLOCKED。

## 任务 1（准确率）: 黄金图集 4→8
从 test截屏.png 裁 4 个新区域（密集小字段落、Action 标题区、filter 长句区、左栏列表区），存 docs/golden/images/；四模型跑通→人工对照原图逐块核对→核对通过的存 labels（"# source: engine-baseline r2"）→ thresholds.txt 增 16 行（1.0）。
验收: `bash tools/regress.sh` 输出 32/32 PASS。
反向验证: 注入 r1 同款 ds_dash bug → ≥1 FAIL → 还原 32/32。

## 任务 2（效率·B4+B3）: V3/V4 KV 缓存 F32→BF16（有界实验）
K/V 存储改 uint16（V3/V4 的 8192 槽，省 ~0.5GB 驻留），读取点改为「按位置换一行进 staging，供该行全部 10 head 使用」（求和顺序不变→同 head 内逐元素结果仅受 BF16 舍入影响）。实现范围: alloc/store、RSWA worker（位置主序重构）、causal decode worker、prefill 批量注意力（整层 staging）、ds_dash 两处读取。
死规矩: regress 红 → 全部回退记 BLOCKED（不许改阈值硬凑）；V3/V4 decode 较 39.0/45.0 基线慢 >5% → 回退。
验收: `bash tools/regress.sh` 32/32 PASS 且 V3 decode ≥45 tok/s 且 V1/V2 decode 变化 ≤5%。三条输出都贴。

## 任务 3（效率·B3）: 专家块去拷贝（有界实验）
砍掉 3.2GB 连续块：gate_up 融合 sgemm 改为对 mmap 内 gate/up 两个张量分别 sgemm（每输出元素是独立点积，值不依赖 M 形状→理论逐位一致，回归仲裁）。decode 的 madvise WILLNEED 预取改到 mmap 源页。
死规矩: 回归红或 V3/V4 decode 慢 >5% → 回退记 BLOCKED。
验收: 回归 32/32 PASS 且 V4 RSS ≤9.5GB（分阶段目标，8GiB 视结果记 BLOCKED 或达标）且 decode 速度门槛同任务 2。

## 任务 4（效率·B5）: V3 编码 7.18→≤6.8s 收尾实验
预算内试两招: a) 波次 worker 数 14→10 + 每 crop 窗口注意力 2 线程（总线程 ~30，测过载是否净收益）；b) global worker 的 CLIP 提前（CLIP 与 SAM 流水线化）。任一红/劣即弃。
验收: `grep Encoding` ≤6800ms 且回归 32/32。两招皆劣 → BLOCKED 定稿（地板论证已成立）。

## 规矩
防作弊同 r1（skip/mock/放宽阈值/`|| true` = 失败）；同验收连败 3 次换下一项；任何指标差于 r1 末基线必须回滚如实报。诊断环境变量不许删。

## 完成条件
1. `bash tools/regress.sh` 32/32 PASS（精度硬指标）。
2. 效率: V3 decode ≥45 tok/s 与 RSS 阶段目标二者至少其一达成，且其余指标不劣化（效率硬指标）。
3. 每条验收贴实际命令输出（含任务 1 反向验证红→绿）。
4. PROGRESS.md/BLOCKED.md 随交付更新；止损: 各任务满 3 轮即停如实汇报。
