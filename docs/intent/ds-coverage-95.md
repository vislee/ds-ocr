# 单元测试覆盖率 ≥95%（B9 立项）

slug: ds-coverage-95
date: 2026-10-31
type: 执行型
status: approved（B9 立项，领导批准）

现状：基础设施已建成（`make test_unit` / `coverage-unit` / `coverage`，23 用例 ~368 断言全绿，
含四模型 e2e 对照冻结标签）。无 e2e 行覆盖：safetensors 61.4%、tokenizer 58.3%、neon 44.8%、
kernels 13.0%，模型大文件 0%（依赖 e2e）；全引擎合计（含 e2e）预估 55–70%。

## 已知阻碍
- 插桩下 V3/V4 解码慢至 ~30s/token（CLI ~1000×，机理未明——疑似 gcov 计数写与 madvise 重入交互），
  全量 e2e 覆盖一次 ≈ 数小时。立项后第一件事：定位并解决插桩减速（如 gcov 换 llvm-instr 精简计数、
  或 e2e 分模型并行采集 profraw）。

## 任务（按序）
1. 插桩减速定位与解决（目标：全量 e2e 覆盖一次 ≤30 分钟）
2. 模型大文件补测：visual_tokenizer / deep_encoder / moe_decoder 的分支遍历
   （SAM 窗口边界、CLIP flow-query、预填充分支、INT4/Metal 旁路）
3. 错误路径套件：malloc 失败注入、损坏文件、极端参数
4. tokenizer 全词表边界与 4 字节 UTF-8
5. 每轮跑 `make coverage` 核对，迭代 2–5 直到 ≥95%

## 验收
`make coverage` 报告核心 10 文件行覆盖 ≥95%（AVX 文件在 ARM 上无行，排除在分母外）。

## 止损
预估 2–3 个完整工作日；单任务连败 3 次记 BLOCKED。
