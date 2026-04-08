# 参数说明（窗口替换 + 二阶段LB）

## 新增关键超参数（默认值）
- --rl-topk 3：每步 RL 取 top-k 候选，commit 只执行 top-1。
- --util-min-aux 0.8：top-2/top-3 入小列池的最小利用率。
- --mini-pool-cap 10：小列池容量（完整加入一次后才触发求解，允许从 8 变到 11 后触发）。
- --mini-pool-max-steps 5：窗口最大 step 数，达到强制触发。
- --mini-ilp-timelimit 0.5：小列池 ILP 时间上限（秒）。
- --tail-drop-last 1：是否启用“拆最后一根”。
- --tail-util-threshold 0.6：最后一根利用率阈值。
- --log-trace：打印窗口 trace 摘要。
- --trace-json <path>：导出窗口 trace JSON。

## 账本语义
- 窗口内先按 top-1 临时更新 d_tmp，仅供下一步 CG/RL。
- 窗口结束后用 mini-pool 替换方案重算覆盖并覆盖全局 d_remain。
- overprod 仅统计，不计库存。

## 批量汇总
`run_dataset_and_summarize1.py` 的 CSV 除 stage1/stage2 外新增：
- lb_before_*：Phase2（LB之前）
- lb_after_*：LB后 final
