# 随机切割库存问题（基于论文的复现实现）

目录结构（示例）:
- data/patterns.json        # 示例切割模式与问题参数
- src/env.py                # 环境实现（CSP）
- src/features.py           # 基函数（polynomial / fourier）
- src/cross_entropy.py      # 交叉熵启发式求贪婪动作
- src/api.py                # 近似策略迭代主算法（训练 / 评估）
- train.py                  # 训练入口脚本
- evaluate.py               # 评估与绘图脚本

快速开始:
1. 创建并激活虚拟环境 :
   python -m venv .venv
   .venv\Scripts\activate     # Windows

2. 安装依赖:
   pip install -r requirements.txt

3. 运行训练（示例，使用小参数做快跑）:
   python train.py --config data/patterns.json --L1 5 --L2 2000

4. 训练完成后评估并绘图:
   python evaluate.py --config data/patterns.json --theta_file theta_iters.npy

主要参数说明:
- L1: 外层策略迭代次数（论文用 30）
- L2: 内层样本数量（论文用 50_000）
- gamma: 折扣因子（论文用 0.8）
- basis: 'fourier' 或 'polynomial'

注意:
- 真实复现要尽量使用与论文相同的切割模式与超参数，上述 data/patterns.json 是示例（包含 15 个模式与 trim losses），如你有论文附带数据文件可替换之。
- 大样本训练耗时较长，建议在 GPU/多核或高性能 CPU 上运行，或先调小 L2 做调试。