# 港口优化与决策 (Port Optimization) - 两阶段随机规划

本项目为《决策与优化》课程作业，主要研究在不确定性环境下的港口设施投资与运营优化问题。我们通过建立两阶段随机规划模型（Two-Stage Stochastic Programming），分析了考虑未来环境不确定性（即随机规划）在面对极端情景时，相较于确定性期望值模型产生的经济价值（VSS, Value of Stochastic Solution）以及投资策略的结构性差异。

## 📂 文件与结构说明

- **`run_experiments.py`**  
  核心实验代码。构建并求解了多情景的确定性等价问题（DEP）以及期望值模型及其期望（EV / EEV），包含数据生成和gurobi求解。比较了第一阶段决策在不同环境设想下的结构性变化，并输出核心指标与 VSS（不确定性相关指标）。

- **`benders_template.py`**  
  多割 Benders 分解（Multi-Cut Benders Decomposition / L-shaped method）求解算法的实现。将大规模问题分解为第一阶段主问题（Master Problem）和各个情景独立的第二阶段子问题（Subproblems），通过动态生成最优性割（Optimality cuts）和可行性割（Feasibility cuts）迭代寻找最优解。

- **`visualize.py`**  
  数据可视化分析脚本。负责读取 `.pkl` 结果并基于 Matplotlib 绘制各设备运营时间序列的堆叠面积图、运行对比柱状图等。

- **`experiment_results.json`**  
  保存的优化结果文件及对应提炼的运行统计分析数据。

- **`vis_page1.png`** / **`vis_page2.png`**  
  由 `visualize.py` 生成的数据可视化验证图表。

## ⚙️ 环境依赖

需要安装相应的 Python 包以及有效的求解器：
- Python 3.8+
- `gurobipy` (Gurobi Optimizer)
- `matplotlib`
- `numpy`
- `pandas`

## 🚀 运行方式

1. **执行对比实验与 VSS 分析**：
   ```bash
   python run_experiments.py
   ```
2. **运行 Benders 分解算法测试**：
   ```bash
   python benders_template.py
   ```
3. **生成结果可视化图表**：
   ```bash
   python visualize.py
   ```

## 💡 分析简述

研究结果表明，不确定性环境会促使决策者在**第一阶段做出更多的缓冲性资本投入**。这是典型的“风险对冲”策略，用早期稳定的建设成本换取在恶劣随机场景中（如极端超额排放惩罚或需求激增）的低昂补救成本，从而使得整体期望总成本最优。
