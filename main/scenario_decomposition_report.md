# 场景分解算法报告

## 算法定位

场景分解算法利用两阶段随机规划中“给定第一阶段决策后，各场景第二阶段可独立求解”的结构。与 Benders 分解不同，这里采用更轻量的启发式场景分解：先分别求解单场景问题，再将不同场景诱导出的第一阶段方案放回全场景模型中评估。

## 核心思想

1. 对每个场景单独构建随机规划退化问题，即只考虑该场景下的需求。
2. 求解每个单场景模型，得到一组候选第一阶段投资决策。
3. 去除重复或高度相似的候选方案，保留多样化策略。
4. 将每个候选第一阶段方案固定到全场景模型中，计算真实期望成本。
5. 选择全场景评估成本最低的候选方案。

## 伪代码

```text
Input: 场景集合 S, 最大候选方案数 M
Output: 第一阶段方案和全场景评估成本

1. candidates ← ∅
2. for each s in S do
3.     构建只包含场景 s 的单场景模型
4.     求解单场景模型，得到第一阶段决策 x_s
5.     if x_s 与已有候选方案不重复 then
6.         candidates ← candidates ∪ {x_s}
7.     end if
8.     if |candidates| ≥ M then break
9. end for
10. best_obj ← +∞
11. for each x in candidates do
12.     固定 x，在全场景模型中评估期望成本 obj(x)
13.     if obj(x) < best_obj then
14.         best_solution ← x, best_obj ← obj(x)
15.     end if
16. end for
17. return best_solution, best_obj
```

## 优势与局限

优势：
- 每个单场景问题规模小，求解速度快。
- 可自然并行：不同场景的单场景模型彼此独立。
- 能够产生多种候选投资方案，适合用作大规模问题的快速筛选器。

局限：
- 单场景最优方案未必对全场景期望最优。
- 如果场景数量很多，需要设计更好的候选筛选策略。

## 本项目实现

代码文件：`scenario_decomposition_algo.py`
运行脚本：`run_scenario_decomposition.py`
结果文件：`scenario_decomposition_results.json`
