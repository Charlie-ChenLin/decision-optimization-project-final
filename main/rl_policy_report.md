# 强化学习/数据驱动策略搜索算法报告

## 算法定位

作业要求中提到可以尝试“强化学习”等数据驱动方法。本项目实现的是一个轻量级强化学习风格的策略搜索算法，用于在第一阶段投资决策空间中快速搜索候选方案。由于课程项目规模有限，算法不引入深度学习框架，而采用 Q-learning 思想中的“状态—动作—奖励”结构进行启发式策略改进。

## 状态、动作与奖励设计

- 状态：根据当前投资强度分为 `high_invest` 和 `low_invest` 两类。
- 动作：随机扰动第一阶段策略，包括改变区域电网改造月份、增加或减少 EYC 购买量、增加或减少 DYC 转 EYC 改造量。
- 奖励：将固定第一阶段策略后在全场景模型中的负目标函数值作为奖励，即成本越低，奖励越高。

## 核心流程

1. 先求解均值场景模型，得到初始策略。
2. 在每一轮 episode 中，对当前策略执行随机扰动，产生新策略。
3. 固定新策略，在全场景模型中评估期望成本。
4. 若新策略成本更低，则接受该策略；否则以一定概率探索。
5. 更新简化 Q 表，用于记录不同投资强度状态下的历史回报。
6. 输出搜索过程中评估成本最低的策略。

## 伪代码

```text
Input: 初始策略 x0, 探索率 epsilon, 训练轮数 E
Output: 最优搜索策略 best_solution

1. 用均值模型求解初始第一阶段策略 x0
2. current ← x0, best ← x0
3. for episode = 1,...,E do
4.     以 epsilon 概率扰动 current，得到 candidate
5.     固定 candidate，在全场景模型中评估 obj(candidate)
6.     reward ← -obj(candidate)
7.     根据投资强度状态更新 Q[state]
8.     if obj(candidate) < obj(best) then
9.         best ← candidate, current ← candidate
10.    else with probability epsilon:
11.        current ← candidate
12.    end if
13. end for
14. return best
```

## 优势与局限

优势：
- 不需要完整求解原始大规模 MIP，可用于快速生成可行策略。
- 适合扩展到更多历史运营数据或仿真数据上，逐步学习港口改造策略。
- 可作为 Gurobi、Benders 或场景生成算法的 warm start。

局限：
- 当前实现是轻量级策略搜索，不是深度强化学习。
- 搜索质量受 episode 数、扰动规则和初始策略影响。
- 不保证全局最优，需要与精确解或下界进行对比。

## 本项目实现

代码文件：`rl_policy_algo.py`
运行脚本：`run_rl_policy.py`
结果文件：`rl_policy_results.json`
