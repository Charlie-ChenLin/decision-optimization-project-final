#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值实验主脚本 —— Gurobi直接求解 + EEV/VSS分析
================================================
功能：
  1. 生成不同规模的随机规划实例（3种规模）
  2. Gurobi直接求解DEP（确定性等价模型）
  3. EEV计算（对比基准）
  4. VSS分析
  5. 结果汇总输出

使用方式：
  python run_experiments.py

输出：
  - experiment_results.json : 汇总表格（可直接用于论文）
  - 终端显示求解时间、目标函数值、VSS

数据接口（供Benders算法组使用）：
  from run_experiments import generate_instance, Instance
  inst = generate_instance(n_zones=3, n_periods=24, n_scenes=20)
  # inst.h[(z,t,s)]     → 场景s下区域z时段t的需求
  # inst.mu_det[(z,t)]  → 确定性均值
  # inst.params         → 全部参数字典
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ============================================================
# 0. 数据结构
# ============================================================

@dataclass
class Instance:
    """问题实例数据 —— Benders算法组的数据接口"""
    n_zones: int
    n_periods: int
    n_scenes: int
    Z: list       # 区域集合
    T: list       # 时段集合
    N: list       # 周期集合
    S: list       # 场景集合
    h: dict       # h[z,t,s] : 场景s下区域z时段t的需求
    mu_det: dict  # mu_det[z,t] : 确定性均值
    params: dict  # 全部参数字典


# ============================================================
# 1. 实例生成（核心数据接口）
# ============================================================

def generate_instance(n_zones: int, n_periods: int, n_scenes: int,
                       seed: int = 42, cv: float = 0.15) -> Instance:
    """
    生成指定规模的随机规划实例。
    
    参数:
        n_zones   : 区域数 (建议 3~5)
        n_periods : 时段数 (建议 12/24/36，即1/2/3年)
        n_scenes  : 场景数 (建议 5/20/50)
        seed      : 随机种子 (默认42)
        cv        : 需求变异系数 (默认0.15)
    
    返回:
        Instance 对象，包含全部数据和参数
    
    使用示例:
        inst = generate_instance(3, 24, 20)
        print(inst.h[(0, 1, 0)])  # 区域0, 时段1, 场景0的需求
    """
    np.random.seed(seed)
    
    Z = list(range(n_zones))
    T = list(range(1, n_periods + 1))
    N = list(range(1, (n_periods // 12) + 1))
    S = list(range(n_scenes))
    
    # 基准需求（按区域大小递减）
    base = [10000 - i * 2000 for i in range(n_zones)]
    base = [max(b, 4000) for b in base]
    
    # 季节性因子（已归一化）
    seasonal_raw = {
        1: 0.98, 2: 0.78, 3: 0.95, 4: 0.96, 5: 0.97, 6: 0.98,
        7: 0.97, 8: 1.06, 9: 1.05, 10: 1.04, 11: 1.12, 12: 1.16,
    }
    avg_s = sum(seasonal_raw.values()) / 12
    seasonal = {t: v / avg_s for t, v in seasonal_raw.items()}
    
    # 同比增长（上海港2024年实际增速）
    growth = 0.048
    
    # 确定性均值
    mu_det = {}
    for z in Z:
        for t in T:
            month = ((t - 1) % 12) + 1
            year_mult = 1.0 if t <= 12 else (1.0 + growth)
            mu_det[(z, t)] = base[z] * seasonal[month] * year_mult
    
    # 对数正态分布场景采样
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    h = {}
    for z in Z:
        for t in T:
            mean_val = mu_det[(z, t)]
            mu_ln = np.log(mean_val) - sigma_ln**2 / 2
            for s in S:
                h[(z, t, s)] = np.random.lognormal(mu_ln, sigma_ln)
    
    # 参数（统一缩放，与LaTeX一致）
    params = {
        'k': 0.20,
        'cv': cv,
        'rho': 0.99,
        'p2': 75.0,       # EYC购买单价（万元/台）
        'v2': 50.0,       # DYC改造单价（万元/台）
        'd_cost': 50.0,   # 电网改造成本（万元/区域）
        'o1': 5.0,        # DYC运营成本（元/TEU）
        'o2': 3.0,        # EYC运营成本（元/TEU）
        'g1': 3250.0,     # DYC产能（TEU/台·月）
        'beta': 0.8,      # EYC/DYC产能比
        'e1': 3.2,        # DYC排放系数（kg CO2/TEU）
        'e2': 1.4,        # EYC排放系数（kg CO2/TEU）
        'c_feqp': 0.70,   # 排放削减系数
        'c_etp': 65.0,    # 超额排放成本（元/吨CO2）
        'tau': 2,         # 改造时滞（月）
        'Q1': 645.0,      # 初始碳排放免费额度（吨CO2）
        't_z': {z: max(2, 6 - z) for z in Z},  # 电网改造时长（月）
        'm_cap': {z: max(10, 15 - z) for z in Z}, # 区域容量上限（台）
        'f_init': {z: max(8, 10 - z) for z in Z}, # 初始DYC存量（台）
    }
    
    # 移动成本矩阵
    u = {}
    for z in Z:
        for zp in Z:
            u[(z, zp)] = 0.0 if z == zp else 2000.0 + 500.0 * abs(z - zp)
    params['u'] = u
    
    # 预算约束（万元/月）
    theta = {}; mu_b = {}; q = {}
    for t in T:
        theta[t] = 75.0    # 购买预算（1台EYC/月）
        mu_b[t] = 100.0    # 改造预算（2台DYC/月）
        q[t] = 50.0        # 电网预算（1区域/月）
    params['theta'] = theta
    params['mu'] = mu_b
    params['q'] = q
    
    return Instance(n_zones, n_periods, n_scenes, Z, T, N, S, h, mu_det, params)


# ============================================================
# 2. Gurobi直接求解DEP
# ============================================================

def solve_direct(inst: Instance, time_limit: int = 300,
                 mip_gap: float = 0.01) -> Tuple[float, float, dict]:
    """
    Gurobi直接求解确定性等价模型（DEP）。
    
    这是对比基准。Benders算法组需要与这个函数的输出对比。
    
    参数:
        inst       : Instance对象
        time_limit : 求解时间限制（秒）
        mip_gap    : MIP Gap容忍度
    
    返回:
        (obj_val, solve_time, details)
        - obj_val    : 最优目标函数值
        - solve_time : 实际求解时间（秒）
        - details    : 包含status、mip_gap、变量数、约束数字典
    """
    start = time.time()
    
    Z, T, N, S = inst.Z, inst.T, inst.N, inst.S
    h = inst.h
    p = inst.params
    nS = len(S)
    
    model = gp.Model("DEP_Direct")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('OutputFlag', 0)
    
    # ---- 一阶段变量 ----
    b2 = model.addVars(Z, T, lb=0.0, name="b2")
    y = model.addVars(Z, T, lb=0.0, name="y")
    a = model.addVars(Z, T, vtype=GRB.BINARY, name="a")
    r = model.addVars(Z, T, vtype=GRB.BINARY, name="r")
    x2 = model.addVars(Z, T, lb=0.0, name="x2")
    
    # ---- 二阶段变量 ----
    w1 = model.addVars(Z, T, S, lb=0.0, name="w1")
    w2 = model.addVars(Z, T, S, lb=0.0, name="w2")
    move = model.addVars(Z, Z, T, S, lb=0.0, name="move")  # 跨区移动
    x1 = model.addVars(Z, T, S, lb=0.0, name="x1")
    E = model.addVars(N, S, lb=0.0, name="E")
    dE = model.addVars(N, S, lb=0.0, name="dE")
    
    # ---- 目标函数 ----
    C_inv_raw = gp.quicksum(
        p['rho']**(t-1) * gp.quicksum(
            p['p2'] * b2[z,t] + p['v2'] * y[z,t] + p['d_cost'] * a[z,t]
            for z in Z) for t in T)
    C_inv = (1 - p['k']) * C_inv_raw
    
    C_op = gp.quicksum(
        p['rho']**(t-1) * gp.quicksum(
            p['o1'] * w1[z,t,s] + p['o2'] * w2[z,t,s] +
            gp.quicksum(p['u'][(z,zp)] * move[z,zp,t,s] for zp in Z)
            for z in Z) for t in T for s in S) / nS
    
    C_carbon = gp.quicksum(
        p['rho']**(12*n-1) * p['c_etp'] * dE[n,s]
        for n in N for s in S) / nS
    
    model.setObjective(C_inv + C_op + C_carbon, GRB.MINIMIZE)
    
    # ---- 一阶段约束 ----
    for t in T:
        model.addConstr(gp.quicksum(p['p2'] * b2[z,t] for z in Z) <= p['theta'][t])
        model.addConstr(gp.quicksum(p['v2'] * y[z,t] for z in Z) <= p['mu'][t])
        model.addConstr(gp.quicksum(p['d_cost'] * a[z,t] for z in Z) <= p['q'][t])
    
    for z in Z:
        model.addConstr(x2[z,1] == b2[z,1])
        for t in T:
            if t == 1: continue
            if t > p['tau']:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t] + y[z,t-p['tau']])
            else:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t])
        model.addConstr(gp.quicksum(a[z,t] for t in T) <= 1)
        for t in T:
            if t > p['t_z'][z]:
                upper = t - p['t_z'][z]
                valid_t = [tp for tp in T if tp <= upper]
                if valid_t:
                    model.addConstr(r[z,t] == gp.quicksum(a[z,tp] for tp in valid_t))
                else:
                    model.addConstr(r[z,t] == 0)
            else:
                model.addConstr(r[z,t] == 0)
        for t in T:
            model.addConstr(x2[z,t] <= p['m_cap'][z] * r[z,t])
            # 购买和改造只能在电网完成后进行
            model.addConstr(b2[z,t] <= p['m_cap'][z] * r[z,t])
            if t + p['tau'] <= max(T):
                model.addConstr(y[z,t] <= p['f_init'][z] * r[z, t + p['tau']])
    
    # ---- 二阶段约束（每个场景） ----
    for s in S:
        for z in Z:
            for t in T:
                model.addConstr(w1[z,t,s] + w2[z,t,s] == h[z,t,s])
                model.addConstr(w1[z,t,s] <= p['g1'] * x1[z,t,s])
                model.addConstr(w2[z,t,s] <= p['beta'] * p['g1'] * x2[z,t])
        
        for z in Z:
            model.addConstr(
                x1[z,1,s] == p['f_init'][z] - y[z,1]
                + gp.quicksum(move[zp,z,1,s] for zp in Z)
                - gp.quicksum(move[z,zp,1,s] for zp in Z))
            for t in T:
                if t > 1:
                    model.addConstr(
                        x1[z,t,s] == x1[z,t-1,s] - y[z,t]
                        + gp.quicksum(move[zp,z,t,s] for zp in Z)
                        - gp.quicksum(move[z,zp,t,s] for zp in Z))
            model.addConstr(y[z,1] <= p['f_init'][z])
            for t in T:
                if t > 1:
                    model.addConstr(y[z,t] <= x1[z,t-1,s])
        
        for n in N:
            t_start = 12*(n-1) + 1
            t_end = min(12*n, max(T))
            model.addConstr(E[n,s] == gp.quicksum(
                0.001 * (p['e1']*w1[z,t,s] + p['e2']*w2[z,t,s])
                for t in range(t_start, t_end+1) for z in Z))
        
        for n in N:
            if n == min(N):
                model.addConstr(dE[n,s] >= E[n,s] - p['Q1'])
            else:
                model.addConstr(dE[n,s] >= E[n,s] - p['c_feqp'] * E[n-1,s])
        
        for z in Z:
            for t in T:
                model.addConstr(x1[z,t,s] + x2[z,t] <= p['m_cap'][z])
    
    model.optimize()
    
    solve_time = time.time() - start
    n_vars = model.NumVars
    n_constrs = model.NumConstrs
    
    if model.status == GRB.INFEASIBLE:
        print(f"      模型不可行，正在分析IIS...")
        model.computeIIS()
        model.write("model.ilp")
        print(f"      IIS已写入 model.ilp")
        print(f"      冲突约束（前10个）：")
        count = 0
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"        {c.ConstrName}")
                count += 1
                if count >= 10:
                    break
        for v in model.getVars():
            if v.IISLB:
                print(f"        LB: {v.VarName}")
                count += 1
                if count >= 10:
                    break
            if v.IISUB:
                print(f"        UB: {v.VarName}")
                count += 1
                if count >= 10:
                    break
    
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # 提取第一阶段决策（用于EEV计算）
        first_stage = {
            'b2': {(z,t): b2[z,t].X for z in Z for t in T},
            'y': {(z,t): y[z,t].X for z in Z for t in T},
            'a': {(z,t): a[z,t].X for z in Z for t in T},
            'x2': {(z,t): x2[z,t].X for z in Z for t in T},
        }
        expected_total_emissions = sum(E[n,s].X for n in N for s in S) / nS
        expected_excess_emissions = sum(dE[n,s].X for n in N for s in S) / nS
        eyc_purchase = sum(b2[z,t].X for z in Z for t in T)
        eyc_conversion = sum(y[z,t].X for z in Z for t in T)
        terminal_eyc_stock = sum(x2[z,max(T)].X for z in Z)
        raw_inv_value = C_inv_raw.getValue()
        return model.ObjVal, solve_time, {
            'status': model.status,
            'mip_gap': model.MIPGap if model.SolCount > 0 else None,
            'n_vars': n_vars,
            'n_constrs': n_constrs,
            'cost_breakdown': {
                'investment_after_subsidy': C_inv.getValue(),
                'operation': C_op.getValue(),
                'carbon': C_carbon.getValue(),
                'investment_before_subsidy': raw_inv_value,
            },
            'expected_total_emissions': expected_total_emissions,
            'expected_excess_emissions': expected_excess_emissions,
            'government_subsidy': p['k'] * raw_inv_value,
            'eyc_purchase': eyc_purchase,
            'eyc_conversion': eyc_conversion,
            'eyc_total_investment': eyc_purchase + eyc_conversion,
            'terminal_eyc_stock': terminal_eyc_stock,
        }, first_stage
    else:
        return float('inf'), solve_time, {
            'status': model.status,
            'n_vars': n_vars,
            'n_constrs': n_constrs,
        }, None


# ============================================================
# 3. EEV计算
# ============================================================

def compute_eev(inst: Instance, time_limit: int = 300) -> float:
    """
    计算EEV（Expected Result of using the EV solution）。
    
    步骤：
      1. 用均值场景求解确定性模型 → 得到一阶段决策 x̄
      2. 将 x̄ 代入随机场景 → 计算真实期望成本
    
    EEV > RP 说明"按平均值规划"在不确定环境下表现更差。
    """
    # 第一步：均值场景确定性模型
    mean_inst = Instance(
        inst.n_zones, inst.n_periods, 1,
        inst.Z, inst.T, inst.N, [0],
        {(z,t,0): inst.mu_det[(z,t)] for z in inst.Z for t in inst.T},
        inst.mu_det, inst.params
    )
    
    
    obj_ev, _, _, ev_first_stage = solve_direct(mean_inst, time_limit=time_limit)
    if ev_first_stage is None:
        return float('inf')
        
    # 第二步：将第一阶段决策固定，代入原随机场景进行评估（EEV）
    # 为此我们需要再次建立原始模型，但是固定一阶段决策变量的值
    start = time.time()
    Z, T, N, S = inst.Z, inst.T, inst.N, inst.S
    h = inst.h
    p = inst.params
    nS = len(S)

    model = gp.Model("DEP_EEV")
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 0.01)
    model.setParam('OutputFlag', 0)

    # ---- 一阶段变量（固定值为EV的解） ----
    b2 = model.addVars(Z, T, lb=0.0, name="b2")
    y = model.addVars(Z, T, lb=0.0, name="y")
    a = model.addVars(Z, T, vtype=GRB.BINARY, name="a")
    r = model.addVars(Z, T, vtype=GRB.BINARY, name="r")
    x2 = model.addVars(Z, T, lb=0.0, name="x2")

    for z in Z:
        for t in T:
            b2[z,t].LB = ev_first_stage['b2'][z,t]
            b2[z,t].UB = ev_first_stage['b2'][z,t]
            y[z,t].LB = ev_first_stage['y'][z,t]
            y[z,t].UB = ev_first_stage['y'][z,t]
            a[z,t].LB = round(ev_first_stage['a'][z,t])
            a[z,t].UB = round(ev_first_stage['a'][z,t])

    # ---- 二阶段变量 ----
    w1 = model.addVars(Z, T, S, lb=0.0, name="w1")
    w2 = model.addVars(Z, T, S, lb=0.0, name="w2")
    move = model.addVars(Z, Z, T, S, lb=0.0, name="move")
    x1 = model.addVars(Z, T, S, lb=0.0, name="x1")
    E = model.addVars(N, S, lb=0.0, name="E")
    dE = model.addVars(N, S, lb=0.0, name="dE")

    # ---- 目标函数 ----
    C_inv = (1 - p['k']) * gp.quicksum(
        p['rho']**(t-1) * gp.quicksum(
            p['p2'] * b2[z,t] + p['v2'] * y[z,t] + p['d_cost'] * a[z,t]
            for z in Z) for t in T)

    C_op = gp.quicksum(
        p['rho']**(t-1) * gp.quicksum(
            p['o1'] * w1[z,t,s] + p['o2'] * w2[z,t,s] +
            gp.quicksum(p['u'][(z,zp)] * move[z,zp,t,s] for zp in Z)
            for z in Z) for t in T for s in S) / nS

    C_carbon = gp.quicksum(
        p['rho']**(12*n-1) * p['c_etp'] * dE[n,s]
        for n in N for s in S) / nS

    model.setObjective(C_inv + C_op + C_carbon, GRB.MINIMIZE)

    # ---- 一阶段约束 ----
    for t in T:
        model.addConstr(gp.quicksum(p['p2'] * b2[z,t] for z in Z) <= p['theta'][t])
        model.addConstr(gp.quicksum(p['v2'] * y[z,t] for z in Z) <= p['mu'][t])
        model.addConstr(gp.quicksum(p['d_cost'] * a[z,t] for z in Z) <= p['q'][t])

    for z in Z:
        model.addConstr(x2[z,1] == b2[z,1])
        for t in T:
            if t == 1: continue
            if t > p['tau']:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t] + y[z,t-p['tau']])
            else:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t])
        model.addConstr(gp.quicksum(a[z,t] for t in T) <= 1)
        for t in T:
            if t > p['t_z'][z]:
                upper = t - p['t_z'][z]
                valid_t = [tp for tp in T if tp <= upper]
                if valid_t:
                    model.addConstr(r[z,t] == gp.quicksum(a[z,tp] for tp in valid_t))
                else:
                    model.addConstr(r[z,t] == 0)
            else:
                model.addConstr(r[z,t] == 0)
        for t in T:
            model.addConstr(x2[z,t] <= p['m_cap'][z] * r[z,t])
            # 购买和改造只能在电网完成后进行
            model.addConstr(b2[z,t] <= p['m_cap'][z] * r[z,t])
            if t + p['tau'] <= max(T):
                model.addConstr(y[z,t] <= p['f_init'][z] * r[z, t + p['tau']])

    # ---- 二阶段约束（每个场景） ----
    for s in S:
        for z in Z:
            for t in T:
                model.addConstr(w1[z,t,s] + w2[z,t,s] == h[z,t,s])
                model.addConstr(w1[z,t,s] <= p['g1'] * x1[z,t,s])
                model.addConstr(w2[z,t,s] <= p['beta'] * p['g1'] * x2[z,t])

        for z in Z:
            model.addConstr(
                x1[z,1,s] == p['f_init'][z] - y[z,1]
                + gp.quicksum(move[zp,z,1,s] for zp in Z)
                - gp.quicksum(move[z,zp,1,s] for zp in Z))
            for t in T:
                if t > 1:
                    model.addConstr(
                        x1[z,t,s] == x1[z,t-1,s] - y[z,t]
                        + gp.quicksum(move[zp,z,t,s] for zp in Z)
                        - gp.quicksum(move[z,zp,t,s] for zp in Z))
            model.addConstr(y[z,1] <= p['f_init'][z])
            for t in T:
                if t > 1:
                    model.addConstr(y[z,t] <= x1[z,t-1,s])

        for n in N:
            t_start = 12*(n-1) + 1
            t_end = min(12*n, max(T))
            model.addConstr(E[n,s] == gp.quicksum(
                0.001 * (p['e1']*w1[z,t,s] + p['e2']*w2[z,t,s])
                for t in range(t_start, t_end+1) for z in Z))

        for n in N:
            if n == min(N):
                model.addConstr(dE[n,s] >= E[n,s] - p['Q1'])
            else:
                model.addConstr(dE[n,s] >= E[n,s] - p['c_feqp'] * E[n-1,s])

        for z in Z:
            for t in T:
                model.addConstr(x1[z,t,s] + x2[z,t] <= p['m_cap'][z])

    model.optimize()
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return model.ObjVal, ev_first_stage
    else:
        return float('inf'), None


# ============================================================
# 4. 辅助函数：汇总分析第一阶段决策
# ============================================================

def summarize_first_stage(fs: dict) -> Tuple[float, float, float]:
    """汇总第一阶段的总购买量、总改造量、总电网改造区域数"""
    if not fs:
        return 0.0, 0.0, 0.0
    sum_b2 = sum(fs['b2'].values())
    sum_y = sum(fs['y'].values())
    sum_a = sum(fs['a'].values())
    return sum_b2, sum_y, sum_a


# ============================================================
# 5. 主实验
# ============================================================

def run_experiments():
    """运行全部数值实验并输出汇总表格"""
    
    print("=" * 80)
    print("数值实验：两阶段随机规划模型（Gurobi直接求解）")
    print("=" * 80)
    
    # 三种规模
    scales = [
        ("小规模", 3, 12, 5),
        ("中规模", 3, 24, 20),
        ("大规模", 5, 36, 50),
    ]
    
    results = []
    
    for name, nz, np_, ns in scales:
        print(f"\n{'='*60}")
        print(f"【{name}】区域={nz}, 时段={np_}, 场景={ns}")
        print(f"{'='*60}")
        
        inst = generate_instance(nz, np_, ns)
        print(f"  变量规模: 一阶段~{nz*np_*5}, 二阶段~{nz*np_*ns*6}")
        
        # Gurobi直接求解
        print(f"\n  [1] Gurobi直接求解DEP...")
        obj_g, t_g, det_g, fs_g = solve_direct(inst, time_limit=120 if ns <= 20 else 300)
        gap_str = f"{det_g.get('mip_gap'):.2%}" if det_g.get('mip_gap') is not None else "N/A"
        print(f"      OBJ={obj_g:.2f}, 时间={t_g:.2f}s, Gap={gap_str}")
        
        # EEV 计算（所有规模都计算，便于分析对比）
        print(f"\n  [2] EEV计算（均值场景→随机场景评估）...")
        eev, fs_ev = compute_eev(inst, time_limit=120)
        print(f"      EEV={eev:.2f}" if eev != float('inf') else "      EEV=inf")
        
        # VSS
        vss = eev - obj_g if eev != float('inf') else None
        if vss is not None:
            print(f"\n  [3] VSS = EEV - RP = {vss:.2f}")
            if vss > 0:
                print(f"      >> 考虑不确定性节省 {vss:.2f} 万元（{vss/eev*100:.1f}%）")
                
        # 第一阶段决策对比分析
        sum_b2_g, sum_y_g, sum_a_g = summarize_first_stage(fs_g)
        sum_b2_ev, sum_y_ev, sum_a_ev = summarize_first_stage(fs_ev)
        
        print("\n  [4] 第一阶段决策对比 (随机模型 DEP vs 确定性均值模型 EV):")
        print(f"      - 电网改造总次数 (a)  : DEP = {sum_a_g:>5.1f} | EV = {sum_a_ev:>5.1f}")
        print(f"      - 购入EYC总数      (b2) : DEP = {sum_b2_g:>5.1f} | EV = {sum_b2_ev:>5.1f}")
        print(f"      - 改造DYC总数      (y)  : DEP = {sum_y_g:>5.1f} | EV = {sum_y_ev:>5.1f}")
        
        if sum_y_g > sum_y_ev or sum_a_g > sum_a_ev:
            print("      => 结论: 考虑不确定性后，模型倾向于提前更积极地进行电网或设备改造以对冲风险。")
        elif sum_y_g < sum_y_ev or sum_a_g < sum_a_ev:
            print("      => 结论: 考虑不确定性后，模型做出的投资决策相比均值模型更为保守谨慎。")
        
        results.append({
            'scale': name,
            'nz': nz, 'np': np_, 'ns': ns,
            'gurobi_obj': obj_g,
            'gurobi_time': t_g,
            'gurobi_gap': det_g.get('mip_gap'),
            'n_vars': det_g.get('n_vars'),
            'n_constrs': det_g.get('n_constrs'),
            'eev': eev if eev != float('inf') else None,
            'vss': vss,
            'dep_decisions': {'b2': sum_b2_g, 'y': sum_y_g, 'a': sum_a_g},
            'ev_decisions': {'b2': sum_b2_ev, 'y': sum_y_ev, 'a': sum_a_ev}
        })
    
    # ---- 汇总表格 ----
    print(f"\n\n{'='*80}")
    print("结果汇总表（可直接复制到论文）")
    print(f"{'='*80}")
    print(f"{'规模':<10} {'区域':>4} {'时段':>4} {'场景':>4} "
          f"{'变量数':>8} {'约束数':>8} {'Gurobi OBJ':>12} {'时间(s)':>8} {'VSS':>10}")
    print("-" * 80)
    for r in results:
        vss_str = f"{r['vss']:.2f}" if r['vss'] is not None else "N/A"
        n_vars = r['n_vars'] if r['n_vars'] is not None else 0
        n_constrs = r['n_constrs'] if r['n_constrs'] is not None else 0
        obj_str = f"{r['gurobi_obj']:>12.2f}" if r['gurobi_obj'] < float('inf') else "     inf"
        print(f"{r['scale']:<10} {r['nz']:>4} {r['np']:>4} {r['ns']:>4} "
              f"{n_vars:>8} {n_constrs:>8} {obj_str} "
              f"{r['gurobi_time']:>8.1f} {vss_str:>10}")
    
    # 保存JSON
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: experiment_results.json")


if __name__ == "__main__":
    run_experiments()
