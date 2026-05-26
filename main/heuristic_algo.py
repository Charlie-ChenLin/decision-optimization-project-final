import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from run_experiments import Instance, solve_direct

def solve_heuristic(inst: Instance, time_limit: int = 300) -> tuple:
    """
    Mean-Value Guided Heuristic (均值指导启发式算法)
    ---------------------------------------------------
    思路：
    1. 求解均值场景(Mean-Value)的确定性模型，获取第一阶段的决策(b2, y, a)。
    2. 对于原问题(带所有场景S)，将第一阶段的整数决策(a: 电网改造)固定为均值模型的解。
    3. 将第一阶段的连续决策(b2, y)作为初始可行解/下界传入，或者直接求解一个简化后的MIP。
    4. 由于去掉了二阶段问题中最难的整数变量组合(即电网是否改造决定了后续的购买和改造)，
       此时模型将退化为固定电网架构下的线性规划(或极简MIP)，求解速度大幅提升。
    """
    start = time.time()
    
    # 步骤 1：构建均值场景的确定性实例
    mean_inst = Instance(
        inst.n_zones, inst.n_periods, 1,
        inst.Z, inst.T, inst.N, [0],
        {(z,t,0): inst.mu_det[(z,t)] for z in inst.Z for t in inst.T},
        inst.mu_det, inst.params
    )
    
    # 求解均值模型获取启发式的第一阶段整数变量
    _, ev_time, _, ev_fs = solve_direct(mean_inst, time_limit=time_limit/3)
    
    if ev_fs is None:
        return float('inf'), time.time() - start, "Heuristic Failed at Step 1"
        
    # 步骤 2：构建原随机规划模型，但固定最难的整数变量 a (电网改造决策)
    Z, T, N, S = inst.Z, inst.T, inst.N, inst.S
    h = inst.h
    p = inst.params
    nS = len(S)
    
    model = gp.Model("Heuristic_MIP")
    model.setParam('TimeLimit', time_limit - (time.time() - start))
    model.setParam('MIPGap', 0.01)
    model.setParam('OutputFlag', 0)
    
    # ---- 一阶段变量 ----
    b2 = model.addVars(Z, T, lb=0.0, name="b2")
    y = model.addVars(Z, T, lb=0.0, name="y")
    # 核心启发式降维：直接将 a 固定为 EV 的解，甚至将其设为常数
    a = {}
    for z in Z:
        for t in T:
            a[(z,t)] = round(ev_fs['a'][z,t]) 
            
    r = model.addVars(Z, T, vtype=GRB.CONTINUOUS, name="r") # a固定后，r也可以松弛为连续
    x2 = model.addVars(Z, T, lb=0.0, name="x2")
    
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
            p['p2'] * b2[z,t] + p['v2'] * y[z,t] + p['d_cost'] * a[(z,t)]
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
        # q 约束因为 a 已经固定，可能会产生轻微违反，但既然固定了EV的解，自然满足原约束
        
    for z in Z:
        model.addConstr(x2[z,1] == b2[z,1])
        for t in T:
            if t == 1: continue
            if t > p['tau']:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t] + y[z,t-p['tau']])
            else:
                model.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t])
                
        for t in T:
            if t > p['t_z'][z]:
                upper = t - p['t_z'][z]
                valid_t = [tp for tp in T if tp <= upper]
                if valid_t:
                    model.addConstr(r[z,t] == gp.quicksum(a[(z,tp)] for tp in valid_t))
                else:
                    model.addConstr(r[z,t] == 0)
            else:
                model.addConstr(r[z,t] == 0)
                
        for t in T:
            model.addConstr(x2[z,t] <= p['m_cap'][z] * r[z,t])
            model.addConstr(b2[z,t] <= p['m_cap'][z] * r[z,t])
            if t + p['tau'] <= max(T):
                model.addConstr(y[z,t] <= p['f_init'][z] * r[z, t + p['tau']])
                
    # ---- 二阶段约束 ----
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

    # 为了让启发式效果更好，提供 EV 解作为 y 和 b2 的暖启动 (MIP Start)
    for z in Z:
        for t in T:
            y[z,t].Start = ev_fs['y'][z,t]
            b2[z,t].Start = ev_fs['b2'][z,t]
            
    model.optimize()
    
    solve_time = time.time() - start
    
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return model.ObjVal, solve_time, "Heuristic Success"
    else:
        return float('inf'), solve_time, "Heuristic Failed"
