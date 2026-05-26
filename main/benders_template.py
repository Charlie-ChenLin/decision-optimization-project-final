#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benders Decomposition Algorithm (L-shaped method) - Simple Working Version
====================================
Implemented:
  1. Master Problem: Contains full first-stage constraints, uses Multi-cut method theta[s]
  2. Subproblem: Solves for second-stage decisions using fixed first-stage solutions as constants
  3. Benders Cuts Generation: Automatically constructs Optimality & Feasibility cuts using Gurobi's Pi (dual variables) and FarkasDual
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from run_experiments import generate_instance, Instance, solve_direct

def solve_benders(inst: Instance, max_iter: int = 100,
                  time_limit: int = 300, tol: float = 0.001) -> tuple:
    start = time.time()
    Z, T, N, S = inst.Z, inst.T, inst.N, inst.S
    p = inst.params
    h = inst.h
    nS = len(S)
    
    # STEP 1: Define Master Problem
    master = gp.Model("Benders_Master")
    master.setParam("OutputFlag", 0)
    
    b2 = master.addVars(Z, T, lb=0.0, name="b2")
    y = master.addVars(Z, T, lb=0.0, name="y")
    a = master.addVars(Z, T, vtype=GRB.BINARY, name="a")
    r = master.addVars(Z, T, vtype=GRB.BINARY, name="r")
    x2 = master.addVars(Z, T, lb=0.0, name="x2")
    theta = master.addVars(S, lb=0.0, name="theta")
    
    C_inv = (1 - p["k"]) * gp.quicksum(
        p["rho"]**(t-1) * gp.quicksum(
            p["p2"] * b2[z,t] + p["v2"] * y[z,t] + p["d_cost"] * a[z,t]
            for z in Z) for t in T)
    master.setObjective(C_inv + gp.quicksum(theta[s] for s in S) / nS, GRB.MINIMIZE)
    
    for t in T:
        master.addConstr(gp.quicksum(p["p2"] * b2[z,t] for z in Z) <= p["theta"][t])
        master.addConstr(gp.quicksum(p["v2"] * y[z,t] for z in Z) <= p["mu"][t])
        master.addConstr(gp.quicksum(p["d_cost"] * a[z,t] for z in Z) <= p["q"][t])
    for z in Z:
        master.addConstr(x2[z,1] == b2[z,1])
        for t in T:
            if t == 1: continue
            if t > p["tau"]:
                master.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t] + y[z,t-p["tau"]])
            else:
                master.addConstr(x2[z,t] == x2[z,t-1] + b2[z,t])
        master.addConstr(gp.quicksum(a[z,t] for t in T) <= 1)
        for t in T:
            if t > p["t_z"][z]:
                upper = t - p["t_z"][z]
                valid_t = [tp for tp in T if tp <= upper]
                if valid_t:
                    master.addConstr(r[z,t] == gp.quicksum(a[z,tp] for tp in valid_t))
                else:
                    master.addConstr(r[z,t] == 0)
            else:
                master.addConstr(r[z,t] == 0)
        for t in T:
            master.addConstr(x2[z,t] <= p["m_cap"][z] * r[z,t])
            master.addConstr(b2[z,t] <= p["m_cap"][z] * r[z,t])
            if t + p["tau"] <= max(T):
                master.addConstr(y[z,t] <= p["f_init"][z] * r[z, t + p["tau"]])
            if t == 1:
                master.addConstr(y[z,1] <= p["f_init"][z])
                
    # STEP 2: Benders Iteration
    best_ub = float("inf")
    
    for iteration in range(max_iter):
        if time.time() - start > time_limit:
            break
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print("Master problem infeasible.")
            break
            
        x2_sol = {(z,t): x2[z,t].X for z in Z for t in T}
        y_sol = {(z,t): y[z,t].X for z in Z for t in T}
        theta_sol = {s: theta[s].X for s in S}
        
        current_lb = master.ObjVal
        current_sp_total = C_inv.getValue()
        n_cuts = 0

        for s in S:
            sp = gp.Model(f"SP_s{s}")
            sp.setParam("OutputFlag", 0)
            sp.setParam("InfUnbdInfo", 1)  # Enable retrieval of FarkasDual for infeasible models
            
            w1 = sp.addVars(Z, T, lb=0.0, name="w1")
            w2 = sp.addVars(Z, T, lb=0.0, name="w2")
            move = sp.addVars(Z, Z, T, lb=0.0, name="move")
            x1 = sp.addVars(Z, T, lb=0.0, name="x1")
            E = sp.addVars(N, lb=0.0, name="E")
            dE = sp.addVars(N, lb=0.0, name="dE")
            
            sp_obj = gp.quicksum(p["rho"]**(t-1) * (
                    p["o1"] * w1[z,t] + p["o2"] * w2[z,t] +
                    gp.quicksum(p["u"][(z,zp)] * move[z,zp,t] for zp in Z)
                ) for z in Z for t in T) + \
                gp.quicksum(p["rho"]**(12*n-1) * p["c_etp"] * dE[n] for n in N)
            sp.setObjective(sp_obj, GRB.MINIMIZE)
            
            sp_constrs = []
            
            for z in Z:
                for t in T:
                    c = sp.addConstr(w1[z,t] + w2[z,t] == h[z,t,s])
                    sp_constrs.append((c, h[z,t,s]))
            for z in Z:
                for t in T:
                    c = sp.addConstr(w1[z,t] - p["g1"]*x1[z,t] <= 0)
                    sp_constrs.append((c, 0.0))
                    c2 = sp.addConstr(w2[z,t] <= p["beta"]*p["g1"]*x2_sol[z,t])
                    sp_constrs.append((c2, p["beta"]*p["g1"]*x2[z,t]))
            for z in Z:
                c = sp.addConstr(x1[z,1] - gp.quicksum(move[zp,z,1] for zp in Z) 
                                 + gp.quicksum(move[z,zp,1] for zp in Z) == p["f_init"][z] - y_sol[z,1])
                sp_constrs.append((c, p["f_init"][z] - y[z,1]))
                for t in T:
                    if t > 1:
                        c = sp.addConstr(x1[z,t] - x1[z,t-1] - gp.quicksum(move[zp,z,t] for zp in Z)
                                         + gp.quicksum(move[z,zp,t] for zp in Z) == -y_sol[z,t])
                        sp_constrs.append((c, -y[z,t]))
                    if t > 1:
                        c = sp.addConstr(-x1[z,t-1] <= -y_sol[z,t])
                        sp_constrs.append((c, -y[z,t]))
            for z in Z:
                for t in T:
                    c = sp.addConstr(x1[z,t] <= p["m_cap"][z] - x2_sol[z,t])
                    sp_constrs.append((c, p["m_cap"][z] - x2[z,t]))
            for n in N:
                t_start = 12*(n-1) + 1
                t_end = min(12*n, max(T))
                c = sp.addConstr(E[n] - gp.quicksum(0.001 * (p["e1"]*w1[z,t] + p["e2"]*w2[z,t]) 
                                                    for t in range(t_start, t_end+1) for z in Z) == 0)
                sp_constrs.append((c, 0.0))
            for n in N:
                if n == min(N):
                    c = sp.addConstr(-dE[n] + E[n] <= p["Q1"])
                    sp_constrs.append((c, p["Q1"]))
                else:
                    c = sp.addConstr(-dE[n] + E[n] - p["c_feqp"]*E[n-1] <= 0)
                    sp_constrs.append((c, 0.0))

            sp.optimize()
            
            if sp.status == GRB.OPTIMAL:
                current_sp_total += sp.ObjVal / nS
                if theta_sol[s] < sp.ObjVal - 1e-4:
                    cut_expr = gp.quicksum(c.Pi * master_expr for c, master_expr in sp_constrs)
                    master.addConstr(theta[s] >= cut_expr)
                    n_cuts += 1
            elif sp.status == GRB.INFEASIBLE:
                cut_expr = gp.quicksum(c.FarkasDual * master_expr for c, master_expr in sp_constrs)
                master.addConstr(cut_expr >= 0)
                n_cuts += 1
                current_sp_total = float("inf")
                break 

        if current_sp_total < best_ub:
            best_ub = current_sp_total
            
        gap = (best_ub - current_lb) / max(1.0, current_lb)
        if gap < tol:
            print(f"  Converged! Iteration {iteration}, Gap={gap:.4%}")
            break
            
        print(f"  Iter {iteration}: LB={current_lb:.2f}, UB={best_ub:.2f}, Gap={gap:.2%}, New Cuts={n_cuts}")
    
    solve_time = time.time() - start
    if master.status == GRB.OPTIMAL:
        return master.ObjVal, solve_time, iteration + 1
    else:
        return float("inf"), solve_time, iteration + 1

if __name__ == "__main__":
    inst = generate_instance(3, 12, 5, seed=42)
    print("\n[1] Direct Solve with Gurobi...")
    obj_g, t_g, det, _ = solve_direct(inst, time_limit=60)
    print(f"    OBJ={obj_g:.2f}, Time={t_g:.2f}s")
    
    print("\n[2] Benders Decomposition...")
    obj_b, t_b, n_iter = solve_benders(inst, max_iter=50, time_limit=120)
    print(f"    OBJ={obj_b:.2f}, Time={t_b:.2f}s, Iterations={n_iter}")