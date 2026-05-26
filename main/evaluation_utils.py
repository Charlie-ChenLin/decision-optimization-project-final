import gurobipy as gp
from gurobipy import GRB
from run_experiments import Instance


def make_subinstance(inst: Instance, selected_scenes):
    scenes = list(range(len(selected_scenes)))
    h = {}
    for new_s, old_s in enumerate(selected_scenes):
        for z in inst.Z:
            for t in inst.T:
                h[(z, t, new_s)] = inst.h[(z, t, old_s)]
    return Instance(
        inst.n_zones,
        inst.n_periods,
        len(scenes),
        inst.Z,
        inst.T,
        inst.N,
        scenes,
        h,
        inst.mu_det,
        inst.params,
    )


def make_mean_instance(inst: Instance):
    return Instance(
        inst.n_zones,
        inst.n_periods,
        1,
        inst.Z,
        inst.T,
        inst.N,
        [0],
        {(z, t, 0): inst.mu_det[(z, t)] for z in inst.Z for t in inst.T},
        inst.mu_det,
        inst.params,
    )


def solve_fixed_first_stage(inst: Instance, fs: dict, time_limit: int = 300):
    Z, T, N, S = inst.Z, inst.T, inst.N, inst.S
    h = inst.h
    p = inst.params
    nS = len(S)

    model = gp.Model("Fixed_First_Stage_Evaluation")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.01)
    model.setParam("OutputFlag", 0)

    b2 = model.addVars(Z, T, lb=0.0, name="b2")
    y = model.addVars(Z, T, lb=0.0, name="y")
    a = model.addVars(Z, T, vtype=GRB.BINARY, name="a")
    r = model.addVars(Z, T, vtype=GRB.BINARY, name="r")
    x2 = model.addVars(Z, T, lb=0.0, name="x2")

    for z in Z:
        for t in T:
            b2[z, t].LB = fs["b2"][(z, t)]
            b2[z, t].UB = fs["b2"][(z, t)]
            y[z, t].LB = fs["y"][(z, t)]
            y[z, t].UB = fs["y"][(z, t)]
            fixed_a = round(fs["a"][(z, t)])
            a[z, t].LB = fixed_a
            a[z, t].UB = fixed_a

    w1 = model.addVars(Z, T, S, lb=0.0, name="w1")
    w2 = model.addVars(Z, T, S, lb=0.0, name="w2")
    move = model.addVars(Z, Z, T, S, lb=0.0, name="move")
    x1 = model.addVars(Z, T, S, lb=0.0, name="x1")
    E = model.addVars(N, S, lb=0.0, name="E")
    dE = model.addVars(N, S, lb=0.0, name="dE")

    C_inv = (1 - p["k"]) * gp.quicksum(
        p["rho"] ** (t - 1)
        * gp.quicksum(p["p2"] * b2[z, t] + p["v2"] * y[z, t] + p["d_cost"] * a[z, t] for z in Z)
        for t in T
    )

    C_op = gp.quicksum(
        p["rho"] ** (t - 1)
        * gp.quicksum(
            p["o1"] * w1[z, t, s]
            + p["o2"] * w2[z, t, s]
            + gp.quicksum(p["u"][(z, zp)] * move[z, zp, t, s] for zp in Z)
            for z in Z
        )
        for t in T
        for s in S
    ) / nS

    C_carbon = gp.quicksum(p["rho"] ** (12 * n - 1) * p["c_etp"] * dE[n, s] for n in N for s in S) / nS

    model.setObjective(C_inv + C_op + C_carbon, GRB.MINIMIZE)

    for t in T:
        model.addConstr(gp.quicksum(p["p2"] * b2[z, t] for z in Z) <= p["theta"][t])
        model.addConstr(gp.quicksum(p["v2"] * y[z, t] for z in Z) <= p["mu"][t])
        model.addConstr(gp.quicksum(p["d_cost"] * a[z, t] for z in Z) <= p["q"][t])

    for z in Z:
        model.addConstr(x2[z, 1] == b2[z, 1])
        for t in T:
            if t == 1:
                continue
            if t > p["tau"]:
                model.addConstr(x2[z, t] == x2[z, t - 1] + b2[z, t] + y[z, t - p["tau"]])
            else:
                model.addConstr(x2[z, t] == x2[z, t - 1] + b2[z, t])
        model.addConstr(gp.quicksum(a[z, t] for t in T) <= 1)
        for t in T:
            if t > p["t_z"][z]:
                valid_t = [tp for tp in T if tp <= t - p["t_z"][z]]
                model.addConstr(r[z, t] == gp.quicksum(a[z, tp] for tp in valid_t) if valid_t else 0)
            else:
                model.addConstr(r[z, t] == 0)
            model.addConstr(x2[z, t] <= p["m_cap"][z] * r[z, t])
            model.addConstr(b2[z, t] <= p["m_cap"][z] * r[z, t])
            if t + p["tau"] <= max(T):
                model.addConstr(y[z, t] <= p["f_init"][z] * r[z, t + p["tau"]])

    for s in S:
        for z in Z:
            for t in T:
                model.addConstr(w1[z, t, s] + w2[z, t, s] == h[z, t, s])
                model.addConstr(w1[z, t, s] <= p["g1"] * x1[z, t, s])
                model.addConstr(w2[z, t, s] <= p["beta"] * p["g1"] * x2[z, t])
        for z in Z:
            model.addConstr(
                x1[z, 1, s]
                == p["f_init"][z]
                - y[z, 1]
                + gp.quicksum(move[zp, z, 1, s] for zp in Z)
                - gp.quicksum(move[z, zp, 1, s] for zp in Z)
            )
            for t in T:
                if t > 1:
                    model.addConstr(
                        x1[z, t, s]
                        == x1[z, t - 1, s]
                        - y[z, t]
                        + gp.quicksum(move[zp, z, t, s] for zp in Z)
                        - gp.quicksum(move[z, zp, t, s] for zp in Z)
                    )
            model.addConstr(y[z, 1] <= p["f_init"][z])
            for t in T:
                if t > 1:
                    model.addConstr(y[z, t] <= x1[z, t - 1, s])
        for n in N:
            t_start = 12 * (n - 1) + 1
            t_end = min(12 * n, max(T))
            model.addConstr(
                E[n, s]
                == gp.quicksum(0.001 * (p["e1"] * w1[z, t, s] + p["e2"] * w2[z, t, s]) for t in range(t_start, t_end + 1) for z in Z)
            )
        for n in N:
            if n == min(N):
                model.addConstr(dE[n, s] >= E[n, s] - p["Q1"])
            else:
                model.addConstr(dE[n, s] >= E[n, s] - p["c_feqp"] * E[n - 1, s])
        for z in Z:
            for t in T:
                model.addConstr(x1[z, t, s] + x2[z, t] <= p["m_cap"][z])

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return model.ObjVal, model.Runtime, {"status": model.status, "mip_gap": model.MIPGap if model.SolCount > 0 else None}
    return float("inf"), model.Runtime, {"status": model.status, "mip_gap": None}
