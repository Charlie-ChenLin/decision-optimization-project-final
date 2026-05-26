import time
import random
import numpy as np
from run_experiments import Instance, solve_direct
from evaluation_utils import make_mean_instance, solve_fixed_first_stage


def _clone_fs(fs):
    return {key: dict(value) for key, value in fs.items()}


def _policy_score(inst: Instance, fs: dict):
    invest = 0.0
    for z in inst.Z:
        for t in inst.T:
            invest += fs["a"][(z, t)] * 10.0 + fs["b2"][(z, t)] + fs["y"][(z, t)]
    return invest


def _mutate_policy(inst: Instance, fs: dict, rng: random.Random, epsilon: float):
    new_fs = _clone_fs(fs)
    p = inst.params
    if rng.random() < epsilon:
        z = rng.choice(inst.Z)
        current_times = [t for t in inst.T if round(new_fs["a"][(z, t)]) >= 1]
        for t in inst.T:
            new_fs["a"][(z, t)] = 0.0
        if current_times and rng.random() < 0.35:
            pass
        else:
            t = rng.choice(inst.T)
            if p["d_cost"] <= p["q"][t]:
                new_fs["a"][(z, t)] = 1.0
    for z in inst.Z:
        renovated_periods = []
        for t in inst.T:
            finish = any(round(new_fs["a"][(z, tp)]) >= 1 and tp + p["t_z"][z] < t + 1 for tp in inst.T)
            if finish:
                renovated_periods.append(t)
        if not renovated_periods:
            for t in inst.T:
                new_fs["b2"][(z, t)] = 0.0
                new_fs["y"][(z, t)] = 0.0
            continue
        for t in inst.T:
            if rng.random() < epsilon:
                if t in renovated_periods and p["p2"] <= p["theta"][t]:
                    new_fs["b2"][(z, t)] = max(0.0, min(p["theta"][t] / p["p2"], new_fs["b2"][(z, t)] + rng.choice([-1.0, 0.0, 1.0])))
            if rng.random() < epsilon:
                if t + p["tau"] <= max(inst.T) and p["v2"] <= p["mu"][t]:
                    new_fs["y"][(z, t)] = max(0.0, min(p["mu"][t] / p["v2"], new_fs["y"][(z, t)] + rng.choice([-1.0, 0.0, 1.0])))
    return new_fs


def solve_rl_policy(inst: Instance, episodes: int = 12, epsilon: float = 0.35, time_limit: int = 300, seed: int = 42):
    start = time.time()
    rng = random.Random(seed)
    mean_inst = make_mean_instance(inst)
    _, _, _, base_fs = solve_direct(mean_inst, time_limit=max(1, int(time_limit * 0.25)))
    if base_fs is None:
        return float("inf"), time.time() - start, {"status": "failed_at_initial_policy"}

    q_table = {}
    initial_obj, initial_time, initial_detail = solve_fixed_first_stage(inst, base_fs, time_limit=max(1, int(time_limit - (time.time() - start))))
    best = {"obj": initial_obj, "fs": base_fs, "episode": "initial_mean_policy"}
    current_fs = base_fs
    history = [{"episode": "initial_mean_policy", "obj": initial_obj, "eval_time": initial_time, "status": initial_detail.get("status"), "state": "initial", "accepted": True}]

    for episode in range(episodes):
        if time.time() - start >= time_limit:
            break
        candidate_fs = _mutate_policy(inst, current_fs, rng, epsilon)
        remaining = max(1, int(time_limit - (time.time() - start)))
        obj, eval_time, detail = solve_fixed_first_stage(inst, candidate_fs, time_limit=remaining)
        state = "high_invest" if _policy_score(inst, candidate_fs) > _policy_score(inst, base_fs) else "low_invest"
        reward = -obj if obj < float("inf") else -1e18
        q_table[state] = 0.8 * q_table.get(state, reward) + 0.2 * reward
        accepted = obj < best["obj"]
        if accepted:
            best = {"obj": obj, "fs": candidate_fs, "episode": episode}
            current_fs = candidate_fs
        elif rng.random() < epsilon:
            current_fs = candidate_fs
        history.append({"episode": episode, "obj": obj, "eval_time": eval_time, "status": detail.get("status"), "state": state, "accepted": accepted})

    return best["obj"], time.time() - start, {"best_episode": best["episode"], "episodes": len(history), "q_table": q_table, "history": history}
