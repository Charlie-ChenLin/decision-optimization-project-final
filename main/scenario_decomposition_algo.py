import time
import numpy as np
from run_experiments import Instance, solve_direct
from evaluation_utils import make_subinstance, solve_fixed_first_stage


def _extract_vector(fs, Z, T):
    values = []
    for key in ("a", "b2", "y"):
        for z in Z:
            for t in T:
                values.append(fs[key][(z, t)])
    return np.array(values, dtype=float)


def _distance(fs1, fs2, Z, T):
    return float(np.linalg.norm(_extract_vector(fs1, Z, T) - _extract_vector(fs2, Z, T)))


def solve_scenario_decomposition(inst: Instance, max_candidates: int = 8, time_limit: int = 300):
    start = time.time()
    candidates = []
    scenario_records = []

    for s in inst.S:
        if time.time() - start >= time_limit:
            break
        single_inst = make_subinstance(inst, [s])
        obj, solve_time, detail, fs = solve_direct(single_inst, time_limit=max(1, int((time_limit - (time.time() - start)) / 2)))
        if fs is None:
            scenario_records.append({"scenario": s, "single_obj": obj, "status": detail.get("status"), "accepted": False})
            continue
        accepted = True
        for cand in candidates:
            if _distance(fs, cand["fs"], inst.Z, inst.T) < 1e-5:
                accepted = False
                break
        record = {"scenario": s, "single_obj": obj, "solve_time": solve_time, "status": detail.get("status"), "accepted": accepted}
        scenario_records.append(record)
        if accepted:
            candidates.append({"scenario": s, "single_obj": obj, "fs": fs})
        if len(candidates) >= max_candidates:
            break

    best = {"obj": float("inf"), "candidate_scenario": None, "eval_detail": None}
    eval_records = []
    for cand in candidates:
        if time.time() - start >= time_limit:
            break
        eval_obj, eval_time, eval_detail = solve_fixed_first_stage(inst, cand["fs"], time_limit=max(1, int(time_limit - (time.time() - start))))
        row = {
            "candidate_scenario": cand["scenario"],
            "single_obj": cand["single_obj"],
            "eval_obj": eval_obj,
            "eval_time": eval_time,
            "eval_status": eval_detail.get("status"),
        }
        eval_records.append(row)
        if eval_obj < best["obj"]:
            best = {"obj": eval_obj, "candidate_scenario": cand["scenario"], "eval_detail": eval_detail}

    return best["obj"], time.time() - start, {"best_candidate_scenario": best["candidate_scenario"], "scenario_records": scenario_records, "eval_records": eval_records}
