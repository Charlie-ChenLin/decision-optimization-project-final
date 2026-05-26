import time
from run_experiments import Instance, solve_direct
from evaluation_utils import make_subinstance, solve_fixed_first_stage


def _scenario_score(inst: Instance, s: int):
    total = 0.0
    peak = 0.0
    for z in inst.Z:
        for t in inst.T:
            val = inst.h[(z, t, s)]
            total += val
            peak = max(peak, val / max(1.0, inst.mu_det[(z, t)]))
    return 0.7 * total + 0.3 * peak * total


def solve_column_generation(inst: Instance, initial_scenes: int = 2, add_per_iter: int = 1, max_iter: int = 6, time_limit: int = 300):
    start = time.time()
    ranked = sorted(inst.S, key=lambda s: _scenario_score(inst, s), reverse=True)
    selected = ranked[: min(initial_scenes, len(ranked))]
    history = []
    best = {"obj": float("inf"), "fs": None, "selected_scenes": list(selected)}

    for iteration in range(max_iter):
        if time.time() - start >= time_limit:
            break

        reduced_inst = make_subinstance(inst, selected)
        remaining_time = max(1, int(time_limit - (time.time() - start)))
        reduced_obj, reduced_time, reduced_detail, fs = solve_direct(reduced_inst, time_limit=remaining_time)

        if fs is None:
            history.append({"iteration": iteration, "selected_scenes": list(selected), "reduced_obj": reduced_obj, "eval_obj": float("inf")})
            break

        remaining_time = max(1, int(time_limit - (time.time() - start)))
        eval_obj, eval_time, eval_detail = solve_fixed_first_stage(inst, fs, time_limit=remaining_time)
        history.append(
            {
                "iteration": iteration,
                "selected_scenes": list(selected),
                "reduced_obj": reduced_obj,
                "reduced_time": reduced_time,
                "eval_obj": eval_obj,
                "eval_time": eval_time,
                "eval_status": eval_detail.get("status"),
            }
        )

        if eval_obj < best["obj"]:
            best = {"obj": eval_obj, "fs": fs, "selected_scenes": list(selected)}

        if len(selected) >= len(inst.S):
            break

        candidates = [s for s in ranked if s not in selected]
        selected.extend(candidates[:add_per_iter])

    return best["obj"], time.time() - start, {"selected_scenes": best["selected_scenes"], "iterations": len(history), "history": history}
