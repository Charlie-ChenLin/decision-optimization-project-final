import json
from run_experiments import generate_instance, solve_direct
from scenario_decomposition_algo import solve_scenario_decomposition


def main():
    scales = [
        ("小规模", 3, 12, 5),
        ("中规模", 3, 24, 20),
        ("大规模", 5, 36, 30),
    ]
    results = []
    print("场景分解算法实验")
    print("=" * 80)
    for name, nz, nt, ns in scales:
        inst = generate_instance(nz, nt, ns)
        print(f"\n[{name}] 区域={nz}, 时段={nt}, 场景={ns}")
        base_obj, base_time, base_detail, _ = solve_direct(inst, time_limit=120)
        sd_obj, sd_time, sd_detail = solve_scenario_decomposition(inst, max_candidates=6, time_limit=120)
        gap = None if base_obj == float("inf") else (sd_obj - base_obj) / base_obj
        row = {
            "scale": name,
            "zones": nz,
            "periods": nt,
            "scenes": ns,
            "gurobi_obj": base_obj,
            "gurobi_time": base_time,
            "scenario_decomposition_obj": sd_obj,
            "scenario_decomposition_time": sd_time,
            "relative_gap_to_gurobi": gap,
            "details": sd_detail,
        }
        results.append(row)
        gap_text = "N/A" if gap is None else f"{gap:.2%}"
        print(f"Gurobi OBJ={base_obj:.2f}, Time={base_time:.2f}s")
        print(f"ScenarioDecomposition OBJ={sd_obj:.2f}, Time={sd_time:.2f}s, Gap={gap_text}")

    with open("scenario_decomposition_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: scenario_decomposition_results.json")


if __name__ == "__main__":
    main()
