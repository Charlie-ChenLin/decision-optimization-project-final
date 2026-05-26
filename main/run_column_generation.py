import json
from run_experiments import generate_instance, solve_direct
from column_generation_algo import solve_column_generation


def main():
    scales = [
        ("小规模", 3, 12, 5),
        ("中规模", 3, 24, 20),
        ("大规模", 5, 36, 30),
    ]
    results = []
    print("列生成/场景生成式算法实验")
    print("=" * 80)
    for name, nz, nt, ns in scales:
        inst = generate_instance(nz, nt, ns)
        print(f"\n[{name}] 区域={nz}, 时段={nt}, 场景={ns}")
        base_obj, base_time, base_detail, _ = solve_direct(inst, time_limit=120)
        cg_obj, cg_time, cg_detail = solve_column_generation(inst, initial_scenes=2, add_per_iter=2, max_iter=5, time_limit=120)
        gap = None if base_obj == float("inf") else (cg_obj - base_obj) / base_obj
        row = {
            "scale": name,
            "zones": nz,
            "periods": nt,
            "scenes": ns,
            "gurobi_obj": base_obj,
            "gurobi_time": base_time,
            "column_generation_obj": cg_obj,
            "column_generation_time": cg_time,
            "relative_gap_to_gurobi": gap,
            "details": cg_detail,
        }
        results.append(row)
        gap_text = "N/A" if gap is None else f"{gap:.2%}"
        print(f"Gurobi OBJ={base_obj:.2f}, Time={base_time:.2f}s")
        print(f"ColumnGeneration OBJ={cg_obj:.2f}, Time={cg_time:.2f}s, Gap={gap_text}")

    with open("column_generation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: column_generation_results.json")


if __name__ == "__main__":
    main()
