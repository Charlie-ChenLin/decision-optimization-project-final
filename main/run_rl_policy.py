import json
from run_experiments import generate_instance, solve_direct
from rl_policy_algo import solve_rl_policy


def main():
    scales = [
        ("小规模", 3, 12, 5),
        ("中规模", 3, 24, 20),
        ("大规模", 5, 36, 30),
    ]
    results = []
    print("强化学习/数据驱动策略搜索实验")
    print("=" * 80)
    for name, nz, nt, ns in scales:
        inst = generate_instance(nz, nt, ns)
        print(f"\n[{name}] 区域={nz}, 时段={nt}, 场景={ns}")
        base_obj, base_time, base_detail, _ = solve_direct(inst, time_limit=120)
        rl_obj, rl_time, rl_detail = solve_rl_policy(inst, episodes=8, epsilon=0.3, time_limit=120)
        gap = None if base_obj == float("inf") else (rl_obj - base_obj) / base_obj
        row = {
            "scale": name,
            "zones": nz,
            "periods": nt,
            "scenes": ns,
            "gurobi_obj": base_obj,
            "gurobi_time": base_time,
            "rl_policy_obj": rl_obj,
            "rl_policy_time": rl_time,
            "relative_gap_to_gurobi": gap,
            "details": rl_detail,
        }
        results.append(row)
        gap_text = "N/A" if gap is None else f"{gap:.2%}"
        print(f"Gurobi OBJ={base_obj:.2f}, Time={base_time:.2f}s")
        print(f"RLPolicy OBJ={rl_obj:.2f}, Time={rl_time:.2f}s, Gap={gap_text}")

    with open("rl_policy_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: rl_policy_results.json")


if __name__ == "__main__":
    main()
