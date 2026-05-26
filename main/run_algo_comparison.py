import time
import json
from run_experiments import generate_instance, solve_direct
from benders_template import solve_benders
from heuristic_algo import solve_heuristic

def run_algo_comparison():
    print("=" * 80)
    print("组员2任务：算法性能对比实验 (Gurobi直接求解 vs Benders分解)")
    print("=" * 80)
    
    scales = [
        ("小规模", 3, 12, 5),
        ("中规模", 3, 24, 20),
        # 考虑到时间限制，大规模设定为较小的场景数测试，以免运行超时
        ("大规模", 5, 36, 30) 
    ]
    
    results = []
    
    for name, nz, np_, ns in scales:
        print(f"\n[{name}] 区域={nz}, 时段={np_}, 场景={ns}")
        inst = generate_instance(nz, np_, ns)
        
        print("  [1] 运行 Gurobi 直接求解...")
        # 设定直接求解的时间上限，例如 300 秒
        obj_g, t_g, det_g, _ = solve_direct(inst, time_limit=300)
        print(f"      -> 结果: OBJ={obj_g:.2f}, 时间={t_g:.2f}s")
        
        print("  [2] 运行 Benders 分解算法...")
        # 设定Benders求解的时间上限，例如 300 秒
        obj_b, t_b, n_iter = solve_benders(inst, max_iter=100, time_limit=300)
        print(f"      -> 结果: OBJ={obj_b:.2f}, 时间={t_b:.2f}s, 迭代次数={n_iter}")
        
        print("  [3] 运行 均值指导启发式算法 (Heuristic)...")
        obj_h, t_h, msg_h = solve_heuristic(inst, time_limit=300)
        print(f"      -> 结果: OBJ={obj_h:.2f}, 时间={t_h:.2f}s, 状态={msg_h}")
        
        results.append({
            'scale': name,
            'nz': nz, 'np': np_, 'ns': ns,
            'gurobi_obj': obj_g, 'gurobi_time': t_g,
            'benders_obj': obj_b, 'benders_time': t_b, 'benders_iter': n_iter,
            'heuristic_obj': obj_h, 'heuristic_time': t_h
        })
        
    print("\n" + "="*110)
    print("算法对比实验结果汇总 (Gurobi vs Benders vs Heuristic)")
    print("="*110)
    print(f"{'规模':<6} {'区域':>4} {'时段':>4} {'场景':>4} | {'Gurobi OBJ':>12} {'时间':>8} | {'Benders OBJ':>12} {'时间':>8} {'Iter':>4} | {'Heuristic OBJ':>13} {'时间':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['scale']:<6} {r['nz']:>4} {r['np']:>4} {r['ns']:>4} | {r['gurobi_obj']:>12.2f} {r['gurobi_time']:>7.2f}s | {r['benders_obj']:>12.2f} {r['benders_time']:>7.2f}s {r['benders_iter']:>4} | {r['heuristic_obj']:>13.2f} {r['heuristic_time']:>7.2f}s")
        
    with open("algo_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n实验结果已保存至 algo_comparison_results.json")

if __name__ == "__main__":
    run_algo_comparison()