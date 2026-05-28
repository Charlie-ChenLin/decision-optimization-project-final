#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追加数值实验：真实港口规模、运营端政策与政策组合验证
====================================================

输出：
  - member3_policy_extension_results.json
  - member3_china_scale_case.png
  - member3_operational_policy_sensitivity.png
  - member3_policy_package_validation.png
"""

import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluation_utils import make_mean_instance, make_subinstance, solve_fixed_first_stage
from column_generation_algo import solve_column_generation
from heuristic_algo import solve_heuristic
from run_experiments import Instance, solve_direct, summarize_first_stage
from scenario_decomposition_algo import solve_scenario_decomposition


BASE_DIR = SCRIPT_DIR

CHINA_SCALE_PROFILE = {
    "reference": "上海洋山四期自动化码头公开资料口径",
    "annual_teu": 6_300_000,
    "reported_yard_blocks": 54,
    "model_zones": 12,
    "periods": 60,
    "scenes": 60,
    "cv": 0.20,
    "aggregation_note": "54个生产箱区聚合为12个模型作业区，以控制跨区移动变量规模。",
}

K_GRID = [0.0, 0.5, 1.0]
CARBON_PRICE_GRID = [0.0, 65.0, 300.0, 600.0]


def configure_plot_style():
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "Heiti TC",
        "Songti SC",
        "PingFang SC",
        "SimHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def clean_float(value):
    if value is None:
        return None
    return float(value)


def fmt_float(value, digits=1):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def safe_num(value, default=np.nan):
    return default if value is None else value


def generate_china_port_instance(
    n_zones: int = 12,
    n_periods: int = 60,
    n_scenes: int = 60,
    seed: int = 2026,
    cv: float = 0.20,
    annual_teu: int = 6_300_000,
) -> Instance:
    """生成接近中国大型自动化集装箱码头吞吐量口径的大规模算例。"""
    rng = np.random.default_rng(seed)

    Z = list(range(n_zones))
    T = list(range(1, n_periods + 1))
    N = list(range(1, (n_periods // 12) + 1))
    S = list(range(n_scenes))

    seasonal_raw = np.array([0.98, 0.78, 0.95, 0.96, 0.97, 0.98, 0.97, 1.06, 1.05, 1.04, 1.12, 1.16])
    seasonal = seasonal_raw / seasonal_raw.mean()

    weights = np.linspace(1.35, 0.75, n_zones)
    weights = weights / weights.sum()
    monthly_total = annual_teu / 12.0
    annual_growth = 0.02

    mu_det = {}
    for z in Z:
        for t in T:
            month = (t - 1) % 12
            year_mult = (1.0 + annual_growth) ** ((t - 1) // 12)
            mu_det[(z, t)] = monthly_total * weights[z] * seasonal[month] * year_mult

    sigma_ln = np.sqrt(np.log(1 + cv**2))
    h = {}
    for z in Z:
        for t in T:
            mean_val = mu_det[(z, t)]
            mu_ln = np.log(mean_val) - sigma_ln**2 / 2
            for s in S:
                h[(z, t, s)] = rng.lognormal(mu_ln, sigma_ln)

    f_init = {}
    m_cap = {}
    for z in Z:
        peak_mean = max(mu_det[(z, t)] for t in T)
        f_init[z] = int(np.ceil(2.20 * peak_mean / 3250.0))
        m_cap[z] = int(np.ceil(3.00 * peak_mean / 3250.0))

    params = {
        "k": 0.20,
        "cv": cv,
        "rho": 0.995,
        "p2": 750_000.0,
        "v2": 500_000.0,
        "d_cost": 600_000.0,
        "o1": 5.0,
        "o2": 3.0,
        "dyc_operation_penalty": 0.0,
        "eyc_operation_subsidy": 0.0,
        "g1": 3250.0,
        "beta": 0.8,
        "e1": 3.2,
        "e2": 1.4,
        "c_feqp": 0.72,
        "c_etp": 65.0,
        "tau": 2,
        "Q1": 13_500.0,
        "t_z": {z: 3 + (z % 6) for z in Z},
        "m_cap": m_cap,
        "f_init": f_init,
    }

    params["u"] = {
        (z, zp): 0.0 if z == zp else 3500.0 + 450.0 * abs(z - zp)
        for z in Z
        for zp in Z
    }
    params["theta"] = {t: 3_000_000.0 for t in T}
    params["mu"] = {t: 2_500_000.0 for t in T}
    params["q"] = {t: 1_200_000.0 for t in T}

    return Instance(n_zones, n_periods, n_scenes, Z, T, N, S, h, mu_det, params)


def clone_with_params(inst: Instance, **overrides) -> Instance:
    params = copy.deepcopy(inst.params)
    params.update(overrides)
    return Instance(
        inst.n_zones,
        inst.n_periods,
        inst.n_scenes,
        inst.Z,
        inst.T,
        inst.N,
        inst.S,
        inst.h,
        inst.mu_det,
        params,
    )


def scenario_score(inst: Instance, s: int):
    total = 0.0
    peak_ratio = 0.0
    for z in inst.Z:
        for t in inst.T:
            demand = inst.h[(z, t, s)]
            total += demand
            peak_ratio = max(peak_ratio, demand / max(1.0, inst.mu_det[(z, t)]))
    return 0.75 * total + 0.25 * peak_ratio * total


def select_representative_scenarios(inst: Instance, n_select: int):
    ranked = sorted(inst.S, key=lambda s: scenario_score(inst, s), reverse=True)
    by_total = sorted(
        inst.S,
        key=lambda s: sum(inst.h[(z, t, s)] for z in inst.Z for t in inst.T),
    )
    selected = []
    for s in ranked[: max(1, n_select // 2)]:
        selected.append(s)
    quantile_positions = np.linspace(0, len(by_total) - 1, max(1, n_select - len(selected))).round().astype(int)
    for pos in quantile_positions:
        selected.append(by_total[int(pos)])
    unique = []
    for s in selected + ranked:
        if s not in unique:
            unique.append(s)
        if len(unique) >= n_select:
            break
    return unique


def summarize_from_detail(inst, obj, solve_time, detail, fs, method, extra=None):
    b2, y, a = summarize_first_stage(fs)
    max_t = max(inst.T)
    terminal_eyc = sum(fs.get("x2", {}).get((z, max_t), 0.0) for z in inst.Z) if fs else 0.0
    cost = detail.get("cost_breakdown", {})
    row = {
        "method": method,
        "objective": clean_float(obj),
        "solve_time": clean_float(solve_time),
        "status": detail.get("status"),
        "mip_gap": clean_float(detail.get("mip_gap")),
        "n_vars": detail.get("n_vars"),
        "n_constrs": detail.get("n_constrs"),
        "eyc_purchase": clean_float(b2),
        "eyc_conversion": clean_float(y),
        "grid_renovations": clean_float(a),
        "eyc_total_investment": clean_float(b2 + y),
        "terminal_eyc_stock": clean_float(terminal_eyc),
        "expected_total_emissions": clean_float(detail.get("expected_total_emissions")),
        "expected_dyc_emissions": clean_float(detail.get("expected_dyc_emissions")),
        "expected_eyc_emissions": clean_float(detail.get("expected_eyc_emissions")),
        "expected_excess_emissions": clean_float(detail.get("expected_excess_emissions")),
        "expected_total_work": clean_float(detail.get("expected_total_work")),
        "expected_dyc_work": clean_float(detail.get("expected_dyc_work")),
        "expected_eyc_work": clean_float(detail.get("expected_eyc_work")),
        "expected_eyc_work_share": clean_float(detail.get("expected_eyc_work_share")),
        "expected_dyc_work_share": clean_float(detail.get("expected_dyc_work_share")),
        "investment_before_subsidy": clean_float(cost.get("investment_before_subsidy")),
        "investment_after_subsidy": clean_float(cost.get("investment_after_subsidy")),
        "operation_cost": clean_float(cost.get("operation")),
        "carbon_cost": clean_float(cost.get("carbon")),
        "investment_subsidy": clean_float(detail.get("investment_subsidy")),
        "operation_subsidy": clean_float(detail.get("operation_subsidy")),
        "total_policy_cost": clean_float(detail.get("total_policy_cost")),
    }
    if extra:
        row.update(extra)
    return row


def summarize_algorithm_only(obj, solve_time, method, extra=None):
    row = {
        "method": method,
        "objective": clean_float(obj),
        "solve_time": clean_float(solve_time),
    }
    if extra:
        row.update(extra)
    return row


def solve_mean_value_then_evaluate(inst: Instance, time_limit: int = 180):
    mean_inst = make_mean_instance(inst)
    obj, solve_time, detail, fs = solve_direct(mean_inst, time_limit=max(30, time_limit // 3), mip_gap=0.03)
    if fs is None:
        return summarize_from_detail(inst, obj, solve_time, detail, {}, "EV均值方案")
    eval_obj, eval_time, eval_detail = solve_fixed_first_stage(inst, fs, time_limit=max(30, time_limit - int(solve_time)))
    return summarize_from_detail(
        inst,
        eval_obj,
        solve_time + eval_time,
        eval_detail,
        fs,
        "EV均值方案",
        {"mean_model_objective": clean_float(obj)},
    )


def solve_representative_then_evaluate(
    inst: Instance,
    n_select: int = 12,
    time_limit: int = 180,
    label=None,
    allow_retry: bool = True,
):
    selected = select_representative_scenarios(inst, n_select)
    reduced_inst = make_subinstance(inst, selected)
    obj, solve_time, detail, fs = solve_direct(reduced_inst, time_limit=max(45, time_limit // 2), mip_gap=0.03)
    if fs is None:
        return summarize_from_detail(inst, obj, solve_time, detail, {}, label or f"场景生成{n_select}")
    eval_obj, eval_time, eval_detail = solve_fixed_first_stage(inst, fs, time_limit=max(45, time_limit - int(solve_time)))
    if (
        allow_retry
        and eval_obj == float("inf")
        and n_select < len(inst.S)
    ):
        next_select = min(len(inst.S), max(n_select + 8, n_select * 2))
        print(f"    全情景评估不可行，扩大代表情景数至 {next_select} 后重试")
        return solve_representative_then_evaluate(
            inst,
            n_select=next_select,
            time_limit=max(time_limit, 180),
            label=label or f"场景生成{next_select}",
            allow_retry=False,
        )
    return summarize_from_detail(
        inst,
        eval_obj,
        solve_time + eval_time,
        eval_detail,
        fs,
        label or f"场景生成{n_select}",
        {"selected_scenarios": selected, "reduced_model_objective": clean_float(obj)},
    )


def run_large_scale_case():
    print("\n[1] 中国港口真实规模算例：算法可扩展性")
    inst = generate_china_port_instance()
    rows = []

    print("  - Gurobi直接求解完整DEP（限时基准）")
    obj, solve_time, detail, fs = solve_direct(inst, time_limit=120, mip_gap=0.05)
    rows.append(summarize_from_detail(inst, obj, solve_time, detail, fs or {}, "Gurobi完整DEP"))

    print("  - EV均值方案评估")
    rows.append(solve_mean_value_then_evaluate(inst, time_limit=150))

    print("  - 场景生成式算法：12个代表情景")
    rows.append(solve_representative_then_evaluate(inst, n_select=12, time_limit=150, label="场景生成12"))

    print("  - 场景生成式算法：20个代表情景")
    rows.append(solve_representative_then_evaluate(inst, n_select=20, time_limit=180, label="场景生成20"))

    print("  - 均值指导启发式")
    h_obj, h_time, h_msg = solve_heuristic(inst, time_limit=120)
    rows.append(summarize_algorithm_only(h_obj, h_time, "均值指导启发式", {"status_message": h_msg}))

    print("  - 场景分解算法")
    sd_obj, sd_time, sd_detail = solve_scenario_decomposition(inst, max_candidates=6, time_limit=120)
    rows.append(summarize_algorithm_only(sd_obj, sd_time, "场景分解", {"details": sd_detail}))

    print("  - 列生成/场景生成算法")
    cg_obj, cg_time, cg_detail = solve_column_generation(
        inst,
        initial_scenes=4,
        add_per_iter=4,
        max_iter=4,
        time_limit=150,
    )
    rows.append(summarize_algorithm_only(cg_obj, cg_time, "列生成/场景生成", {"details": cg_detail}))

    incumbent = next((r for r in rows if r["method"] == "Gurobi完整DEP" and r["objective"] not in (None, float("inf"))), None)
    if incumbent:
        for row in rows:
            if row["objective"] not in (None, float("inf")):
                row["relative_gap_to_gurobi"] = clean_float((row["objective"] - incumbent["objective"]) / incumbent["objective"])
    return rows


def run_operational_policy_sensitivity(base_inst):
    print("\n[2] 运营端政策灵敏度：补贴k × 碳交易价格")
    rows = []
    for k in K_GRID:
        for carbon_price in CARBON_PRICE_GRID:
            inst = clone_with_params(base_inst, k=k, c_etp=carbon_price)
            row = solve_representative_then_evaluate(
                inst,
                n_select=12,
                time_limit=120,
                label="场景生成12",
            )
            row.update({"k": k, "carbon_price": carbon_price})
            rows.append(row)
            print(
                f"  k={k:.1f}, carbon={carbon_price:.0f}: "
                f"emission={fmt_float(row['expected_total_emissions'])}, "
                f"EYCshare={fmt_float(100 * row['expected_eyc_work_share'] if row['expected_eyc_work_share'] is not None else None, 2)}%, "
                f"EYC={fmt_float(row['eyc_total_investment'])}"
            )
    return rows


def run_policy_package_validation(base_inst):
    print("\n[3] 管理建议验证：政策组合实验")
    packages = [
        ("无新增政策", {"k": 0.0, "c_etp": 0.0, "c_feqp": 0.90, "eyc_operation_subsidy": 0.0}),
        ("仅投资补贴", {"k": 0.5, "c_etp": 0.0, "c_feqp": 0.90, "eyc_operation_subsidy": 0.0}),
        ("仅碳价", {"k": 0.0, "c_etp": 300.0, "c_feqp": 0.72, "eyc_operation_subsidy": 0.0}),
        ("补贴+碳价", {"k": 0.5, "c_etp": 300.0, "c_feqp": 0.72, "eyc_operation_subsidy": 0.0}),
        ("补贴+碳价+配额递减", {"k": 0.5, "c_etp": 300.0, "c_feqp": 0.55, "eyc_operation_subsidy": 0.0}),
        ("组合+绩效补贴", {"k": 0.5, "c_etp": 300.0, "c_feqp": 0.55, "eyc_operation_subsidy": 0.8}),
    ]
    rows = []
    for name, params in packages:
        inst = clone_with_params(base_inst, **params)
        row = solve_representative_then_evaluate(inst, n_select=12, time_limit=120, label="场景生成12")
        row.update({"policy_package": name, **params})
        rows.append(row)
        print(
            f"  {name}: emission={fmt_float(row['expected_total_emissions'])}, "
            f"EYCshare={fmt_float(100 * row['expected_eyc_work_share'] if row['expected_eyc_work_share'] is not None else None, 2)}%, "
            f"policy_cost={fmt_float(row['total_policy_cost'])}"
        )

    baseline = rows[0]["expected_total_emissions"]
    for row in rows:
        if baseline is None or row["expected_total_emissions"] is None:
            row["emission_reduction_vs_no_policy"] = None
            row["emission_reduction_rate_vs_no_policy"] = None
            continue
        row["emission_reduction_vs_no_policy"] = clean_float(baseline - row["expected_total_emissions"])
        row["emission_reduction_rate_vs_no_policy"] = clean_float(
            (baseline - row["expected_total_emissions"]) / baseline if baseline else 0.0
        )
    return rows


def plot_large_scale_case(rows, path):
    labels = [r["method"] for r in rows]
    times = [r["solve_time"] for r in rows]
    objectives = [safe_num(r["objective"], 0.0) / 1e7 for r in rows]
    gaps = [100 * r.get("relative_gap_to_gurobi", 0.0) if r.get("relative_gap_to_gurobi") is not None else 0.0 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    axes[0].bar(labels, times, color="#2F66B3")
    axes[0].set_title("求解时间")
    axes[0].set_ylabel("秒")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, objectives, color="#3E8A5E")
    axes[1].set_title("完整情景评估目标值")
    axes[1].set_ylabel("目标值 / 10^7")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(labels, gaps, color="#D0612C")
    axes[2].set_title("相对Gurobi限时基准差距")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("中国大型港口规模算例：12区×60期×60场景", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_operational_policy_sensitivity(rows, path):
    k_vals = sorted({r["k"] for r in rows})
    c_vals = sorted({r["carbon_price"] for r in rows})
    emission = np.zeros((len(k_vals), len(c_vals)))
    eyc_share = np.zeros_like(emission)
    eyc_invest = np.zeros_like(emission)
    for i, k in enumerate(k_vals):
        for j, c in enumerate(c_vals):
            row = next(r for r in rows if r["k"] == k and r["carbon_price"] == c)
            emission[i, j] = safe_num(row["expected_total_emissions"])
            eyc_share[i, j] = safe_num(100 * row["expected_eyc_work_share"] if row["expected_eyc_work_share"] is not None else None)
            eyc_invest[i, j] = safe_num(row["eyc_total_investment"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, data, title, fmt in [
        (axes[0], emission, "期望总排放（吨CO2）", ".0f"),
        (axes[1], eyc_share, "EYC作业占比（%）", ".1f"),
        (axes[2], eyc_invest, "EYC总投入（台）", ".1f"),
    ]:
        im = ax.imshow(data, cmap="YlGnBu", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(c_vals)))
        ax.set_xticklabels([f"{c:.0f}" for c in c_vals])
        ax.set_yticks(range(len(k_vals)))
        ax.set_yticklabels([f"{k:.1f}" for k in k_vals])
        ax.set_xlabel("碳交易价格")
        ax.set_ylabel("补贴比例 k")
        for i in range(len(k_vals)):
            for j in range(len(c_vals)):
                label = "N/A" if np.isnan(data[i, j]) else format(data[i, j], fmt)
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("运营端政策灵敏度：投资补贴与碳价协同", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_policy_package_validation(rows, path):
    labels = [r["policy_package"] for r in rows]
    emissions = [safe_num(r["expected_total_emissions"], 0.0) for r in rows]
    eyc_share = [safe_num(100 * r["expected_eyc_work_share"] if r["expected_eyc_work_share"] is not None else None, 0.0) for r in rows]
    reductions = [safe_num(r["emission_reduction_rate_vs_no_policy"] * 100 if r["emission_reduction_rate_vs_no_policy"] is not None else None, 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    colors = ["#9AA7B7", "#2F66B3", "#D0612C", "#3E8A5E", "#7C5AA6", "#C79B32"]
    axes[0].bar(labels, emissions, color=colors)
    axes[0].set_title("期望总排放")
    axes[0].set_ylabel("吨CO2")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, eyc_share, color=colors)
    axes[1].set_title("EYC作业占比")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(labels, reductions, color=colors)
    axes[2].set_title("相对无新增政策减排率")
    axes[2].set_ylabel("%")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("管理建议验证：单一政策 vs 政策组合", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plot_style()
    base_inst = generate_china_port_instance()

    large_case = run_large_scale_case()
    operational_sensitivity = run_operational_policy_sensitivity(base_inst)
    policy_packages = run_policy_package_validation(base_inst)

    results = {
        "metadata": {
            "china_scale_profile": CHINA_SCALE_PROFILE,
            "source_notes": [
                "上海港2024年集装箱吞吐量超过5150万TEU，用于说明中国头部港口总体规模。",
                "洋山四期自动化码头远期设计吞吐能力约630万TEU/年，公开资料提及54个生产箱区；本实验将其聚合为12个模型作业区。",
                "大规模算例使用统一元口径成本，以便量化投资补贴、碳价和运营绩效补贴的边际影响。",
            ],
            "k_grid": K_GRID,
            "carbon_price_grid": CARBON_PRICE_GRID,
        },
        "large_scale_case": large_case,
        "operational_policy_sensitivity": operational_sensitivity,
        "policy_package_validation": policy_packages,
    }

    json_path = BASE_DIR / "member3_policy_extension_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_large_scale_case(large_case, BASE_DIR / "member3_china_scale_case.png")
    plot_operational_policy_sensitivity(
        operational_sensitivity,
        BASE_DIR / "member3_operational_policy_sensitivity.png",
    )
    plot_policy_package_validation(
        policy_packages,
        BASE_DIR / "member3_policy_package_validation.png",
    )
    print(f"\n结果已保存至 {json_path}")


if __name__ == "__main__":
    main()
