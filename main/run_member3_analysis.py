#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值实验与政策仿真
================================
输出：
  - member3_results.json
  - member3_uncertainty_impact.png
  - member3_subsidy_sensitivity.png
  - member3_cv_marginal_effect.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_experiments import (
    compute_eev,
    generate_instance,
    solve_direct,
    summarize_first_stage,
)


BASE_DIR = Path(__file__).resolve().parent
SCALES = [
    ("小规模", 3, 12, 5, 120),
    ("中规模", 3, 24, 20, 120),
    ("大规模", 5, 36, 50, 300),
]
K_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
CV_GRID = [0.05, 0.15, 0.30]


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


def summarize_solution(inst, obj, solve_time, details, first_stage):
    b2, y, a = summarize_first_stage(first_stage)
    max_t = max(inst.T)
    terminal_eyc = sum(first_stage.get("x2", {}).get((z, max_t), 0.0) for z in inst.Z)
    cost = details.get("cost_breakdown", {})
    return {
        "objective": clean_float(obj),
        "solve_time": clean_float(solve_time),
        "mip_gap": clean_float(details.get("mip_gap")),
        "n_vars": details.get("n_vars"),
        "n_constrs": details.get("n_constrs"),
        "eyc_purchase": clean_float(b2),
        "eyc_conversion": clean_float(y),
        "grid_renovations": clean_float(a),
        "eyc_total_investment": clean_float(b2 + y),
        "terminal_eyc_stock": clean_float(terminal_eyc),
        "expected_total_emissions": clean_float(details.get("expected_total_emissions")),
        "expected_excess_emissions": clean_float(details.get("expected_excess_emissions")),
        "government_subsidy": clean_float(details.get("government_subsidy")),
        "investment_before_subsidy": clean_float(cost.get("investment_before_subsidy")),
        "investment_after_subsidy": clean_float(cost.get("investment_after_subsidy")),
        "operation_cost": clean_float(cost.get("operation")),
        "carbon_cost": clean_float(cost.get("carbon")),
    }


def run_uncertainty_impact():
    rows = []
    print("\n[1] 不确定性影响实验：DEP vs EV/EEV")
    for scale, nz, nt, ns, time_limit in SCALES:
        print(f"  - {scale}: Z={nz}, T={nt}, S={ns}")
        inst = generate_instance(nz, nt, ns, seed=42, cv=0.15)
        dep_obj, dep_time, dep_detail, dep_fs = solve_direct(inst, time_limit=time_limit)
        eev, ev_fs = compute_eev(inst, time_limit=time_limit)
        dep = summarize_solution(inst, dep_obj, dep_time, dep_detail, dep_fs)
        ev_b2, ev_y, ev_a = summarize_first_stage(ev_fs)
        max_t = max(inst.T)
        ev_terminal_eyc = sum(ev_fs.get("x2", {}).get((z, max_t), 0.0) for z in inst.Z)
        row = {
            "scale": scale,
            "zones": nz,
            "periods": nt,
            "scenes": ns,
            "dep": dep,
            "ev": {
                "eev": clean_float(eev),
                "eyc_purchase": clean_float(ev_b2),
                "eyc_conversion": clean_float(ev_y),
                "grid_renovations": clean_float(ev_a),
                "eyc_total_investment": clean_float(ev_b2 + ev_y),
                "terminal_eyc_stock": clean_float(ev_terminal_eyc),
            },
            "vss": clean_float(eev - dep_obj),
            "vss_rate": clean_float((eev - dep_obj) / eev if eev else None),
            "eyc_buffer_increase": clean_float((dep["eyc_total_investment"] or 0) - (ev_b2 + ev_y)),
            "terminal_eyc_increase": clean_float((dep["terminal_eyc_stock"] or 0) - ev_terminal_eyc),
        }
        rows.append(row)
        print(
            f"    DEP_EYC={row['dep']['eyc_total_investment']:.2f}, "
            f"EV_EYC={row['ev']['eyc_total_investment']:.2f}, "
            f"VSS={row['vss']:.2f}"
        )
    return rows


def run_subsidy_sensitivity(cv=0.15):
    rows = []
    print(f"\n[2] 补贴灵敏度实验：中规模，CV={cv}")
    for k in K_GRID:
        inst = generate_instance(3, 24, 20, seed=42, cv=cv)
        inst.params["k"] = k
        obj, solve_time, details, fs = solve_direct(inst, time_limit=120)
        row = {
            "k": clean_float(k),
            **summarize_solution(inst, obj, solve_time, details, fs),
        }
        rows.append(row)
        print(
            f"  k={k:.1f}: OBJ={row['objective']:.2f}, "
            f"Emissions={row['expected_total_emissions']:.2f}, "
            f"EYC={row['eyc_total_investment']:.2f}"
        )
    return rows


def annotate_policy_threshold(rows):
    base = rows[0]["expected_total_emissions"]
    threshold = None
    for row in rows[1:]:
        reduction_rate = (base - row["expected_total_emissions"]) / base if base else 0.0
        row["emission_reduction_vs_k0"] = clean_float(base - row["expected_total_emissions"])
        row["emission_reduction_rate_vs_k0"] = clean_float(reduction_rate)
        row["marginal_reduction_per_subsidy"] = clean_float(
            row["emission_reduction_vs_k0"] / row["government_subsidy"]
            if row["government_subsidy"]
            else 0.0
        )
        if threshold is None and reduction_rate >= 0.005:
            threshold = row["k"]
    rows[0]["emission_reduction_vs_k0"] = 0.0
    rows[0]["emission_reduction_rate_vs_k0"] = 0.0
    rows[0]["marginal_reduction_per_subsidy"] = 0.0
    if threshold is None:
        message = "当前参数下补贴主要降低企业成本，未形成显著减排临界点。"
    else:
        message = f"当 k 达到 {threshold:.1f} 时，总排放相对 k=0 下降超过 0.5%。"
    return {"threshold_k": clean_float(threshold), "message": message}


def run_cv_extension():
    cv_results = {}
    print("\n[3] 不同不确定性水平下的补贴边际效用")
    for cv in CV_GRID:
        rows = run_subsidy_sensitivity(cv=cv)
        threshold = annotate_policy_threshold(rows)
        cv_results[f"{cv:.2f}"] = {
            "rows": rows,
            "policy_threshold": threshold,
        }
    return cv_results


def plot_uncertainty_impact(rows, path):
    labels = [r["scale"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    dep_eyc = [r["dep"]["eyc_total_investment"] for r in rows]
    ev_eyc = [r["ev"]["eyc_total_investment"] for r in rows]
    vss = [r["vss"] / 10000 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(x - width / 2, dep_eyc, width, label="随机规划 DEP", color="#2F66B3")
    axes[0].bar(x + width / 2, ev_eyc, width, label="均值模型 EV", color="#9AA7B7")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("EYC总投入（购买+改造，台）")
    axes[0].set_title("需求不确定性促使企业增加EYC缓冲")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(labels, vss, marker="o", linewidth=2.2, color="#D0612C")
    axes[1].set_ylabel("VSS（模型成本单位/10^4）")
    axes[1].set_title("考虑不确定性的经济价值")
    axes[1].grid(alpha=0.25)
    for i, value in enumerate(vss):
        axes[1].annotate(f"{value:.2f}", (i, value), textcoords="offset points", xytext=(0, 7), ha="center")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_subsidy_sensitivity(rows, threshold, path):
    k = [r["k"] for r in rows]
    emissions = [r["expected_total_emissions"] for r in rows]
    obj = [r["objective"] / 10000 for r in rows]
    subsidy = [r["government_subsidy"] for r in rows]
    eyc = [r["eyc_total_investment"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    axes[0, 0].plot(k, emissions, marker="o", color="#2F66B3")
    axes[0, 0].set_title("总排放量")
    axes[0, 0].set_ylabel("期望总排放（吨CO2）")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(k, obj, marker="o", color="#3E8A5E")
    axes[0, 1].set_title("企业期望总成本")
    axes[0, 1].set_ylabel("目标值（模型成本单位/10^4）")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].bar(k, subsidy, width=0.07, color="#D0612C")
    axes[1, 0].set_title("政府补贴支出")
    axes[1, 0].set_xlabel("补贴比例 k")
    axes[1, 0].set_ylabel("补贴支出（模型成本单位）")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].plot(k, eyc, marker="o", color="#7C5AA6")
    axes[1, 1].set_title("EYC总投入")
    axes[1, 1].set_xlabel("补贴比例 k")
    axes[1, 1].set_ylabel("EYC总投入（台）")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(threshold["message"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cv_marginal_effect(cv_results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = ["#2F66B3", "#D0612C", "#3E8A5E"]
    for color, cv_key in zip(colors, sorted(cv_results.keys())):
        rows = cv_results[cv_key]["rows"]
        k = [r["k"] for r in rows]
        emissions = [r["expected_total_emissions"] for r in rows]
        marginal = [r["marginal_reduction_per_subsidy"] for r in rows]
        axes[0].plot(k, emissions, marker="o", linewidth=2, color=color, label=f"CV={cv_key}")
        axes[1].plot(k, marginal, marker="o", linewidth=2, color=color, label=f"CV={cv_key}")

    axes[0].set_title("不同需求波动下的排放响应")
    axes[0].set_xlabel("补贴比例 k")
    axes[0].set_ylabel("期望总排放（吨CO2）")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("补贴边际减排效用")
    axes[1].set_xlabel("补贴比例 k")
    axes[1].set_ylabel("减排量 / 补贴额")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plot_style()
    uncertainty = run_uncertainty_impact()
    subsidy = run_subsidy_sensitivity(cv=0.15)
    subsidy_threshold = annotate_policy_threshold(subsidy)
    cv_results = run_cv_extension()

    results = {
        "metadata": {
            "main_instance": {"zones": 3, "periods": 24, "scenes": 20, "seed": 42},
            "k_grid": K_GRID,
            "cv_grid": CV_GRID,
            "policy_threshold_rule": "最小k使总排放相对k=0下降至少0.5%",
        },
        "uncertainty_impact": uncertainty,
        "subsidy_sensitivity": {
            "rows": subsidy,
            "policy_threshold": subsidy_threshold,
        },
        "cv_marginal_effect": cv_results,
    }

    json_path = BASE_DIR / "member3_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    plot_uncertainty_impact(uncertainty, BASE_DIR / "member3_uncertainty_impact.png")
    plot_subsidy_sensitivity(subsidy, subsidy_threshold, BASE_DIR / "member3_subsidy_sensitivity.png")
    plot_cv_marginal_effect(cv_results, BASE_DIR / "member3_cv_marginal_effect.png")
    print(f"\n结果已保存至 {json_path}")


if __name__ == "__main__":
    main()
