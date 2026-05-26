#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化模型结果可视化
读取 sol_beta08.pkl 生成决策变量时序图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ==================== 配置 ====================
# Windows 用户请改为系统自带中文字体路径，例如：
font_path = 'C:/Windows/Fonts/simhei.ttf'
# font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = FontProperties(fname=font_path, size=10)
font_prop_title = FontProperties(fname=font_path, size=11)
font_prop_large = FontProperties(fname=font_path, size=13)
# =============================================

def load_solution(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def get_series(sol, var_name, z):
    return np.array([sol[var_name][(z, t)] for t in sol['T']])

def plot_page1(sol, save_path='vis_page1.png'):
    Z, T = sol['Z'], sol['T']
    zone_names = {0: '区域A', 1: '区域B', 2: '区域C'}

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'优化模型决策变量时序可视化 (beta={sol["beta"]}, Q1=8.0)\n'
                 f'OBJ={sol["ObjVal"]:.1f} | C_inv={sol["C_inv"]:.1f} | C_op={sol["C_op"]:.1f} | C_carbon={sol["C_carbon"]:.1f}',
                 fontproperties=font_prop_large, fontweight='bold')

    for idx, z in enumerate(Z):
        ax = axes[idx, 0]
        x1v, x2v = get_series(sol, 'x1', z), get_series(sol, 'x2', z)
        ax.fill_between(T, x1v, alpha=0.5, label='$x_1$ (类型1存量)', color='#3498db')
        ax.fill_between(T, x2v, alpha=0.5, label='$x_2$ (类型2存量)', color='#e74c3c')
        ax.plot(T, x1v, color='#2980b9', linewidth=1.5)
        ax.plot(T, x2v, color='#c0392b', linewidth=1.5)
        ax.set_title(f'{zone_names[z]}：设备存量时序', fontproperties=font_prop_title)
        ax.set_xlabel('时段 t', fontproperties=font_prop)
        ax.set_ylabel('设备数量', fontproperties=font_prop)
        ax.legend(prop=font_prop, loc='upper right')
        ax.set_xlim(1, 24); ax.grid(True, alpha=0.3)

    for idx, z in enumerate(Z):
        ax = axes[idx, 1]
        w1v, w2v = get_series(sol, 'w1', z), get_series(sol, 'w2', z)
        hv = np.array([sol['h'][(z, t)] for t in T])
        ax.fill_between(T, w1v, alpha=0.6, label='$w_1$ (类型1作业)', color='#2ecc71')
        ax.fill_between(T, w1v+w2v, w1v, alpha=0.6, label='$w_2$ (类型2作业)', color='#f39c12')
        ax.plot(T, hv, 'k--', linewidth=1, label='$h$ (总需求)', alpha=0.7)
        ax.set_title(f'{zone_names[z]}：作业分配时序', fontproperties=font_prop_title)
        ax.set_xlabel('时段 t', fontproperties=font_prop)
        ax.set_ylabel('作业量', fontproperties=font_prop)
        ax.legend(prop=font_prop, loc='upper right')
        ax.set_xlim(1, 24); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_page2(sol, save_path='vis_page2.png'):
    Z, T, N = sol['Z'], sol['T'], sol['N']
    zone_names = {0: '区域A', 1: '区域B', 2: '区域C'}
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'决策变量时序可视化 第2页 (beta={sol["beta"]}, Q1=8.0)',
                 fontproperties=font_prop_large, fontweight='bold')

    # 改造量
    ax = axes2[0, 0]
    for z in Z:
        yv = get_series(sol, 'y', z)
        if np.any(yv > 0.01):
            ax.bar(np.array(T)+z*0.25-0.25, yv, width=0.25, label=zone_names[z], alpha=0.8)
    ax.set_title('改造量 $y_{zt}$ 时序', fontproperties=font_prop_title)
    ax.set_xlabel('时段 t', fontproperties=font_prop)
    ax.set_ylabel('改造量', fontproperties=font_prop)
    ax.legend(prop=font_prop); ax.grid(True, alpha=0.3, axis='y')

    # 购买量
    ax = axes2[0, 1]
    has_buy = False
    for z in Z:
        b2v = get_series(sol, 'b2', z)
        if np.any(b2v > 0.01):
            has_buy = True
            ax.bar(np.array(T)+z*0.25-0.25, b2v, width=0.25, label=zone_names[z], alpha=0.8, color=colors[z])
    if not has_buy:
        ax.text(0.5, 0.5, '未购买设备\n(b2 = 0)', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='gray', fontproperties=font_prop_title)
    ax.set_title('购买量 $b_{2zt}$ 时序', fontproperties=font_prop_title)
    ax.set_xlabel('时段 t', fontproperties=font_prop)
    ax.set_ylabel('购买量', fontproperties=font_prop)
    if has_buy: ax.legend(prop=font_prop)
    ax.grid(True, alpha=0.3, axis='y')

    # 电网改造指示
    ax = axes2[1, 0]
    for idx, z in enumerate(Z):
        av, rv = get_series(sol, 'a', z), get_series(sol, 'r', z)
        ta = [t for t in T if av[t-1] > 0.5]
        tr = [t for t in T if rv[t-1] > 0.5]
        ypos_a, ypos_r = 3-idx, 2.7-idx
        if ta: ax.scatter(ta, [ypos_a]*len(ta), marker='o', s=100, color=colors[idx], zorder=3)
        if tr: ax.scatter(tr, [ypos_r]*len(tr), marker='x', s=80, color=colors[idx], zorder=3)
    ax.set_title('电网改造指示 $a_{zt}$ (圆点) 与 $r_{zt}$ (叉号)', fontproperties=font_prop_title)
    ax.set_xlabel('时段 t', fontproperties=font_prop)
    ax.set_yticks([1, 1.3, 2, 2.3, 3, 3.3])
    ax.set_yticklabels(['rA', 'aA', 'rB', 'aB', 'rC', 'aC'])
    ax.grid(True, alpha=0.3, axis='x')

    # 碳排放
    ax = axes2[1, 1]
    periods = list(N)
    Ev = [sol['E'][n] for n in N]
    dEv = [sol['dE'][n] for n in N]
    x_pos = np.arange(len(periods))
    b1 = ax.bar(x_pos-0.2, Ev, 0.4, label='$E_n$ (周期排放)', color='#e74c3c', alpha=0.8)
    b2 = ax.bar(x_pos+0.2, dEv, 0.4, label='$\Delta E_n$ (超额排放)', color='#f39c12', alpha=0.8)
    ax.axhline(y=8, color='gray', linestyle='--', alpha=0.5, label='Q1=8 (免费额度)')
    ax.set_title('碳排放量与超额排放', fontproperties=font_prop_title)
    ax.set_xlabel('周期 n', fontproperties=font_prop)
    ax.set_ylabel('排放量', fontproperties=font_prop)
    ax.set_xticks(x_pos); ax.set_xticklabels([f'n={n}' for n in periods])
    ax.legend(prop=font_prop); ax.grid(True, alpha=0.3, axis='y')
    for bar in b1+b2:
        h = bar.get_height()
        if h > 0.1:
            ax.annotate(f'{h:.1f}', xy=(bar.get_x()+bar.get_width()/2, h),
                       xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    sol = load_solution("sol_beta08.pkl")
    plot_page1(sol)
    plot_page2(sol)
    print("Done!")
