import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# (导入 AutoMinorLocator 保持不变)
from matplotlib.ticker import AutoMinorLocator
# (Patch 暂时不需要)

# (字体和 RCParams 设置保持不变)
FONT_SIZE_BASE = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 20
FONT_SIZE_LEGEND = 20
FONT_SIZE_XLABEL = 20
FONT_SIZE_YLABEL = 20
FONT_SIZE_PLOT_TITLE = 24
FONT_SIZE_TICK_PARAMS = 20
FONT_SIZE_LEGEND_SMALL = 20
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['font.size'] = FONT_SIZE_BASE
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# (全局 spines 设置保持关闭，我们在函数内控制)

# (全局配色方案保持不变)
ACC_COLOR = '#01579B'   # 聚合图 - 良性
ASR_COLOR = '#D72638'   # 聚合图 - 恶意
DEFENSE_COLOR = '#00695C' # 防御阈值

# =========================
# Figure 1: FL Accuracy & ASR
# (此函数保持不变, 严格使用 蓝/红 聚合配色)
# =========================
def plot_attack_performance_enhanced(json_file_path, output_dir=None):
    # (数据加载)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    metrics = data['progressive_metrics']
    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # (作图设置)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    rounds = metrics['rounds'][:20]
    asr = metrics['attack_asr'][:20]
    fl_acc = metrics['clean_acc'][:20]
    ax2 = ax1.twinx()

    # (应用全边框)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    # (应用高级网格)
    ax1.grid(True, which='major', alpha=0.3, linestyle='--', axis='both', zorder=0)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':', axis='y', zorder=0)

    # (Plot Learning Accuracy - 统一蓝色)
    line1 = ax1.plot(rounds, fl_acc, 's-', color=ACC_COLOR, linewidth=3,
                    markersize=10, 
                    markerfacecolor=ACC_COLOR,  # 实心填充
                    markeredgewidth=1.0,        # 细黑边
                    markeredgecolor='black',    # 细黑边
                    label='Learning Accuracy (Mean)')

    ax1.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax1.set_ylabel('Global Learning Accuracy', fontsize=FONT_SIZE_YLABEL, color=ACC_COLOR, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=ACC_COLOR, labelsize=FONT_SIZE_TICK_PARAMS)

    # (Plot ASR - 统一红色)
    line2 = ax2.plot(rounds, asr, 'o-', color=ASR_COLOR, linewidth=3,
                    markersize=10, 
                    markerfacecolor=ASR_COLOR,  # 实心填充
                    markeredgewidth=1.0,        # 细黑边
                    markeredgecolor='black',    # 细黑边
                    label='ASR (Mean)')

    ax2.set_ylabel('Attack Success Rate', fontsize=FONT_SIZE_YLABEL, color=ASR_COLOR, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=ASR_COLOR, labelsize=FONT_SIZE_TICK_PARAMS)

    # (高亮峰值)
    max_asr_idx = asr.index(max(asr))
    ax2.scatter(rounds[max_asr_idx], asr[max_asr_idx], s=200,
               color=ASR_COLOR, zorder=5, edgecolors='black', linewidth=2)

    # (图例)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='best',
                    frameon=True, fancybox=False, shadow=False,
                    handlelength=1.8, handletextpad=0.5, borderpad=0.3,
                    labelspacing=0.3)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)

    # (坐标轴范围)
    ax1.set_xlim(0, max(rounds) + 0.5)
    acc_min = min(fl_acc)
    acc_max = max(fl_acc)
    acc_range = acc_max - acc_min
    ax1.set_ylim(acc_min - 0.05 * acc_range, acc_max + 0.4 * acc_range)
    asr_min = min(asr)
    asr_max = max(asr)
    asr_range = asr_max - asr_min
    ax2.set_ylim(asr_min - 0.05 * asr_range, asr_max + 0.4 * asr_range)

    # (X轴刻度)
    ax1.set_xticks(rounds)

    # (保存)
    plt.tight_layout()
    output_path_png = output_dir / 'figure1_attack_performance.png'
    output_path_pdf = output_dir / 'figure1_attack_performance.pdf'
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Figure 1 saved to: {output_path_png}")
    plt.close()


# ============================================================
# Figure 3: Individual benign users' similarity evolution
# (已按“黄金标准”风格修改 + 恢复多种配色)
# ============================================================
def plot_similarity_individual_benign(json_file_path, output_dir=None):
    # (数据加载保持不变)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    config = data['config']
    results = data['results'][:20]
    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # (作图设置保持不变)
    fig, ax = plt.subplots(figsize=(12, 8))

    # (应用全边框 - 保持不变)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # (数据收集部分保持不变)
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    num_benign = num_clients - num_attackers
    attacker_ids = list(range(num_clients - num_attackers, num_clients))
    rounds = []
    thresholds = []
    attacker_sims_by_round = []
    benign_sims_by_round = []
    for round_data in results:
        rounds.append(round_data['round'])
        thresholds.append(round_data['defense']['threshold'])
        sims = round_data['defense']['similarities']
        benign_sims = []
        attacker_sims = []
        for i, sim in enumerate(sims):
            if i in attacker_ids:
                attacker_sims.append(sim)
            else:
                benign_sims.append(sim)
        benign_sims_by_round.append(benign_sims)
        attacker_sims_by_round.append(attacker_sims)
        
    # (应用高级网格 - 保持不变)
    ax.grid(True, which='major', alpha=0.3, linestyle='--', axis='both', zorder=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', axis='y', zorder=0)

    # (Plot defense threshold - 保持不变)
    bar_width = 0.8
    threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
                           color=DEFENSE_COLOR, alpha=0.3, edgecolor=DEFENSE_COLOR,
                           linewidth=1.5, label='Defense Threshold', zorder=1)

    # Plot each benign user individually
    # ===> 修改点 1：恢复良性配色列表 <===
    benign_colors = [
        '#1D7A99', '#5F9EA0', '#6495ED', '#0052CC', 
        '#1E90FF', '#00BFFF', '#87CEEB', '#B0C4DE' 
    ]
    benign_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    benign_trajectories = {uid: [] for uid in range(num_benign)}
    for benign_sims in benign_sims_by_round:
        for uid in range(num_benign):
            if uid < len(benign_sims):
                benign_trajectories[uid].append(benign_sims[uid])
            else:
                benign_trajectories[uid].append(None)
    
    # ===> 修改点 2：应用多色 + 实心标记 <===
    for uid, trajectory in benign_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]
        
        if valid_sims:
            color = benign_colors[uid % len(benign_colors)] # 获取个体颜色
            ax.plot(valid_rounds, valid_sims, 
                   f'{benign_markers[uid % len(benign_markers)]}-',
                   color=color, # 应用个体颜色
                   linewidth=3, markersize=10, 
                   markerfacecolor=color, # 实心填充 (用个体颜色)
                   markeredgewidth=1.0,     # 保留细黑边
                   markeredgecolor='black', # 保留细黑边
                   label=f'Benign Agent {uid + 1}',
                   zorder=4)

    # Plot attackers as lines
    # ===> 修改点 3：恢复恶意配色列表 <===
    attacker_trajectories = {aid: [] for aid in range(num_attackers)}
    attacker_markers = ['o', 's', '^', 'D']
    attacker_colors = ['#E63946', '#F77F00'] # 使用您原版中的红色和橙色
    
    for round_sims in attacker_sims_by_round:
        for aid in range(num_attackers):
            if aid < len(round_sims):
                attacker_trajectories[aid].append(round_sims[aid])
            else:
                attacker_trajectories[aid].append(None)

    # ===> 修改点 4：应用多色 + 实心标记 <===
    for aid, trajectory in attacker_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]

        if valid_sims:
            color = attacker_colors[aid % len(attacker_colors)] # 获取个体颜色
            ax.plot(valid_rounds, valid_sims, 
                   f'{attacker_markers[aid % len(attacker_markers)]}-',
                   color=color, # 应用个体颜色
                   linewidth=3, markersize=10, 
                   markerfacecolor=color, # 实心填充 (用个体颜色)
                   markeredgewidth=1.0,       # 保留细黑边
                   markeredgecolor='black',   # 保留细黑边
                   label=f'Attacker {aid + 1}',
                   zorder=5)

    # (Styling 和 Legend 保持不变)
    ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=FONT_SIZE_YLABEL, fontweight='bold')
    leg = ax.legend(loc='best', ncol=2, frameon=True, fancybox=False, shadow=False,
                handlelength=1.8, handletextpad=0.5, borderpad=0.3,
                labelspacing=0.3, fontsize=FONT_SIZE_LEGEND_SMALL)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)

    # (坐标轴范围保持不变)
    ax.set_xlim(0, max(rounds) + 0.5)
    ax.set_ylim(0, 1.0) 
    ax.set_xticks(rounds)

    # (保存部分保持不变)
    plt.tight_layout()
    output_path_png = output_dir / 'figure3_individual_similarity.png'
    output_path_pdf = output_dir / 'figure3_individual_similarity.pdf'
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Figure 3 saved to: {output_path_png}")
    plt.close()


# (主函数保持不变)
def generate_paper_figures(json_file_path):
    print("Generating figures ...")
    plot_attack_performance_enhanced(json_file_path)
    plot_similarity_individual_benign(json_file_path)
    print("\nAll figures generated successfully!")
    # (修正了路径打印)
    output_dir = Path(json_file).resolve().parent / 'figures'
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    # Path to your JSON file
    json_file = './results/progressive_grmp_progressive_semantic_poisoning.json'

    # Generate all three figures
    generate_paper_figures(json_file)