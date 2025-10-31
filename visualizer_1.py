import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Font size configuration variables
FONT_SIZE_BASE = 20
FONT_SIZE_TITLE = 24
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 20
FONT_SIZE_LEGEND = 20

# Specific font sizes used in functions
FONT_SIZE_XLABEL = 20  # Used for x-axis labels in functions
FONT_SIZE_YLABEL = 20  # Used for y-axis labels in functions
FONT_SIZE_PLOT_TITLE = 24  # Used for plot titles in functions
FONT_SIZE_TICK_PARAMS = 20  # Used for tick parameters in functions
FONT_SIZE_LEGEND_SMALL = 20  # Used for smaller legends

# Set professional font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['font.size'] = FONT_SIZE_BASE
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
plt.rcParams['mathtext.fontset'] = 'stix'

# 确保PDF输出中字体可编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Remove top and right spines for cleaner look
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False









# =========================
# Figure 1: FL Accuracy & ASR
# =========================

def plot_attack_performance_enhanced(json_file_path, output_dir=None):
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    metrics = data['progressive_metrics']

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    rounds = metrics['rounds'][:20]
    asr = metrics['attack_asr'][:20]
    fl_acc = metrics['clean_acc'][:20]

    # Create second y-axis
    ax2 = ax1.twinx()

    # Remove top spine for ax2 as well
    ax2.spines['top'].set_visible(False)

    # Add background shading with soft colors
    # Trust building phase (light green)
    # ax1.axvspan(0.5, 6, alpha=0.15, color='#90EE90', zorder=0)
    # Attack escalation phase (light red)
    # ax1.axvspan(6, 20.5, alpha=0.15, color='#FFB6C1', zorder=0)

    # Plot Learning Accuracy (left axis)
    line1 = ax1.plot(rounds, fl_acc, 's-', color='#0052CC', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#0052CC', label='Learning Accuracy')

    ax1.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax1.set_ylabel('Global Learning Accuracy', fontsize=FONT_SIZE_YLABEL, color='#0052CC', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#0052CC', labelsize=FONT_SIZE_TICK_PARAMS)

    # Plot ASR (right axis)
    line2 = ax2.plot(rounds, asr, 'o-', color='#D72638', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#D72638', label='Attack Success Rate (ASR)')

    ax2.set_ylabel('Attack Success Rate', fontsize=FONT_SIZE_YLABEL, color='#D72638', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#D72638', labelsize=FONT_SIZE_TICK_PARAMS)

    # Highlight peak ASR
    max_asr_idx = asr.index(max(asr))
    ax2.scatter(rounds[max_asr_idx], asr[max_asr_idx], s=200,
               color='#D72638', zorder=5, edgecolors='black', linewidth=2)

    # Combined legend
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper left', frameon=True,
    #         fancybox=True, shadow=True, framealpha=0.9)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='upper left',
                    frameon=True, fancybox=False, shadow=False,
                    handlelength=1.8, handletextpad=0.5, borderpad=0.3,
                    labelspacing=0.3)

    # 透明底 + 细黑边
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)



    # Title
    # ax1.set_title('Impact of GRMP Attack on Federated Learning Performance',
                # fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)

    # Grid
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Set x-axis limits with extra space on the left
    ax1.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
    # Dynamic y-axis limits for Learning Accuracy
    acc_min = min(fl_acc)
    acc_max = max(fl_acc)
    acc_range = acc_max - acc_min
    ax1.set_ylim(acc_min - 0.05 * acc_range, acc_max + 0.4 * acc_range)  # 40% extra space on top
    
    # Dynamic y-axis limits for ASR
    asr_min = min(asr)
    asr_max = max(asr)
    asr_range = asr_max - asr_min
    ax2.set_ylim(asr_min - 0.05 * asr_range, asr_max + 0.4 * asr_range)  # 40% extra space on top

    # Ensure integer x-ticks
    ax1.set_xticks(rounds)

    # Save figure
    plt.tight_layout()

    output_path_png = output_dir / 'figure1_attack_performance.png'
    output_path_pdf = output_dir / 'figure1_attack_performance.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"Figure 1 saved to: {output_path_png}")
    plt.close()





# Figure 2: Similarity evolution (mean value for benign users)
# def plot_similarity_evolution_bars_style(json_file_path, output_dir=None):

#     # Load data
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     config = data['config']
#     results = data['results'][:20]

#     if output_dir is None:
#         output_dir = Path('./results/figures')
#     output_dir.mkdir(exist_ok=True, parents=True)

#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 8))

#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Identify attackers
#     num_clients = config['num_clients']
#     num_attackers = config['num_attackers']
#     attacker_ids = list(range(num_clients - num_attackers, num_clients))

#     # Collect data
#     rounds = []
#     thresholds = []
#     attacker_sims_by_round = []
#     benign_sims_by_round = []

#     for round_data in results:
#         rounds.append(round_data['round'])
#         thresholds.append(round_data['defense']['threshold'])

#         sims = round_data['defense']['similarities']

#         # Separate benign and attacker similarities
#         benign_sims = []
#         attacker_sims = []

#         for i, sim in enumerate(sims):
#             if i in attacker_ids:
#                 attacker_sims.append(sim)
#             else:
#                 benign_sims.append(sim)

#         benign_sims_by_round.append(benign_sims)
#         attacker_sims_by_round.append(attacker_sims)

#     # Plot defense threshold as bars
#     bar_width = 0.8
#     threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
#                            color='#2E8B57', alpha=0.3, edgecolor='#2E8B57',
#                            linewidth=1.5, label='Defense threshold', zorder=1)

#     # Plot benign users as average line
#     benign_avg_sims = []
#     for benign_sims in benign_sims_by_round:
#         if benign_sims:
#             benign_avg_sims.append(np.mean(benign_sims))
#         else:
#             benign_avg_sims.append(0)

#     # Plot benign users average line
#     ax.plot(rounds, benign_avg_sims, 'o-', color='#1D7A99',
#            linewidth=3, markersize=10, markerfacecolor='white',
#            markeredgewidth=2.5, label='Benign users (avg)',
#            zorder=4)

#     # Plot attackers as lines
#     # Reorganize attacker data by attacker ID
#     attacker_trajectories = {aid: [] for aid in range(num_attackers)}

#     for round_sims in attacker_sims_by_round:
#         for aid in range(num_attackers):
#             if aid < len(round_sims):
#                 attacker_trajectories[aid].append(round_sims[aid])
#             else:
#                 attacker_trajectories[aid].append(None)

#     # Plot each attacker's trajectory
#     attacker_colors = ['#E63946', '#F77F00']
#     for aid, trajectory in attacker_trajectories.items():
#         valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
#         valid_sims = [s for s in trajectory if s is not None]

#         if valid_sims:
#             ax.plot(valid_rounds, valid_sims, 'o-',
#                    color=attacker_colors[aid % len(attacker_colors)],
#                    linewidth=3, markersize=10, markerfacecolor='white',
#                    markeredgewidth=2.5,
#                    label=f'Attacker {aid + 1}',
#                    zorder=5)

#     # Styling
#     ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
#     ax.set_ylabel('Cosine Similarity', fontsize=FONT_SIZE_YLABEL, fontweight='bold')
#     # ax.set_title('Stealthiness of GRMP Attack: Similarity Evolution Analysis',
#                 # fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)

#     # Legend
#     ax.legend(loc='best', frameon=True, fancybox=True,
#              shadow=True, framealpha=0.9)

#     # Grid
#     ax.grid(True, alpha=0.3, linestyle='--', axis='y')
#     ax.set_axisbelow(True)

#     # Set x-axis limits with extra space on the left
#     ax.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
#     # Dynamic y-axis limits
#     all_values = []
#     all_values.extend(thresholds)
#     all_values.extend(benign_avg_sims)
#     for trajectory in attacker_trajectories.values():
#         all_values.extend([v for v in trajectory if v is not None])
    
#     y_min = min(all_values)
#     y_max = max(all_values)
#     y_range = y_max - y_min
#     ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.3 * y_range)  # 30% extra space on top

#     # Set x-ticks
#     ax.set_xticks(rounds)

#     # Save figure
#     plt.tight_layout()

#     output_path_png = output_dir / 'figure2_similarity_evolution.png'
#     output_path_pdf = output_dir / 'figure2_similarity_evolution.pdf'

#     fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
#     fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

#     print(f"Figure 2 saved to: {output_path_png}")
#     plt.close()











# ============================================================
# Figure 3: Individual benign users' similarity evolution
# ============================================================
def plot_similarity_individual_benign(json_file_path, output_dir=None):

    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results'][:20]

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Identify attackers
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    num_benign = num_clients - num_attackers
    attacker_ids = list(range(num_clients - num_attackers, num_clients))

    # Collect data
    rounds = []
    thresholds = []
    attacker_sims_by_round = []
    benign_sims_by_round = []

    for round_data in results:
        rounds.append(round_data['round'])
        thresholds.append(round_data['defense']['threshold'])

        sims = round_data['defense']['similarities']

        # Separate benign and attacker similarities
        benign_sims = []
        attacker_sims = []

        for i, sim in enumerate(sims):
            if i in attacker_ids:
                attacker_sims.append(sim)
            else:
                benign_sims.append(sim)

        benign_sims_by_round.append(benign_sims)
        attacker_sims_by_round.append(attacker_sims)

    # Plot defense threshold as bars (normal height)
    bar_width = 0.8
    threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
                           color='#2E8B57', alpha=0.3, edgecolor='#2E8B57',
                           linewidth=1.5, label='Defense Threshold', zorder=1)

    # Plot each benign user individually
    # Support up to 8 benign users
    benign_colors = [
        '#1D7A99', 
        '#5F9EA0', 
        '#6495ED',
        '#0052CC', 
        '#1E90FF',
        '#00BFFF',
        '#87CEEB',
        '#B0C4DE' 
    ]
    
    benign_markers = [
        'o',  # Circle
        's',  # Square
        '^',  # Triangle up
        'D',  # Diamond
        'v',  # Triangle down
        'p',  # Pentagon
        'h',  # Hexagon
        '*'   # Star
    ]
    
    # Organize benign data by user ID
    benign_trajectories = {uid: [] for uid in range(num_benign)}
    
    for benign_sims in benign_sims_by_round:
        for uid in range(num_benign):
            if uid < len(benign_sims):
                benign_trajectories[uid].append(benign_sims[uid])
            else:
                benign_trajectories[uid].append(None)
    
    # Plot each benign user's trajectory
    for uid, trajectory in benign_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]
        
        if valid_sims:
            ax.plot(valid_rounds, valid_sims, 
                   f'{benign_markers[uid % len(benign_markers)]}-',
                   color=benign_colors[uid % len(benign_colors)],
                   linewidth=3, markersize=10, markerfacecolor='white',
                   markeredgewidth=2.5,
                   label=f'Benign Agent {uid + 1}',
                   zorder=4)

    # Plot attackers as lines (same as Figure 2)
    attacker_trajectories = {aid: [] for aid in range(num_attackers)}

    for round_sims in attacker_sims_by_round:
        for aid in range(num_attackers):
            if aid < len(round_sims):
                attacker_trajectories[aid].append(round_sims[aid])
            else:
                attacker_trajectories[aid].append(None)

    # Plot each attacker's trajectory
    attacker_colors = ['#E63946', '#F77F00']
    for aid, trajectory in attacker_trajectories.items():
        valid_rounds = [r for r, s in zip(rounds, trajectory) if s is not None]
        valid_sims = [s for s in trajectory if s is not None]

        if valid_sims:
            ax.plot(valid_rounds, valid_sims, 'o-',
                   color=attacker_colors[aid % len(attacker_colors)],
                   linewidth=3, markersize=10, markerfacecolor='white',
                   markeredgewidth=2.5,
                   label=f'Attacker {aid + 1}',
                   zorder=5)

    # Styling
    ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=FONT_SIZE_YLABEL, fontweight='bold')
    # ax.set_title('Individual Client Similarity Evolution',
                # fontsize=FONT_SIZE_PLOT_TITLE, fontweight='bold', pad=20)
    
    # Legend
    leg = ax.legend(loc='best', ncol=2, frameon=True, fancybox=False, shadow=False,
                handlelength=1.8, handletextpad=0.5, borderpad=0.3,
                labelspacing=0.3, fontsize=FONT_SIZE_LEGEND_SMALL)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)


    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Set x-axis limits with extra space on the left
    ax.set_xlim(0, max(rounds) + 0.5)  # Changed from 0.5 to 0 for left margin
    
    # Dynamic y-axis limits
    all_values = []
    all_values.extend(thresholds)
    
    # Add all benign values
    for trajectory in benign_trajectories.values():
        all_values.extend([v for v in trajectory if v is not None])
    
    # Add all attacker values
    for trajectory in attacker_trajectories.values():
        all_values.extend([v for v in trajectory if v is not None])
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    # ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.4 * y_range)  # 40% extra space on top
    ax.set_ylim(0, 1.0)

    # Set x-ticks
    ax.set_xticks(rounds)

    # Save figure
    plt.tight_layout()

    output_path_png = output_dir / 'figure3_individual_similarity.png'
    output_path_pdf = output_dir / 'figure3_individual_similarity.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"Figure 3 saved to: {output_path_png}")
    plt.close()





# --- Global palettes (shared by Fig.3/4/5) ---
BENIGN_COLORS  = ['#1D7A99', '#5F9EA0', '#6495ED', '#0052CC', '#1E90FF', '#00BFFF', '#87CEEB', '#B0C4DE']
BENIGN_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
ATTACKER_COLORS  = ['#E63946', '#F77F00']  # red, orange (match Fig.3)
ATTACKER_MARKER  = 'o'

def plot_client_accuracy_evolution(json_file_path,
                                   which='local_train',
                                   client_ids=None,
                                   max_rounds=20,
                                   output_dir=None):
    """
    绘制指定客户端在多个通信轮上的 accuracy 变化。
    which: 'local_train' 或 'clean_test'
    client_ids: 要画的客户端ID列表（默认：优先包含攻击者，再补足良性到 6 个）
    """
    assert which in ('local_train', 'clean_test'), "which must be 'local_train' or 'clean_test'"

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])[:max_rounds]
    config  = data.get('config', {})

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1) 汇总每个 client_id 的角色（优先用 JSON 中的 role；否则回退到 config 规则） ----
    client_roles = {}  # {cid: 'benign'|'attacker'}
    for rd in results:
        for m in rd.get('local_client_metrics', []):
            cid  = m.get('client_id')
            role = m.get('role')
            if cid is not None and role in ('benign', 'attacker') and cid not in client_roles:
                client_roles[cid] = role

    # 如果还没识别出角色，按常见配置：最大 ID 的若干个为攻击者
    if not client_roles and 'num_clients' in config and 'num_attackers' in config:
        num_clients   = config['num_clients']
        num_attackers = config['num_attackers']
        attacker_ids_cfg = list(range(num_clients - num_attackers, num_clients))
        # 先收集出现过的 client_id
        seen_cids = set()
        for rd in results:
            for m in rd.get('local_client_metrics', []):
                if 'client_id' in m:
                    seen_cids.add(m['client_id'])
        for cid in seen_cids:
            client_roles[cid] = 'attacker' if cid in attacker_ids_cfg else 'benign'

    # 若仍为空则无法绘图
    if not client_roles:
        print("No client roles found (local_client_metrics/role missing). Skip plotting client accuracy.")
        return

    attackers = sorted([cid for cid, r in client_roles.items() if r == 'attacker'])
    benigns   = sorted([cid for cid, r in client_roles.items() if r == 'benign'])
    
    attacker_index = {cid: i for i, cid in enumerate(attackers)}   # 0→Attacker 1
    benign_index   = {cid: i for i, cid in enumerate(benigns)}      # 0→Benign user 1

    # ---- 2) 选择要绘制的 client_ids（优先攻击者，确保两条攻击者曲线出现并固定配色）----
    if client_ids is None:
        ordered = attackers + benigns
        client_ids = ordered[:6]
    else:
        # 仅保留角色已知的且存在的
        client_ids = [cid for cid in client_ids if cid in client_roles][:6]

    # ---- 3) 绘图 ----
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    rounds = [rd.get('round', idx + 1) for idx, rd in enumerate(results)]

    benign_color_ptr  = 0
    benign_marker_ptr = 0

    for cid in client_ids:
        acc_seq = []
        for rd in results:
            acc_val = None
            for m in rd.get('local_client_metrics', []):
                if m.get('client_id') == cid:
                    block = m.get(which, None)  # {'acc','loss','num_samples'}
                    if isinstance(block, dict) and 'acc' in block:
                        acc_val = block['acc']
                    break
            acc_seq.append(acc_val)

        valid_rounds = [r for r, a in zip(rounds, acc_seq) if a is not None]
        valid_accs   = [a for a in acc_seq if a is not None]
        if not valid_accs:
            continue

        role = client_roles.get(cid, 'benign')
        if role == 'attacker' and cid in attacker_index:
            col = ATTACKER_COLORS[attacker_index[cid] % len(ATTACKER_COLORS)]
            mrk = ATTACKER_MARKER
            lbl = f'Attacker {attacker_index[cid]+1}'
        else:
            col = BENIGN_COLORS[benign_color_ptr % len(BENIGN_COLORS)]
            mrk = BENIGN_MARKERS[benign_marker_ptr % len(BENIGN_MARKERS)]
            lbl = f'Benign Agent {benign_index.get(cid, benign_color_ptr) + 1}'
            benign_color_ptr  += 1
            benign_marker_ptr += 1

        ax.plot(valid_rounds, valid_accs,
                f'{mrk}-', color=col,
                linewidth=3, markersize=10, markerfacecolor='white',
                markeredgewidth=2.0, label=lbl, zorder=3)

    ax.set_xlabel('Communication Round', fontsize=FONT_SIZE_XLABEL, fontweight='bold')
    ax.set_ylabel('Local Training Accuracy' if which == 'local_train' else 'Local Testing Accuracy',
                fontsize=FONT_SIZE_YLABEL, fontweight='bold')

    leg = ax.legend(loc='best',
                    frameon=True, fancybox=False, shadow=False,
                    handlelength=1.8, handletextpad=0.5, borderpad=0.3,
                    labelspacing=0.3, fontsize=FONT_SIZE_LEGEND_SMALL)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(rounds) + 0.5)
    ax.set_ylim(0.0, 1.0)          # 与 Figure 3 一致的固定刻度
    ax.set_xticks(rounds)
    plt.tight_layout()

    suffix = 'local' if which == 'local_train' else 'clean'
    out_png = output_dir / f'figure4_client_accuracy_{suffix}.png'
    out_pdf = output_dir / f'figure4_client_accuracy_{suffix}.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    print(f"Figure 4/5 ({suffix}) saved to: {out_png}")
    plt.close()






# Function to generate all three figures for the paper
def generate_paper_figures(json_file_path):

    print("Generating figures ...")

    # Generate Figure 1: Attack Performance
    plot_attack_performance_enhanced(json_file_path)

    # Generate Figure 2: Similarity Evolution (with average)
    # plot_similarity_evolution_bars_style(json_file_path)
    
    # Generate Figure 3: Individual Similarity Evolution
    plot_similarity_individual_benign(json_file_path)

    # Figure 4: 6 clients - Local Train Accuracy
    plot_client_accuracy_evolution(json_file_path, which='local_train', client_ids=None, max_rounds=20)

    # Figure 5: 6 clients - Clean Test Accuracy
    plot_client_accuracy_evolution(json_file_path, which='clean_test', client_ids=None, max_rounds=20)


    print("\nAll figures generated successfully!")
    print("Files saved in: ./results/figures/")




if __name__ == "__main__":
    # Path to your JSON file
    json_file = './results/progressive_grmp_progressive_semantic_poisoning.json'

    # Generate all three figures
    generate_paper_figures(json_file)