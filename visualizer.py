# visualizer.py — IEEE minimal style
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


# =========================
# === IEEE Style Helpers ==
# =========================
def use_ieee_minimal(font_family='Arial', column='one'):
    """
    Apply an IEEE-like minimal style to Matplotlib.
    column: 'one' -> single-column (3.5 in width), 'two' -> double-column (7.16 in).
    """
    width_in = 3.5 if column == 'one' else 7.16
    height_in = 2.4 if column == 'one' else 3.0

    mpl.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': [font_family, 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',

        # Embed TrueType (editable in PDF)
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # Sizes (pt)
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,     # 若使用标题
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,

        # Lines/Markers
        'lines.linewidth': 1.4,
        'lines.markersize': 5,

        # Axes/Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,

        # Legends
        'legend.frameon': False,
    })
    return (width_in, height_in)


def percent_formatter():
    """Format 0..1 values as percentages."""
    return FuncFormatter(lambda y, _: f"{y*100:.0f}%")


def set_minimal_grid(ax, axis='y'):
    ax.grid(True, axis=axis, linestyle='--', linewidth=0.6, alpha=0.2)
    ax.set_axisbelow(True)


def safe_ylim(ax, y_values, top_extra=0.25, bottom_extra=0.05, min_span=1e-6):
    """Robust y-limits even if the series is constant."""
    y_values = np.asarray(y_values, dtype=float)
    if y_values.size == 0:
        return
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_range = y_max - y_min
    if y_range < min_span:
        pad = max(abs(y_min) * 0.05, 1e-3)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(y_min - bottom_extra * y_range, y_max + top_extra * y_range)


def add_ieee_caption(fig, text, fig_id="Fig. 1.", column='one'):
    """
    Draw an IEEE-style caption inside the figure bottom area.
    If you will use LaTeX \caption, you can skip calling this.
    """
    if not text:
        return
    # Caption style: "Fig. X. Sentence case caption …"
    # Put slightly below axes area; adjust pad per column width
    pad = 0.02 if column == 'one' else 0.015
    caption = f"{fig_id} {text}"
    fig.text(0.5, -pad, caption, ha='center', va='top', fontsize=8)


# ==========================================
# === Figure 1: Accuracy (left) & ASR (right)
# ==========================================
def plot_attack_performance_enhanced(json_file_path, output_dir=None,
                                     column='one',
                                     caption="Impact of GRMP attack on FL performance (clean accuracy and ASR).",
                                     fig_id="Fig. 1."
                                     ):
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    metrics = data['progressive_metrics']
    rounds = metrics['rounds'][:20]
    asr = metrics['attack_asr'][:20]
    acc = metrics['clean_acc'][:20]

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    if not rounds:
        print("No rounds in progressive_metrics; skip Figure 1.")
        return

    # Apply IEEE style
    w, h = use_ieee_minimal(column=column)

    fig, ax1 = plt.subplots(figsize=(w, h))
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)

    # --- No background shading (requirement 1) ---

    # Colors & styles (requirement 2: distinguish by color+linestyle+marker)
    c_acc = '#1a1a1a'   # near-black
    c_asr = '#B22222'   # dark red
    l1, = ax1.plot(rounds, acc, '-o', label='Learning Accuracy', color=c_acc)
    l2, = ax2.plot(rounds, asr, '--s', label='ASR', color=c_asr)

    # Axes labels
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('ASR')

    # Percentage formatter (if values are 0..1)
    ax1.yaxis.set_major_formatter(percent_formatter())
    ax2.yaxis.set_major_formatter(percent_formatter())

    # Ticks & limits
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.set_xlim(min(rounds) - 0.2, max(rounds) + 0.2)

    set_minimal_grid(ax1, axis='y')
    safe_ylim(ax1, acc, top_extra=0.25)
    safe_ylim(ax2, asr, top_extra=0.25)

    # Legend (minimal)
    lines, labels = [l1, l2], [l1.get_label(), l2.get_label()]
    ax1.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout()
    # add_ieee_caption(fig, caption, fig_id=fig_id, column=column)

    # Save
    out_png = output_dir / 'figure1_attack_performance.png'
    out_pdf = output_dir / 'figure1_attack_performance.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')             # vector for paper
    fig.savefig(out_png, dpi=600, bbox_inches='tight')    # high-res preview
    print(f"Figure 1 saved to: {out_pdf}")
    plt.close(fig)


# ============================================================
# === Figure 2: Similarity evolution (mean benign + attackers)
# ============================================================
def plot_similarity_evolution_bars_style(json_file_path, output_dir=None,
                                         column='one',
                                         caption="Stealthiness analysis: evolution of cosine similarity "
                                                 "(benign mean vs. individual attackers) with threshold.",
                                         fig_id="Fig. 2."):
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    # Skip round==0 (initial_eval has no 'defense')
    results_all = data['results']
    results = [r for r in results_all if r.get('round', 0) > 0][:20]

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    if not results:
        print("No round>0 entries; skip Figure 2.")
        return

    # Apply IEEE style
    w, h = use_ieee_minimal(column=column)
    fig, ax = plt.subplots(figsize=(w, h))

    # Identify attackers (assumption consistent with main.py)
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
    attacker_ids = list(range(num_clients - num_attackers, num_clients))

    rounds = []
    thresholds = []
    attacker_sims_by_round = []
    benign_sims_by_round = []

    for round_data in results:
        rounds.append(round_data['round'])
        defense = round_data.get('defense', {})
        thresholds.append(defense.get('threshold', np.nan))
        sims = defense.get('similarities', [])

        benign_sims, attacker_sims = [], []
        for i, sim in enumerate(sims):
            (attacker_sims if i in attacker_ids else benign_sims).append(sim)

        benign_sims_by_round.append(benign_sims)
        attacker_sims_by_round.append(attacker_sims)

    # Compute benign mean similarity
    benign_avg = [float(np.mean(b)) if b else np.nan for b in benign_sims_by_round]

    # Minimal lines (no bars/background). Threshold as thin gray line.
    c_th = '#7f7f7f'
    c_benign = '#1a1a1a'  # near-black
    c_att = ['#B22222', '#FF8C00', '#2F4F4F', '#6A5ACD']  # attackers (rotate if >2)

    ax.plot(rounds, thresholds, '-', color=c_th, label='Threshold')
    ax.plot(rounds, benign_avg, '--o', color=c_benign, label='Benign (mean)')

    # Attackers
    for k in range(num_attackers):
        traj = []
        for sims in attacker_sims_by_round:
            traj.append(sims[k] if k < len(sims) else np.nan)
        # remove nans at ends for plotting clarity
        valid_r = [r for r, v in zip(rounds, traj) if not np.isnan(v)]
        valid_v = [v for v in traj if not np.isnan(v)]
        if valid_v:
            style = ['-.', ':', '--', '-.'][k % 4]
            marker = ['s', '^', 'd', 'v'][k % 4]
            ax.plot(valid_r, valid_v, style, marker=marker, color=c_att[k % len(c_att)],
                    label=f'Attacker {k+1}')

    # Labels/axes
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Cosine Similarity')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlim(min(rounds) - 0.2, max(rounds) + 0.2)

    set_minimal_grid(ax, axis='y')
    # y-limits (robust)
    all_vals = []
    all_vals.extend([v for v in thresholds if not np.isnan(v)])
    all_vals.extend([v for v in benign_avg if not np.isnan(v)])
    for k in range(num_attackers):
        for sims in attacker_sims_by_round:
            if k < len(sims):
                all_vals.append(sims[k])
    safe_ylim(ax, all_vals, top_extra=0.25)

    # Legend
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout()
    # add_ieee_caption(fig, caption, fig_id=fig_id, column=column)

    out_png = output_dir / 'figure2_similarity_evolution.png'
    out_pdf = output_dir / 'figure2_similarity_evolution.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    print(f"Figure 2 saved to: {out_pdf}")
    plt.close(fig)


# =======================================================
# === Figure 3: Individual benign users' similarity lines
# =======================================================
def plot_similarity_individual_benign(json_file_path, output_dir=None,
                                      column='one',
                                      caption="Individual benign users vs. attackers: "
                                              "evolution of cosine similarity.",
                                      fig_id="Fig. 3."):
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results_all = data['results']
    results = [r for r in results_all if r.get('round', 0) > 0][:20]

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    if not results:
        print("No round>0 entries; skip Figure 3.")
        return

    # Apply IEEE style
    w, h = use_ieee_minimal(column=column)
    fig, ax = plt.subplots(figsize=(w, h))

    # Identify attackers / benign
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
        defense = round_data.get('defense', {})
        thresholds.append(defense.get('threshold', np.nan))
        sims = defense.get('similarities', [])

        benign_sims, attacker_sims = [], []
        for i, sim in enumerate(sims):
            (attacker_sims if i in attacker_ids else benign_sims).append(sim)

        benign_sims_by_round.append(benign_sims)
        attacker_sims_by_round.append(attacker_sims)

    # Colors & styles
    # benign users: grayscale hues + diverse markers
    benign_colors = ['#1a1a1a', '#4d4d4d', '#7f7f7f', '#a6a6a6',
                     '#595959', '#737373', '#8c8c8c', '#b3b3b3']
    benign_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    # attackers: distinct colors
    attacker_colors = ['#B22222', '#FF8C00', '#2F4F4F', '#6A5ACD']

    # Plot threshold as thin gray line
    ax.plot(rounds, thresholds, '-', color='#7f7f7f', label='Threshold')

    # Plot benign trajectories
    for uid in range(min(num_benign, 8)):  # 支持最多8条，更多时可合并平均
        traj = []
        for b in benign_sims_by_round:
            traj.append(b[uid] if uid < len(b) else np.nan)
        valid_r = [r for r, v in zip(rounds, traj) if not np.isnan(v)]
        valid_v = [v for v in traj if not np.isnan(v)]
        if valid_v:
            ax.plot(valid_r, valid_v,
                    '-', marker=benign_markers[uid % len(benign_markers)],
                    color=benign_colors[uid % len(benign_colors)],
                    label=f'Benign {uid+1}')

    # Plot attackers
    for k in range(num_attackers):
        traj = []
        for sims in attacker_sims_by_round:
            traj.append(sims[k] if k < len(sims) else np.nan)
        valid_r = [r for r, v in zip(rounds, traj) if not np.isnan(v)]
        valid_v = [v for v in traj if not np.isnan(v)]
        if valid_v:
            style = ['--', '-.', ':', '--'][k % 4]
            marker = ['s', '^', 'd', 'v'][k % 4]
            ax.plot(valid_r, valid_v, style, marker=marker,
                    color=attacker_colors[k % len(attacker_colors)],
                    label=f'Attacker {k+1}')

    # Labels/axes
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Cosine Similarity')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlim(min(rounds) - 0.2, max(rounds) + 0.2)

    set_minimal_grid(ax, axis='y')

    # y-limits
    all_vals = []
    all_vals.extend([v for v in thresholds if not np.isnan(v)])
    for uid in range(min(num_benign, 8)):
        for b in benign_sims_by_round:
            if uid < len(b):
                all_vals.append(b[uid])
    for k in range(num_attackers):
        for sims in attacker_sims_by_round:
            if k < len(sims):
                all_vals.append(sims[k])
    safe_ylim(ax, all_vals, top_extra=0.25)

    # Legend (compact)
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout()
    # add_ieee_caption(fig, caption, fig_id=fig_id, column=column)

    out_png = output_dir / 'figure3_individual_similarity.png'
    out_pdf = output_dir / 'figure3_individual_similarity.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    print(f"Figure 3 saved to: {out_pdf}")
    plt.close(fig)


# ===============================
# === Driver to generate figures
# ===============================
def generate_paper_figures(json_file_path, column='one'):
    print("Generating figures (IEEE minimal style)...")
    plot_attack_performance_enhanced(json_file_path, column=column)
    plot_similarity_evolution_bars_style(json_file_path, column=column)
    plot_similarity_individual_benign(json_file_path, column=column)
    print("\nAll figures saved in: ./results/figures/")


if __name__ == "__main__":
    # Path to your JSON file
    json_file = './results/progressive_grmp_progressive_semantic_poisoning.json'
    # 'one' for single-column (3.5in), 'two' for double-column (7.16in)
    generate_paper_figures(json_file, column='one')