"""
Enhanced GRMP Attack Visualization for Top-tier Journal
Three core figures with professional styling

SIZE ADJUSTMENT GUIDE:
---------------------
To adjust figure and font sizes, modify these parameters:

1. FIGURE SIZE: Change figsize in plot functions
   - Figure 1: fig, ax1 = plt.subplots(figsize=(10, 6))  # (width, height) in inches
   - Figure 2: fig, ax = plt.subplots(figsize=(12, 7))
   - Recommended for larger: (12, 8) or (14, 8)

2. FONT SIZES: Modify plt.rcParams at the beginning
   - Base font: plt.rcParams['font.size'] = 14  # Try 16 or 18
   - Title: plt.rcParams['axes.titlesize'] = 18  # Try 20 or 22
   - Axis labels: plt.rcParams['axes.labelsize'] = 16  # Try 18 or 20
   - Tick labels: plt.rcParams['xtick/ytick.labelsize'] = 14  # Try 16
   - Legend: plt.rcParams['legend.fontsize'] = 13  # Try 15

3. LINE WIDTH & MARKER SIZE:
   - linewidth=3  # Try 4 or 5 for thicker lines
   - markersize=10  # Try 12 or 14 for larger markers

4. DPI for high resolution:
   - fig.savefig(..., dpi=300)  # Try 600 for ultra-high resolution
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set professional font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# Remove top and right spines for cleaner look
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def plot_attack_performance_enhanced(json_file_path, output_dir=None):
    """
    Figure 1: Learning Accuracy (left) and ASR (right) with professional styling
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    metrics = data['progressive_metrics']

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 6))

    rounds = metrics['rounds']
    asr = metrics['attack_asr']
    fl_acc = metrics['clean_acc']

    # Create second y-axis
    ax2 = ax1.twinx()

    # Remove top spine for ax2 as well
    ax2.spines['top'].set_visible(False)

    # Add background shading with soft colors
    # Trust building phase (light green)
    ax1.axvspan(0.5, 5.5, alpha=0.15, color='#90EE90', zorder=0)
    # ax1.text(3, 0.85, 'Trust Building', ha='center', va='center',
    #          fontsize=14, fontweight='bold', color='#2E8B57', alpha=0.8)

    # Attack escalation phase (light red)
    ax1.axvspan(5.5, 10.5, alpha=0.15, color='#FFB6C1', zorder=0)
    # ax1.text(8, 0.85, 'Attack Phase', ha='center', va='center',
    #          fontsize=14, fontweight='bold', color='#DC143C', alpha=0.8)

    # Plot Learning Accuracy (left axis) - MODIFIED
    line1 = ax1.plot(rounds, fl_acc, 's-', color='#4169E1', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#4169E1', label='Learning Accuracy')

    ax1.set_xlabel('Communication Round', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Learning Accuracy', fontsize=16, color='#4169E1', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4169E1', labelsize=14)

    # Plot ASR (right axis) - MODIFIED
    line2 = ax2.plot(rounds, asr, 'o-', color='#DC143C', linewidth=3,
                    markersize=10, markerfacecolor='white', markeredgewidth=2.5,
                    markeredgecolor='#DC143C', label='Attack Success Rate (ASR)')

    ax2.set_ylabel('Attack Success Rate', fontsize=16, color='#DC143C', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#DC143C', labelsize=14)

    # Highlight peak ASR
    max_asr_idx = asr.index(max(asr))
    ax2.scatter(rounds[max_asr_idx], asr[max_asr_idx], s=200,
               color='#DC143C', zorder=5, edgecolors='black', linewidth=2)

    # Add annotation for peak

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True,
              fancybox=True, shadow=True, framealpha=0.9)

    # Title
    ax1.set_title('Impact of GRMP Attack on Federated Learning Performance',
                 fontsize=18, fontweight='bold', pad=20)

    # Grid
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Set limits
    ax1.set_xlim(0.5, 10.5)
    ax1.set_ylim(0.75, 0.95)  # Adjusted for accuracy range
    ax2.set_ylim(-0.05, 0.9)

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


def plot_similarity_evolution_bars_style(json_file_path, output_dir=None):
    """
    Figure 2: Similarity evolution with bars for benign users and lines for attackers
    Following the reference style from the advisor
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results']

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Identify attackers
    num_clients = config['num_clients']
    num_attackers = config['num_attackers']
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

    # Plot defense threshold as bars - MODIFIED
    bar_width = 0.8
    threshold_bars = ax.bar(rounds, thresholds, width=bar_width,
                           color='#228B22', alpha=0.3, edgecolor='#228B22',
                           linewidth=1.5, label='Defense threshold', zorder=1)

    # Plot benign users as average line - MODIFIED back to average
    benign_avg_sims = []
    for benign_sims in benign_sims_by_round:
        if benign_sims:
            benign_avg_sims.append(np.mean(benign_sims))
        else:
            benign_avg_sims.append(0)

    # Plot benign users average line
    ax.plot(rounds, benign_avg_sims, 'o-', color='#4682B4',
           linewidth=3, markersize=10, markerfacecolor='white',
           markeredgewidth=2.5, label='Benign users (avg)',
           zorder=4)

    # Plot attackers as lines (keep original) - MODIFIED z-order
    # Reorganize attacker data by attacker ID
    attacker_trajectories = {aid: [] for aid in range(num_attackers)}

    for round_sims in attacker_sims_by_round:
        for aid in range(num_attackers):
            if aid < len(round_sims):
                attacker_trajectories[aid].append(round_sims[aid])
            else:
                attacker_trajectories[aid].append(None)

    # Plot each attacker's trajectory
    attacker_colors = ['#DC143C', '#FF6347']
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

    # Add subtle background regions (adjusted for bar chart)
    # No need for fill_between since threshold is now shown as bars

    # Add text annotations for zones
    # ax.text(5.5, 0.15, 'Below Threshold', ha='center', va='center',
    #        fontsize=12, color='red', alpha=0.5, fontweight='bold')
    # ax.text(5.5, 0.65, 'Above Threshold', ha='center', va='center',
    #        fontsize=12, color='green', alpha=0.5, fontweight='bold')

    # Styling
    ax.set_xlabel('Communication Round', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=16, fontweight='bold')
    ax.set_title('Stealthiness of GRMP Attack: Similarity Evolution Analysis',
                fontsize=18, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True,
             shadow=True, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Set limits
    ax.set_xlim(0.5, max(rounds) + 0.5)
    ax.set_ylim(0, 0.8)

    # Set x-ticks
    ax.set_xticks(rounds)

    # Removed annotation about attackers evading detection

    # Save figure
    plt.tight_layout()

    output_path_png = output_dir / 'figure2_similarity_evolution.png'
    output_path_pdf = output_dir / 'figure2_similarity_evolution.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')

    print(f"Figure 2 saved to: {output_path_png}")
    plt.close()


def plot_similarity_individual_benign(json_file_path, output_dir=None):
    """
    Figure 3: Individual benign users' similarity evolution
    Same style as Figure 2 but showing each benign user separately
    """
    # Load data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results']

    if output_dir is None:
        output_dir = Path('./results/figures')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

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
                           color='#228B22', alpha=0.3, edgecolor='#228B22',
                           linewidth=1.5, label='Defense threshold', zorder=1)

    # Plot each benign user individually
    benign_colors = ['#4682B4', '#5F9EA0', '#6495ED']  # Different blues
    benign_markers = ['o', 's', '^']  # Different markers
    
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
            ax.plot(valid_rounds, valid_sims, f'{benign_markers[uid]}-',
                   color=benign_colors[uid % len(benign_colors)],
                   linewidth=3, markersize=10, markerfacecolor='white',
                   markeredgewidth=2.5,
                   label=f'Benign user {uid + 1}',
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
    attacker_colors = ['#DC143C', '#FF6347']
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
    ax.set_xlabel('Communication Round', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=16, fontweight='bold')
    ax.set_title('Individual Client Similarity Evolution',
                fontsize=18, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True,
             shadow=True, framealpha=0.9, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Set limits
    ax.set_xlim(0.5, max(rounds) + 0.5)
    ax.set_ylim(0, 0.8)

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


def generate_paper_figures(json_file_path):
    """
    Generate all three figures for the paper
    """
    print("Generating enhanced figures for paper...")

    # Generate Figure 1: Attack Performance
    plot_attack_performance_enhanced(json_file_path)

    # Generate Figure 2: Similarity Evolution (with average)
    plot_similarity_evolution_bars_style(json_file_path)
    
    # Generate Figure 3: Individual Similarity Evolution
    plot_similarity_individual_benign(json_file_path)

    print("\nAll three figures generated successfully!")
    print("Files saved in: ./results/figures/")


if __name__ == "__main__":
    # Path to your JSON file
    json_file = 'progressive_grmp_progressive_semantic_poisoning.json'

    # Generate all three figures
    generate_paper_figures(json_file)