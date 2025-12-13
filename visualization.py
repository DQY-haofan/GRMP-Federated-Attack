# visualization.py
# Visualization module for GRMP attack experiment results
# Generates plots matching the paper's Figure 3, 4, 5, and 6

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')  # Fallback to default style

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


class ExperimentVisualizer:
    """Visualizer for GRMP attack experiment results"""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_results(self, results_path: str) -> Dict:
        """Load experiment results from JSON file"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_figure3_global_accuracy_stability(self, log_data: List[Dict], save_path: Optional[str] = None, num_rounds: Optional[int] = None):
        """
        Figure 3 (revised): Global learning accuracy, stability (|Δacc|), and rejection rate.
        Suitable for data-agnostic poisoning where ASR is not the key metric.
        """
        rounds = [log['round'] for log in log_data]
        clean_acc = [log.get('clean_accuracy', 0.0) for log in log_data]
        rejection_rate = [log.get('defense', {}).get('rejection_rate', 0.0) for log in log_data]
        
        # Accuracy change magnitude |Δacc|
        acc_diff = [0.0]
        for i in range(1, len(clean_acc)):
            acc_diff.append(abs(clean_acc[i] - clean_acc[i-1]))
        
        # Validate/pad
        if num_rounds is not None and len(rounds) != num_rounds:
            print(f"  ⚠️  Warning: Figure 3 - Expected {num_rounds} rounds, got {len(rounds)}")
            expected_rounds = list(range(1, num_rounds + 1))
            if len(rounds) < num_rounds:
                missing = [r for r in expected_rounds if r not in rounds]
                for _ in missing:
                    rounds.append(expected_rounds[len(rounds)])
                    clean_acc.append(clean_acc[-1] if clean_acc else 0.0)
                    rejection_rate.append(rejection_rate[-1] if rejection_rate else 0.0)
                    acc_diff.append(0.0)
                print(f"     Padded {len(missing)} missing rounds")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_acc = 'tab:blue'
        ax1.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Global Learning Accuracy', color=color_acc, fontsize=12, fontweight='bold')
        line1 = ax1.plot(rounds, clean_acc, 'o-', color=color_acc, linewidth=2,
                         markersize=6, label='Accuracy', zorder=3)
        acc_min = [a - 0.01 for a in clean_acc]
        acc_max = [a + 0.01 for a in clean_acc]
        ax1.fill_between(rounds, acc_min, acc_max, alpha=0.15, color=color_acc,
                         label='Accuracy Range')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim([max(0.0, min(clean_acc) - 0.05), min(1.0, max(clean_acc) + 0.05)])
        ax1.set_xlim([1, max(rounds) if rounds else 1])
        ax1.grid(True, alpha=0.3)
        
        # Right Y-axis: rejection rate and |Δacc|
        ax2 = ax1.twinx()
        color_rej = 'tab:red'
        color_delta = 'tab:orange'
        ax2.set_ylabel('Rejection Rate / |Δacc|', color=color_rej, fontsize=12, fontweight='bold')
        line2 = ax2.plot(rounds, rejection_rate, 's-', color=color_rej, linewidth=2,
                         markersize=6, label='Rejection Rate', zorder=3)
        line3 = ax2.plot(rounds, acc_diff, 'd--', color=color_delta, linewidth=2,
                         markersize=5, label='|Δacc|', zorder=2)
        ax2.tick_params(axis='y', labelcolor=color_rej)
        ax2.set_ylim([0.0, max(0.01, max(rejection_rate + acc_diff)) * 1.2])
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
        
        plt.title('Figure 3: Global Accuracy, Stability (|Δacc|) and Rejection Rate',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        out_path = save_path or (self.results_dir / 'figure3_global_accuracy_stability.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved Figure 3 to: {out_path}")
        plt.close()
    
    def plot_figure4_cosine_similarity(self, log_data: List[Dict], 
                                      attacker_ids: Optional[List[int]] = None,
                                      save_path: Optional[str] = None,
                                      num_rounds: Optional[int] = None):
        """
        Figure 4: Temporal evolution of cosine similarity for each LLM agent 
        with dynamic detection threshold over communication rounds.
        
        Args:
            log_data: List of round logs
            attacker_ids: List of attacker client IDs (if None, assumes last 2 clients are attackers)
            save_path: Path to save the figure
            num_rounds: Total number of rounds (for validation)
        """
        rounds = [log['round'] for log in log_data]
        
        # Validate data length
        if num_rounds is not None and len(rounds) != num_rounds:
            print(f"  ⚠️  Warning: Figure 4 - Expected {num_rounds} rounds, got {len(rounds)}")
            if len(rounds) < num_rounds:
                # Pad missing rounds (similarities will be handled in the loop)
                expected_rounds = list(range(1, num_rounds + 1))
                missing_rounds = [r for r in expected_rounds if r not in rounds]
                # Add placeholder logs for missing rounds
                if log_data:
                    last_log = log_data[-1].copy()
                    for r in missing_rounds:
                        placeholder_log = last_log.copy()
                        placeholder_log['round'] = r
                        placeholder_log['defense'] = last_log.get('defense', {}).copy()
                        log_data.append(placeholder_log)
                    rounds = expected_rounds
                print(f"     Padded {len(missing_rounds)} missing rounds")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extract similarities and client info from defense logs
        # Build a mapping of client_id -> list of similarities over rounds
        client_similarities = {}  # {client_id: [sim1, sim2, ...]}
        thresholds = []
        
        for log in log_data:
            defense = log.get('defense', {})
            similarities = defense.get('similarities', [])
            accepted_clients = defense.get('accepted_clients', [])
            rejected_clients = defense.get('rejected_clients', [])
            threshold = defense.get('threshold', 0.0)
            thresholds.append(threshold)
            
            # Map similarities to client IDs
            # The similarities list is ordered by sorted client IDs (from aggregate_updates)
            # Combine accepted and rejected to get all client IDs
            all_client_ids = sorted(set(accepted_clients + rejected_clients))
            
            # Ensure we have the same number of similarities as clients
            if len(similarities) != len(all_client_ids):
                # If mismatch, try to infer from the log structure
                # Similarities should match the order of sorted client IDs
                print(f"  ⚠️  Warning: Similarity count ({len(similarities)}) != client count ({len(all_client_ids)}) in round {log.get('round', '?')}")
            
            for i, client_id in enumerate(all_client_ids):
                if client_id not in client_similarities:
                    client_similarities[client_id] = []
                
                if i < len(similarities):
                    client_similarities[client_id].append(similarities[i])
                else:
                    # Pad with previous value or 0 if no previous value
                    if len(client_similarities[client_id]) > 0:
                        client_similarities[client_id].append(client_similarities[client_id][-1])
                    else:
                        client_similarities[client_id].append(0.0)
        
        # Separate into benign and attacker
        all_ids = sorted(client_similarities.keys())
        if attacker_ids is None:
            # Default assumption: last 2 clients are attackers
            num_attackers = 2
            attacker_ids_set = set(all_ids[-num_attackers:]) if len(all_ids) >= num_attackers else set()
        else:
            attacker_ids_set = set(attacker_ids)
        
        benign_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                         for cid in all_ids if cid not in attacker_ids_set]
        attacker_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                           for cid in all_ids if cid in attacker_ids_set]
        
        # Plot benign agents (blue lines with different markers)
        markers_benign = ['o', '^', 's', 'D']
        colors_benign = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
        for i, client in enumerate(benign_clients[:4]):  # Limit to 4 benign agents
            sims = client['sims']
            # Align similarities with rounds length
            if len(sims) < len(rounds):
                # Pad with last value if incomplete
                sims = sims + [sims[-1] if len(sims) > 0 else 0.0] * (len(rounds) - len(sims))
            elif len(sims) > len(rounds):
                # Truncate if too long
                sims = sims[:len(rounds)]
            
            # Now lengths should match
            if len(sims) == len(rounds):
                ax.plot(rounds, sims, marker=markers_benign[i % len(markers_benign)], 
                       color=colors_benign[i % len(colors_benign)], linewidth=2, 
                       markersize=5, label=f'Benign Agent {client["id"]+1}', alpha=0.8)
            else:
                print(f"  ⚠️  Warning: Benign Client {client['id']} - sims length ({len(sims)}) != rounds length ({len(rounds)})")
        
        # Plot attacker agents (red/orange lines with square markers)
        markers_attacker = ['s', 's']
        colors_attacker = ['#d62728', '#ff7f0e']
        for i, client in enumerate(attacker_clients[:2]):  # Limit to 2 attackers
            sims = client['sims']
            # Align similarities with rounds length
            if len(sims) < len(rounds):
                # Pad with last value if incomplete
                sims = sims + [sims[-1] if len(sims) > 0 else 0.0] * (len(rounds) - len(sims))
            elif len(sims) > len(rounds):
                # Truncate if too long
                sims = sims[:len(rounds)]
            
            # Now lengths should match
            if len(sims) == len(rounds):
                ax.plot(rounds, sims, marker=markers_attacker[i], 
                       color=colors_attacker[i % len(colors_attacker)], linewidth=2, 
                       markersize=6, label=f'Attacker {client["id"]+1}', alpha=0.8)
            else:
                print(f"  ⚠️  Warning: Attacker Client {client['id']} - sims length ({len(sims)}) != rounds length ({len(rounds)})")
        
        # Align thresholds with rounds length
        if len(thresholds) < len(rounds):
            # Pad with last value if incomplete
            thresholds = thresholds + [thresholds[-1] if len(thresholds) > 0 else 0.0] * (len(rounds) - len(thresholds))
        elif len(thresholds) > len(rounds):
            # Truncate if too long
            thresholds = thresholds[:len(rounds)]
        
        # Plot defense threshold (green bars/line)
        if len(thresholds) == len(rounds):
            ax.plot(rounds, thresholds, 'g-', linewidth=2.5, label='Defense Threshold', 
                   alpha=0.7, linestyle='--')
        else:
            print(f"  ⚠️  Warning: thresholds length ({len(thresholds)}) != rounds length ({len(rounds)})")
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([1, max(rounds) if rounds else 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9, ncol=2)
        
        plt.title('Figure 4: Temporal Evolution of Cosine Similarity for Each LLM Agent\n'
                 'with Dynamic Detection Threshold', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 4 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure4_cosine_similarity.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_figure5_local_accuracy_no_attack(self, local_accuracies: Dict[int, List[float]], 
                                             rounds: List[int], save_path: Optional[str] = None):
        """
        Figure 5: Learning accuracy of local LLM agents with no attack 
        over communication rounds.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each benign agent
        markers = ['o', '^', 's', 'D', 'v', 'p']
        colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i, (client_id, accs) in enumerate(sorted(local_accuracies.items())):
            # Align accuracies with rounds length
            if len(accs) < len(rounds):
                # Pad with last value if incomplete
                accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
            elif len(accs) > len(rounds):
                # Truncate if too long
                accs = accs[:len(rounds)]
            
            # Now lengths should match
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers[i % len(markers)], 
                       color=colors[i % len(colors)], linewidth=2, markersize=5,
                       label=f'Benign Agent {client_id+1}', alpha=0.8)
            else:
                print(f"  ⚠️  Warning: Client {client_id} - accs length ({len(accs)}) != rounds length ({len(rounds)})")
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Local Learning Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0.80, 0.95])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        
        plt.title('Figure 5: Learning Accuracy of Local LLM Agents\n'
                 'with No Attack', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 5 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure5_local_accuracy_no_attack.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_figure6_local_accuracy_with_attack(self, local_accuracies: Dict[int, List[float]], 
                                                rounds: List[int], 
                                                attacker_ids: List[int],
                                                save_path: Optional[str] = None):
        """
        Figure 6: Learning accuracy of local LLM agents under the GRMP attack 
        over communication rounds.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate benign and attacker
        benign_accs = {cid: accs for cid, accs in local_accuracies.items() 
                      if cid not in attacker_ids}
        attacker_accs = {cid: accs for cid, accs in local_accuracies.items() 
                        if cid in attacker_ids}
        
        # Plot benign agents (blue lines)
        markers_benign = ['o', '^', 's', 'D']
        colors_benign = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
        for i, (client_id, accs) in enumerate(sorted(benign_accs.items())[:4]):
            # Align accuracies with rounds length
            if len(accs) < len(rounds):
                # Pad with last value if incomplete
                accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
            elif len(accs) > len(rounds):
                # Truncate if too long
                accs = accs[:len(rounds)]
            
            # Now lengths should match
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers_benign[i % len(markers_benign)], 
                       color=colors_benign[i % len(colors_benign)], linewidth=2, 
                       markersize=5, label=f'Benign Agent {client_id+1}', alpha=0.8)
            else:
                print(f"  ⚠️  Warning: Benign Client {client_id} - accs length ({len(accs)}) != rounds length ({len(rounds)})")
        
        # Plot attacker agents (red/orange lines with square markers)
        markers_attacker = ['s', 's']
        colors_attacker = ['#d62728', '#ff7f0e']
        for i, (client_id, accs) in enumerate(sorted(attacker_accs.items())[:2]):
            # Align accuracies with rounds length
            if len(accs) < len(rounds):
                # Pad with last value if incomplete
                accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
            elif len(accs) > len(rounds):
                # Truncate if too long
                accs = accs[:len(rounds)]
            
            # Now lengths should match
            if len(accs) == len(rounds):
                ax.plot(rounds, accs, marker=markers_attacker[i], 
                       color=colors_attacker[i % len(colors_attacker)], linewidth=2, 
                       markersize=6, label=f'Attacker {client_id+1}', alpha=0.8)
            else:
                print(f"  ⚠️  Warning: Attacker Client {client_id} - accs length ({len(accs)}) != rounds length ({len(rounds)})")
        
        ax.set_xlabel('Communication Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Local Learning Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim([0.70, 1.00])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        
        plt.title('Figure 6: Learning Accuracy of Local LLM Agents\n'
                 'under the GRMP Attack', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved Figure 6 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure6_local_accuracy_with_attack.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_figures(self, server_log_data: List[Dict], 
                            local_accuracies: Optional[Dict[int, List[float]]] = None,
                            attacker_ids: Optional[List[int]] = None,
                            experiment_name: str = "experiment",
                            baseline_local_accuracies: Optional[Dict[int, List[float]]] = None,
                            num_rounds: Optional[int] = None,
                            attack_start_round: Optional[int] = None):
        """
        Generate all figures from the paper.
        
        Args:
            server_log_data: List of round logs from server (attack experiment)
            local_accuracies: Dict mapping client_id to list of local accuracies per round (attack experiment)
            attacker_ids: List of attacker client IDs
            experiment_name: Name for output files
            baseline_local_accuracies: Dict for baseline (no attack) experiment - used for Figure 5
            num_rounds: Total number of rounds (from config) - ensures all figures use correct round count
            attack_start_round: Round when attack phase starts (for Figure 5 fallback)
        """
        print("\n" + "=" * 60)
        print("Generating Visualization Figures")
        print("=" * 60)
        
        # Extract rounds from log_data, but ensure alignment with num_rounds
        rounds = [log['round'] for log in server_log_data]
        
        # Validate and align rounds with num_rounds if provided
        if num_rounds is not None:
            expected_rounds = list(range(1, num_rounds + 1))
            if len(rounds) != num_rounds:
                print(f"  ⚠️  Warning: log_data has {len(rounds)} rounds, but num_rounds={num_rounds}")
                print(f"     Expected rounds: 1 to {num_rounds}")
                if len(rounds) > 0:
                    print(f"     Actual rounds in log_data: {rounds[:min(5, len(rounds))]}...{rounds[-min(5, len(rounds)):] if len(rounds) > 10 else rounds}")
                # Use expected rounds if log_data is incomplete
                if len(rounds) < num_rounds:
                    print(f"     Using expected rounds (1 to {num_rounds}) for all figures")
                    rounds = expected_rounds
                else:
                    # If log_data has more rounds, truncate to num_rounds
                    print(f"     Truncating to first {num_rounds} rounds")
                    rounds = rounds[:num_rounds]
                    # Create a copy to avoid modifying original data
                    server_log_data = server_log_data[:num_rounds]
        
        # Figure 3: Global Accuracy, Stability, Rejection Rate (from attack experiment)
        print("\n📊 Generating Figure 3: Global Accuracy, Stability, Rejection Rate...")
        self.plot_figure3_global_accuracy_stability(
            server_log_data,
            save_path=self.results_dir / f'{experiment_name}_figure3.png',
            num_rounds=num_rounds
        )
        
        # Figure 4: Cosine Similarity (from attack experiment)
        print("📊 Generating Figure 4: Cosine Similarity...")
        self.plot_figure4_cosine_similarity(
            server_log_data,
            attacker_ids=attacker_ids,
            save_path=self.results_dir / f'{experiment_name}_figure4.png',
            num_rounds=num_rounds
        )
        
        # Figure 5: No Attack (requires baseline experiment data)
        if baseline_local_accuracies is not None:
            print("📊 Generating Figure 5: Local Accuracy (No Attack)...")
            # Use num_rounds if provided, otherwise infer from baseline data
            if num_rounds is not None:
                baseline_rounds = list(range(1, num_rounds + 1))
                # Ensure local_accuracies align with num_rounds
                aligned_baseline_accs = {}
                for cid, accs in baseline_local_accuracies.items():
                    if len(accs) < num_rounds:
                        # Pad with last value if incomplete
                        accs = accs + [accs[-1]] * (num_rounds - len(accs))
                    elif len(accs) > num_rounds:
                        # Truncate if too long
                        accs = accs[:num_rounds]
                    aligned_baseline_accs[cid] = accs
                baseline_local_accuracies = aligned_baseline_accs
            else:
                # Fallback: infer from data length
                baseline_rounds = list(range(1, len(baseline_local_accuracies.get(
                    list(baseline_local_accuracies.keys())[0], [])) + 1))
            
            self.plot_figure5_local_accuracy_no_attack(
                baseline_local_accuracies, baseline_rounds,
                save_path=self.results_dir / f'{experiment_name}_figure5.png'
            )
        elif local_accuracies is not None:
            # Fallback: Use learning phase data (rounds < attack_start_round) as "no attack"
            print("📊 Generating Figure 5: Local Accuracy (Learning Phase - Low Attack)...")
            if attacker_ids is None:
                attacker_ids = []
            
            # Use attack_start_round from parameter, or default to 10
            learning_phase_end = attack_start_round if attack_start_round is not None else 10
            # Extract learning phase data (first few rounds with minimal attack)
            learning_phase_rounds = [r for r in rounds if r <= learning_phase_end]
            benign_ids = [cid for cid in local_accuracies.keys() if cid not in attacker_ids]
            
            if benign_ids and len(learning_phase_rounds) > 0:
                learning_accs = {cid: local_accuracies[cid][:len(learning_phase_rounds)] 
                                for cid in benign_ids}
                self.plot_figure5_local_accuracy_no_attack(
                    learning_accs, learning_phase_rounds,
                    save_path=self.results_dir / f'{experiment_name}_figure5_learning_phase.png'
                )
                print("  ⚠️  Note: Using learning phase data. For true 'no attack' baseline,")
                print("     run a separate experiment with 'num_attackers': 0")
        else:
            print("  ⚠️  Figure 5 (No Attack) skipped: No baseline data available.")
            print("     To generate Figure 5, run a baseline experiment with 'num_attackers': 0")
            print("     or set 'run_both_experiments': True in config")
        
        # Figure 6: With Attack (from attack experiment)
        if local_accuracies is not None:
            print("📊 Generating Figure 6: Local Accuracy (With Attack)...")
            if attacker_ids is None:
                attacker_ids = []
            
            # Ensure local_accuracies align with rounds
            aligned_local_accs = {}
            for cid, accs in local_accuracies.items():
                if len(accs) < len(rounds):
                    # Pad with last value if incomplete
                    accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
                elif len(accs) > len(rounds):
                    # Truncate if too long
                    accs = accs[:len(rounds)]
                aligned_local_accs[cid] = accs
            
            self.plot_figure6_local_accuracy_with_attack(
                aligned_local_accs, rounds, attacker_ids,
                save_path=self.results_dir / f'{experiment_name}_figure6.png'
            )
        else:
            print("  ⚠️  Figure 6 skipped: Local accuracies not available.")
            print("     Local accuracies are automatically tracked during training.")
        
        print("\n✅ All available figures generated successfully!")
        print(f"   Output directory: {self.results_dir}")

