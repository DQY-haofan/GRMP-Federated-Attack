# visualization.py
# Visualization module for AugMP / federated fine-tuning experiment results
# Generates plots matching the paper's Figure 3, 4, 5, and 6

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style for IEEE publication-quality figures
# Use clean, minimal style without heavy grid
plt.style.use('default')

# IEEE-style parameters: clean, professional, publication-ready
plt.rcParams['figure.figsize'] = (6.5, 5)  # IEEE column width (6.5 inches)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif font family
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']  # Arial as primary font
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.borderpad'] = 0.4
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 5

# IEEE-style color palette: professional, distinct colors
# Optimized for maximum distinguishability
# Colors are carefully selected to be easily distinguishable in both print and screen
IEEE_COLORS = {
    'benign': [
        '#0066CC',  # Blue (Agent 1)
        '#FF6600',  # Orange (Agent 2) 
        '#00B050',  # Green (Agent 3)
        '#FFC000',  # Amber/Yellow (Agent 4)
        '#7030A0',  # Purple (Agent 5)
        '#C55A11',  # Brown (Agent 6)
        '#70AD47',  # Light Green (Agent 7)
        '#5B9BD5',  # Light Blue (Agent 8)
        '#2E75B6',  # Dark Blue (Agent 9)
        '#0070C0',  # Cyan Blue (Agent 10)
        '#954F72',  # Rose (Agent 11)
        '#1F4E79',  # Navy (Agent 12)
        '#000000',  # Black (Agent 13)
        '#C00000',  # Red (Agent 14) - use carefully, distinguish from attackers
        '#FF0000'   # Bright Red (Agent 15)
    ],
    'attacker': [
        '#DC143C',  # Crimson (Attacker 1)
        '#C00000',  # Dark Red (Attacker 2)
        '#FF4500',  # Orange Red (Attacker 3)
        '#B22222',  # Fire Brick (Attacker 4)
        '#E74C3C',  # Red (Attacker 5)
        '#C0392B',  # Dark Red (Attacker 6)
        '#8B0000',  # Dark Red (Attacker 7)
        '#A52A2A'   # Brown Red (Attacker 8)
    ],
    'global': '#0066CC'  # Professional blue for global accuracy
}

# IEEE-style markers: distinct, professional, optimized for clarity
IEEE_MARKERS = {
    'benign': ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd', '<', '>', 'P', 'H', '8'],
    'attacker': ['s', 'D', '^', 'v', 'p', '*', 'h', 'X']
}


class ExperimentVisualizer:
    """Visualizer for federated experiment results (including AugMP runs)"""
    
    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_results(self, results_path: str) -> Dict:
        """Load experiment results from JSON file"""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_figure3_global_accuracy_stability(self, log_data: List[Dict], save_path: Optional[str] = None, num_rounds: Optional[int] = None):
        """
        Figure 3: Global learning accuracy over communication rounds.
        Displays only the global accuracy curve without additional metrics.
        """
        rounds = [log['round'] for log in log_data]
        clean_acc = [log.get('clean_accuracy', 0.0) for log in log_data]
        
        # Ensure all arrays have the same length
        min_len = min(len(rounds), len(clean_acc))
        if min_len == 0:
            print("  ⚠️  Warning: Figure 3 - No data to plot")
            return
        
        # Truncate all arrays to the same length
        rounds = rounds[:min_len]
        clean_acc = clean_acc[:min_len]
        
        # Validate/pad
        if num_rounds is not None and len(rounds) != num_rounds:
            print(f"  ⚠️  Warning: Figure 3 - Expected {num_rounds} rounds, got {len(rounds)}")
            expected_rounds = list(range(1, num_rounds + 1))
            if len(rounds) < num_rounds:
                missing = [r for r in expected_rounds if r not in rounds]
                for _ in missing:
                    rounds.append(expected_rounds[len(rounds)])
                    clean_acc.append(clean_acc[-1] if clean_acc else 0.0)
                print(f"     Padded {len(missing)} missing rounds")
        
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # IEEE-style: clean, professional appearance
        ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
        ax.set_ylabel('Global Testing Accuracy (%)', fontsize=11, fontweight='normal')
        
        # Convert to percentage for IEEE style
        clean_acc_pct = [acc * 100 for acc in clean_acc]
        
        # Plot accuracy line - IEEE style: solid line, clear marker
        ax.plot(rounds, clean_acc_pct, '-', color=IEEE_COLORS['global'], 
                linewidth=2, marker='o', markersize=4, markevery=max(1, len(rounds)//20),
                label='Global Accuracy', zorder=3, markerfacecolor=IEEE_COLORS['global'],
                markeredgecolor='white', markeredgewidth=0.5)
        
        # IEEE-style: subtle grid, clean axes
        ax.set_ylim([max(0.0, min(clean_acc_pct) - 2), min(100.0, max(clean_acc_pct) + 2)])
        ax.set_xlim([1, max(rounds) if rounds else 1])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # IEEE-style legend: clear, professional
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                 edgecolor='black', framealpha=1.0, fontsize=9)
        
        # No title for IEEE style (usually added in LaTeX)
        plt.tight_layout()
        
        out_path = save_path or (self.results_dir / 'figure3_global_accuracy_stability.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        print(f"  ✅ Saved Figure 3 to: {out_path}")
        plt.close()
    
    def plot_figure4_cosine_similarity(self, log_data: List[Dict], 
                                      attacker_ids: Optional[List[int]] = None,
                                      save_path: Optional[str] = None,
                                      num_rounds: Optional[int] = None,
                                      num_clients: Optional[int] = None,
                                      num_attackers: Optional[int] = None):
        """
        Figure 4: Temporal evolution of cosine similarity for each LLM agent 
        over communication rounds.
        
        Args:
            log_data: List of round logs
            attacker_ids: List of attacker client IDs (if None, will infer from num_clients and num_attackers)
            save_path: Path to save the figure
            num_rounds: Total number of rounds (for validation)
            num_clients: Total number of clients (for inferring attacker_ids if not provided)
            num_attackers: Number of attacker clients (for inferring attacker_ids if not provided)
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
                        placeholder_log['aggregation'] = last_log.get('aggregation', {}).copy()
                        log_data.append(placeholder_log)
                    rounds = expected_rounds
                print(f"     Padded {len(missing_rounds)} missing rounds")
        
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # Extract similarities and client info from aggregation logs
        # Build a mapping of client_id -> list of similarities over rounds
        client_similarities = {}  # {client_id: [sim1, sim2, ...]}
        
        for log in log_data:
            aggregation = log.get('aggregation', {})
            similarities = aggregation.get('similarities', [])
            accepted_clients = aggregation.get('accepted_clients', [])
            
            # Map similarities to client IDs
            # The similarities list is ordered by sorted client IDs (from aggregate_updates)
            all_client_ids = sorted(set(accepted_clients))
            
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
            # Infer attacker_ids from num_clients and num_attackers
            if num_clients is not None and num_attackers is not None:
                attacker_ids_set = set(range(num_clients - num_attackers, num_clients))
            else:
                # Fallback: use all_ids to infer (assume last clients are attackers)
                if num_attackers is not None:
                    attacker_ids_set = set(all_ids[-num_attackers:]) if len(all_ids) >= num_attackers else set()
                else:
                    # Last resort: assume 2 attackers (old behavior)
                    print("  ⚠️  Warning: Could not infer attacker_ids, assuming last 2 clients are attackers")
                    attacker_ids_set = set(all_ids[-2:]) if len(all_ids) >= 2 else set()
        else:
            attacker_ids_set = set(attacker_ids)
        
        benign_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                         for cid in all_ids if cid not in attacker_ids_set]
        attacker_clients = [{'id': cid, 'sims': client_similarities[cid]} 
                           for cid in all_ids if cid in attacker_ids_set]
        
        # Collect all similarity values for adaptive y-axis range (before plotting)
        all_similarity_values = []
        
        # Process benign clients - align and collect data
        aligned_benign_data = []
        for client in benign_clients:
            sims = client['sims']
            # Align similarities with rounds length
            if len(sims) < len(rounds):
                sims = sims + [sims[-1] if len(sims) > 0 else 0.0] * (len(rounds) - len(sims))
            elif len(sims) > len(rounds):
                sims = sims[:len(rounds)]
            if len(sims) == len(rounds):
                all_similarity_values.extend(sims)
                aligned_benign_data.append({'id': client['id'], 'sims': sims})
        
        # Process attacker clients - align and collect data
        aligned_attacker_data = []
        for client in attacker_clients:
            sims = client['sims']
            if len(sims) < len(rounds):
                sims = sims + [sims[-1] if len(sims) > 0 else 0.0] * (len(rounds) - len(sims))
            elif len(sims) > len(rounds):
                sims = sims[:len(rounds)]
            if len(sims) == len(rounds):
                all_similarity_values.extend(sims)
                aligned_attacker_data.append({'id': client['id'], 'sims': sims})
        
        # Calculate adaptive y-axis range with padding
        if all_similarity_values:
            y_min = min(all_similarity_values)
            y_max = max(all_similarity_values)
            y_range = y_max - y_min
            
            # Add 10% padding on both sides
            padding = max(y_range * 0.1, 0.05)  # At least 0.05 padding
            y_min_adjusted = max(0.0, y_min - padding)  # Don't go below 0
            y_max_adjusted = min(1.0, y_max + padding)  # Don't go above 1
            
            # If range is very small, ensure minimum range of 0.2
            if y_max_adjusted - y_min_adjusted < 0.2:
                center = (y_min + y_max) / 2
                y_min_adjusted = max(0.0, center - 0.1)
                y_max_adjusted = min(1.0, center + 0.1)
        else:
            # Fallback to default range if no data
            y_min_adjusted = 0.0
            y_max_adjusted = 1.0
        
        # Plot benign agents - IEEE style colors (use aligned data)
        for i, client_data in enumerate(aligned_benign_data):
            sims = client_data['sims']
            # Use modulo to cycle through colors and markers if needed
            color = IEEE_COLORS['benign'][i % len(IEEE_COLORS['benign'])]
            marker = IEEE_MARKERS['benign'][i % len(IEEE_MARKERS['benign'])]
            ax.plot(rounds, sims, '-', color=color, linewidth=1.5,
                   marker=marker, markersize=4, markevery=max(1, len(rounds)//15),
                   label=f'Agent {client_data["id"]+1}', zorder=2,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        
        # Plot attacker agents - IEEE style red/orange (use aligned data)
        for i, client_data in enumerate(aligned_attacker_data):
            sims = client_data['sims']
            # Use modulo to cycle through colors and markers if needed
            color = IEEE_COLORS['attacker'][i % len(IEEE_COLORS['attacker'])]
            marker = IEEE_MARKERS['attacker'][i % len(IEEE_MARKERS['attacker'])]
            ax.plot(rounds, sims, '-', color=color, linewidth=1.5,
                   marker=marker, markersize=4, markevery=max(1, len(rounds)//15),
                   label=f'Attacker {client_data["id"]+1}', zorder=2,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        
        # IEEE-style axes (y-axis range already calculated above)
        ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
        ax.set_ylabel('Cosine Similarity', fontsize=11, fontweight='normal')
        ax.set_ylim([y_min_adjusted, y_max_adjusted])  # Adaptive y-axis range
        ax.set_xlim([1, max(rounds) if rounds else 1])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # IEEE-style legend: place inside plot area, use 'best' location to avoid blocking data
        # 'best' automatically finds the best location that minimizes overlap with plot elements
        legend = ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          edgecolor='black', framealpha=1.0, fontsize=9, 
                          ncol=1, columnspacing=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  ✅ Saved Figure 4 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure4_cosine_similarity.png', 
                       dpi=600, bbox_inches='tight')
        plt.close()
    
    def plot_figure4_euclidean_distance(self, log_data: List[Dict], 
                                        attacker_ids: Optional[List[int]] = None,
                                        save_path: Optional[str] = None,
                                        num_rounds: Optional[int] = None,
                                        num_clients: Optional[int] = None,
                                        num_attackers: Optional[int] = None):
        """
        Figure 4: Temporal evolution of Euclidean distance for each LLM agent 
        from the mean update over communication rounds.
        
        This figure shows how far each client's update is from the average update,
        which can help identify outliers (potentially malicious clients).
        
        Args:
            log_data: List of round logs
            attacker_ids: List of attacker client IDs (if None, will infer from num_clients and num_attackers)
            save_path: Path to save the figure
            num_rounds: Total number of rounds (for validation)
            num_clients: Total number of clients (for inferring attacker_ids if not provided)
            num_attackers: Number of attacker clients (for inferring attacker_ids if not provided)
        """
        rounds = [log['round'] for log in log_data]
        
        # Validate data length
        if num_rounds is not None and len(rounds) != num_rounds:
            print(f"  ⚠️  Warning: Figure 4 - Expected {num_rounds} rounds, got {len(rounds)}")
            if len(rounds) < num_rounds:
                # Pad missing rounds (distances will be handled in the loop)
                expected_rounds = list(range(1, num_rounds + 1))
                missing_rounds = [r for r in expected_rounds if r not in rounds]
                # Add placeholder logs for missing rounds
                if log_data:
                    last_log = log_data[-1].copy()
                    for r in missing_rounds:
                        placeholder_log = last_log.copy()
                        placeholder_log['round'] = r
                        placeholder_log['aggregation'] = last_log.get('aggregation', {}).copy()
                        log_data.append(placeholder_log)
                    rounds = expected_rounds
                print(f"     Padded {len(missing_rounds)} missing rounds")
        
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # Extract Euclidean distances and client info from aggregation logs
        # Build a mapping of client_id -> list of distances over rounds
        client_distances = {}  # {client_id: [dist1, dist2, ...]}
        
        for log in log_data:
            aggregation = log.get('aggregation', {})
            euclidean_distances = aggregation.get('euclidean_distances', [])
            accepted_clients = aggregation.get('accepted_clients', [])
            
            # Map distances to client IDs
            # The euclidean_distances list is ordered by sorted client IDs (from aggregate_updates)
            all_client_ids = sorted(set(accepted_clients))
            
            # Ensure we have the same number of distances as clients
            if len(euclidean_distances) != len(all_client_ids):
                # If mismatch, try to infer from the log structure
                # Distances should match the order of sorted client IDs
                print(f"  ⚠️  Warning: Distance count ({len(euclidean_distances)}) != client count ({len(all_client_ids)}) in round {log.get('round', '?')}")
            
            for i, client_id in enumerate(all_client_ids):
                if client_id not in client_distances:
                    client_distances[client_id] = []
                
                if i < len(euclidean_distances):
                    client_distances[client_id].append(euclidean_distances[i])
                else:
                    # Pad with previous value or 0 if no previous value
                    if len(client_distances[client_id]) > 0:
                        client_distances[client_id].append(client_distances[client_id][-1])
                    else:
                        client_distances[client_id].append(0.0)
        
        # Separate into benign and attacker
        all_ids = sorted(client_distances.keys())
        if attacker_ids is None:
            # Infer attacker_ids from num_clients and num_attackers
            if num_clients is not None and num_attackers is not None:
                attacker_ids_set = set(range(num_clients - num_attackers, num_clients))
            else:
                # Fallback: use all_ids to infer (assume last clients are attackers)
                if num_attackers is not None:
                    attacker_ids_set = set(all_ids[-num_attackers:]) if len(all_ids) >= num_attackers else set()
                else:
                    # Last resort: assume 2 attackers (old behavior)
                    print("  ⚠️  Warning: Could not infer attacker_ids, assuming last 2 clients are attackers")
                    attacker_ids_set = set(all_ids[-2:]) if len(all_ids) >= 2 else set()
        else:
            attacker_ids_set = set(attacker_ids)
        
        benign_clients = [{'id': cid, 'dists': client_distances[cid]} 
                         for cid in all_ids if cid not in attacker_ids_set]
        attacker_clients = [{'id': cid, 'dists': client_distances[cid]} 
                           for cid in all_ids if cid in attacker_ids_set]
        
        # Collect all distance values for adaptive y-axis range (before plotting)
        all_distance_values = []
        
        # Process benign clients - align and collect data
        aligned_benign_data = []
        for client in benign_clients:
            dists = client['dists']
            # Align distances with rounds length
            if len(dists) < len(rounds):
                dists = dists + [dists[-1] if len(dists) > 0 else 0.0] * (len(rounds) - len(dists))
            elif len(dists) > len(rounds):
                dists = dists[:len(rounds)]
            if len(dists) == len(rounds):
                all_distance_values.extend(dists)
                aligned_benign_data.append({'id': client['id'], 'dists': dists})
        
        # Process attacker clients - align and collect data
        aligned_attacker_data = []
        for client in attacker_clients:
            dists = client['dists']
            if len(dists) < len(rounds):
                dists = dists + [dists[-1] if len(dists) > 0 else 0.0] * (len(rounds) - len(dists))
            elif len(dists) > len(rounds):
                dists = dists[:len(rounds)]
            if len(dists) == len(rounds):
                all_distance_values.extend(dists)
                aligned_attacker_data.append({'id': client['id'], 'dists': dists})
        
        # Calculate adaptive y-axis range with padding
        if all_distance_values:
            y_min = min(all_distance_values)
            y_max = max(all_distance_values)
            y_range = y_max - y_min
            
            # Add 10% padding on both sides
            padding = max(y_range * 0.1, 0.01)  # At least 0.01 padding for distances
            y_min_adjusted = max(0.0, y_min - padding)  # Don't go below 0
            y_max_adjusted = y_max + padding
            
            # If range is very small, ensure minimum range
            if y_max_adjusted - y_min_adjusted < y_max * 0.1:
                center = (y_min + y_max) / 2
                range_size = max(y_max * 0.1, 0.01)
                y_min_adjusted = max(0.0, center - range_size / 2)
                y_max_adjusted = center + range_size / 2
        else:
            # Fallback to default range if no data
            y_min_adjusted = 0.0
            y_max_adjusted = 1.0
        
        # Plot benign agents - IEEE style colors (use aligned data)
        for i, client_data in enumerate(aligned_benign_data):
            dists = client_data['dists']
            # Use modulo to cycle through colors and markers if needed
            color = IEEE_COLORS['benign'][i % len(IEEE_COLORS['benign'])]
            marker = IEEE_MARKERS['benign'][i % len(IEEE_MARKERS['benign'])]
            ax.plot(rounds, dists, '-', color=color, linewidth=1.5,
                   marker=marker, markersize=4, markevery=max(1, len(rounds)//15),
                   label=f'Agent {client_data["id"]+1}', zorder=2,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        
        # Plot attacker agents - IEEE style red/orange (use aligned data)
        for i, client_data in enumerate(aligned_attacker_data):
            dists = client_data['dists']
            # Use modulo to cycle through colors and markers if needed
            color = IEEE_COLORS['attacker'][i % len(IEEE_COLORS['attacker'])]
            marker = IEEE_MARKERS['attacker'][i % len(IEEE_MARKERS['attacker'])]
            ax.plot(rounds, dists, '-', color=color, linewidth=1.5,
                   marker=marker, markersize=4, markevery=max(1, len(rounds)//15),
                   label=f'Attacker {client_data["id"]+1}', zorder=2,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        
        # IEEE-style axes (y-axis range already calculated above)
        ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
        ax.set_ylabel('Euclidean Distance', fontsize=11, fontweight='normal')
        ax.set_ylim([y_min_adjusted, y_max_adjusted])  # Adaptive y-axis range
        ax.set_xlim([1, max(rounds) if rounds else 1])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # IEEE-style legend: place inside plot area, use 'best' location to avoid blocking data
        # 'best' automatically finds the best location that minimizes overlap with plot elements
        legend = ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          edgecolor='black', framealpha=1.0, fontsize=9, 
                          ncol=1, columnspacing=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  ✅ Saved Figure 4 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure4_euclidean_distance.png', 
                       dpi=600, bbox_inches='tight')
        plt.close()
    
    def plot_figure6_local_accuracy_by_client_type(self, local_accuracies: Dict[int, List[float]], 
                                                rounds: List[int], 
                                                attacker_ids: List[int],
                                                save_path: Optional[str] = None,
                                                num_clients: Optional[int] = None,
                                                num_attackers: Optional[int] = None):
        """
        Figure 6: Local testing accuracy of federated clients over communication rounds
        (benign vs AugMP / other non-benign roles; client grouping uses attacker_ids from the run).
        
        Args:
            local_accuracies: Dict mapping client_id to list of local accuracies per round
            rounds: List of round numbers
            attacker_ids: List of attacker client IDs
            save_path: Path to save the figure
            num_clients: Total number of clients (for validation)
            num_attackers: Number of attacker clients (for validation)
        """
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # Separate benign and attacker
        benign_accs = {cid: accs for cid, accs in local_accuracies.items() 
                      if cid not in attacker_ids}
        attacker_accs = {cid: accs for cid, accs in local_accuracies.items() 
                        if cid in attacker_ids}
        
        # Plot benign agents - IEEE style
        # Ensure ALL benign clients are plotted with distinct colors and markers
        for i, (client_id, accs) in enumerate(sorted(benign_accs.items())):
            if len(accs) < len(rounds):
                accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
            elif len(accs) > len(rounds):
                accs = accs[:len(rounds)]
            
            if len(accs) == len(rounds):
                # Convert to percentage
                accs_pct = [acc * 100 for acc in accs]
                # Use modulo to cycle through colors and markers if needed
                color = IEEE_COLORS['benign'][i % len(IEEE_COLORS['benign'])]
                marker = IEEE_MARKERS['benign'][i % len(IEEE_MARKERS['benign'])]
                ax.plot(rounds, accs_pct, '-', color=color, linewidth=1.5,
                       marker=marker, markersize=4, markevery=max(1, len(rounds)//20),
                       label=f'Agent {client_id+1}', zorder=2,
                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
            else:
                print(f"  ⚠️  Warning: Benign Client {client_id} - accs length ({len(accs)}) != rounds length ({len(rounds)})")
        
        # Plot attacker agents - IEEE style red/orange
        # Ensure ALL attackers are plotted with distinct colors and markers
        for i, (client_id, accs) in enumerate(sorted(attacker_accs.items())):
            if len(accs) < len(rounds):
                accs = accs + [accs[-1] if len(accs) > 0 else 0.0] * (len(rounds) - len(accs))
            elif len(accs) > len(rounds):
                accs = accs[:len(rounds)]
            
            if len(accs) == len(rounds):
                # Convert to percentage
                accs_pct = [acc * 100 for acc in accs]
                # Use modulo to cycle through colors and markers if needed
                color = IEEE_COLORS['attacker'][i % len(IEEE_COLORS['attacker'])]
                marker = IEEE_MARKERS['attacker'][i % len(IEEE_MARKERS['attacker'])]
                ax.plot(rounds, accs_pct, '-', color=color, linewidth=1.5,
                       marker=marker, markersize=4, markevery=max(1, len(rounds)//20),
                       label=f'Attacker {client_id+1}', zorder=2,
                       markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
            else:
                print(f"  ⚠️  Warning: Attacker Client {client_id} - accs length ({len(accs)}) != rounds length ({len(rounds)})")
        
        # Calculate dynamic y-axis range based on actual data (in percentage)
        all_acc_values = []
        for accs in local_accuracies.values():
            if accs:
                all_acc_values.extend([acc * 100 for acc in accs])
        
        if all_acc_values:
            min_acc = min(all_acc_values)
            max_acc = max(all_acc_values)
            y_min = max(0.0, min_acc - 2)
            y_max = min(100.0, max_acc + 2)
            if y_max - y_min < 10:
                center = (y_min + y_max) / 2
                y_min = max(0.0, center - 5)
                y_max = min(100.0, center + 5)
        else:
            y_min, y_max = 0.0, 100.0
        
        # IEEE-style axes
        ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
        ax.set_ylabel('Local Testing Accuracy (%)', fontsize=11, fontweight='normal')
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([1, max(rounds)])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # IEEE-style legend: place inside plot area, use 'best' location to avoid blocking data
        # 'best' automatically finds the best location that minimizes overlap with plot elements
        legend = ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                          edgecolor='black', framealpha=1.0, fontsize=9, 
                          ncol=1, columnspacing=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  ✅ Saved Figure 6 to: {save_path}")
        else:
            plt.savefig(self.results_dir / 'figure6_local_accuracy_by_client_type.png', 
                       dpi=600, bbox_inches='tight')
        plt.close()
    
    def plot_global_loss(self, log_data: List[Dict], save_path: Optional[str] = None, num_rounds: Optional[int] = None):
        """
        Plot global loss over communication rounds.
        
        Args:
            log_data: List of round logs from server
            save_path: Path to save the figure
            num_rounds: Total number of rounds (for validation)
        """
        rounds = [log['round'] for log in log_data]
        global_losses = [log.get('global_loss', 0.0) for log in log_data]
        
        # Ensure all arrays have the same length
        min_len = min(len(rounds), len(global_losses))
        if min_len == 0:
            print("  ⚠️  Warning: Global Loss - No data to plot")
            return
        
        # Truncate all arrays to the same length
        rounds = rounds[:min_len]
        global_losses = global_losses[:min_len]
        
        # Validate/pad
        if num_rounds is not None and len(rounds) != num_rounds:
            print(f"  ⚠️  Warning: Global Loss - Expected {num_rounds} rounds, got {len(rounds)}")
            expected_rounds = list(range(1, num_rounds + 1))
            if len(rounds) < num_rounds:
                missing = [r for r in expected_rounds if r not in rounds]
                for _ in missing:
                    rounds.append(expected_rounds[len(rounds)])
                    global_losses.append(global_losses[-1] if global_losses else 0.0)
                print(f"     Padded {len(missing)} missing rounds")
        
        fig, ax = plt.subplots(figsize=(6.5, 5))
        
        # IEEE-style: clean, professional appearance
        ax.set_xlabel('Episodes', fontsize=11, fontweight='normal')
        ax.set_ylabel('Global Loss', fontsize=11, fontweight='normal')
        
        # Plot loss line - IEEE style: solid line, clear marker
        ax.plot(rounds, global_losses, '-', color=IEEE_COLORS['global'], 
                linewidth=2, marker='o', markersize=4, markevery=max(1, len(rounds)//20),
                label='Global Loss', zorder=3, markerfacecolor=IEEE_COLORS['global'],
                markeredgecolor='white', markeredgewidth=0.5)
        
        # IEEE-style: subtle grid, clean axes
        if global_losses:
            y_min = max(0.0, min(global_losses) * 0.9)
            y_max = max(global_losses) * 1.1
            ax.set_ylim([y_min, y_max])
        else:
            ax.set_ylim([0.0, 1.0])
        ax.set_xlim([1, max(rounds) if rounds else 1])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # IEEE-style legend: clear, professional
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False,
                 edgecolor='black', framealpha=1.0, fontsize=9)
        
        # No title for IEEE style (usually added in LaTeX)
        plt.tight_layout()
        
        out_path = save_path or (self.results_dir / 'global_loss.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        print(f"  ✅ Saved Global Loss figure to: {out_path}")
        plt.close()
    
    def generate_all_figures(self, server_log_data: List[Dict], 
                            local_accuracies: Optional[Dict[int, List[float]]] = None,
                            attacker_ids: Optional[List[int]] = None,
                            experiment_name: str = "experiment",
                            num_rounds: Optional[int] = None,
                            attack_start_round: Optional[int] = None,
                            num_clients: Optional[int] = None,
                            num_attackers: Optional[int] = None):
        """
        Generate all figures from the paper.
        
        Args:
            server_log_data: List of round logs from server (attack experiment)
            local_accuracies: Dict mapping client_id to list of local accuracies per round (attack experiment)
            attacker_ids: List of attacker client IDs
            experiment_name: Name for output files
            num_rounds: Total number of rounds (from config) - ensures all figures use correct round count
            attack_start_round: Round when attack phase starts
            num_clients: Total number of clients (from config) - used to correctly identify all clients
            num_attackers: Number of attacker clients (from config) - used to correctly identify attackers
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
        
        # Figure 1: Global Accuracy Stability (non-baseline run)
        print("\n📊 Generating Figure 1: Global Accuracy Stability...")
        self.plot_figure3_global_accuracy_stability(
            server_log_data,
            save_path=self.results_dir / f'{experiment_name}_figure1.png',
            num_rounds=num_rounds
        )
        
        # Figure 2: Cosine Similarity (non-baseline run)
        print("📊 Generating Figure 2: Cosine Similarity...")
        self.plot_figure4_cosine_similarity(
            server_log_data,
            attacker_ids=attacker_ids,
            save_path=self.results_dir / f'{experiment_name}_figure2.png',
            num_rounds=num_rounds,
            num_clients=num_clients,
            num_attackers=num_attackers
        )
        
        # Figure 3: Local accuracy by client
        if local_accuracies is not None:
            print("📊 Generating Figure 3: Local accuracy by client...")
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
            
            self.plot_figure6_local_accuracy_by_client_type(
                aligned_local_accs, rounds, attacker_ids,
                save_path=self.results_dir / f'{experiment_name}_figure3.png',
                num_clients=num_clients,
                num_attackers=num_attackers
            )
        else:
            print("  ⚠️  Figure 3 skipped: Local accuracies not available.")
            print("     Local accuracies are automatically tracked during training.")
        
        # Figure 4: Euclidean Distance (non-baseline run)
        print("📊 Generating Figure 4: Euclidean Distance...")
        self.plot_figure4_euclidean_distance(
            server_log_data,
            attacker_ids=attacker_ids,
            save_path=self.results_dir / f'{experiment_name}_figure4.png',
            num_rounds=num_rounds,
            num_clients=num_clients,
            num_attackers=num_attackers
        )
        
        # Figure 5: Global Loss (new figure)
        print("📊 Generating Figure 5: Global Loss...")
        self.plot_global_loss(
            server_log_data,
            save_path=self.results_dir / f'{experiment_name}_figure5.png',
            num_rounds=num_rounds
        )
        
        print("\n✅ All available figures generated successfully!")
        print(f"   Output directory: {self.results_dir}")

