# main.py
# This script sets up and runs a federated learning experiment with a progressive GRMP attack.

import torch
import torch.nn as nn
import numpy as np
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from typing import Dict

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server
from visualization import ExperimentVisualizer

warnings.filterwarnings('ignore')

# Initialize experiment components
def setup_experiment(config):
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print(f"Setting up Experiment: {config['experiment_name']}")
    print("=" * 50)

    # 1. Initialize Data Manager
    # Note: dataset_size_limit=None uses FULL AG News dataset (~120K train, ~7.6K test) per paper
    # Set dataset_size_limit to a positive int (e.g., 30000) only for faster experimentation
    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'],
        test_seed=config['seed'],
        dataset_size_limit=config['dataset_size_limit'],
        batch_size=config['batch_size'],
        test_batch_size=config['test_batch_size']
    )

    # 2. Partition data among clients (Non-IID distribution per paper)
    # Per paper: "heterogeneous IoA system" with heterogeneous data distributions
    print("\nPartitioning data (Non-IID distribution)...")
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    
    # Non-IID distribution: Use Dirichlet distribution to create heterogeneous data
    # Each client gets data with different label distributions
    dirichlet_alpha = config['dirichlet_alpha']
    num_labels = 4
    client_indices = {i: [] for i in range(config['num_clients'])}
    
    # Partition data by label first
    label_indices = {label: [] for label in range(num_labels)}
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Assign samples to clients using Dirichlet distribution for non-IID
    for label in range(num_labels):
        label_list = np.array(label_indices[label])
        rng.shuffle(label_list)
        
        # Generate proportions for each client using Dirichlet distribution
        # Lower dirichlet_alpha creates more heterogeneous (non-IID) distribution
        proportions = rng.dirichlet([dirichlet_alpha] * config['num_clients'])
        proportions = np.cumsum(proportions)
        proportions[-1] = 1.0  # Ensure last is exactly 1.0
        
        # Assign samples based on proportions
        start_idx = 0
        for client_id in range(config['num_clients']):
            end_idx = int(len(label_list) * proportions[client_id])
            client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    # Shuffle within each client to mix labels (but distribution remains non-IID)
    for client_id in range(config['num_clients']):
        client_list = np.array(client_indices[client_id])
        rng.shuffle(client_list)
        client_indices[client_id] = client_list.tolist()
    
    # Print distribution statistics
    print(f"  Non-IID distribution (Dirichlet alpha={dirichlet_alpha})")
    num_attackers = config.get('num_attackers', 0)  # Allow 0 attackers for baseline experiment
    num_benign = config['num_clients'] - num_attackers
    for client_id in range(config['num_clients']):  # Show all clients
        client_labels = [labels[idx] for idx in client_indices[client_id]]
        label_counts = {l: client_labels.count(l) for l in range(num_labels)}
        total = len(client_indices[client_id])
        if total > 0:
            dist_str = ", ".join([f"Label {l}: {label_counts[l]/total:.1%}" for l in range(num_labels)])
            client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
            print(f"    Client {client_id} ({client_type}): {total} samples ({dist_str})")
        else:
            client_type = "BENIGN" if client_id < num_benign else "ATTACKER"
            print(f"    Client {client_id} ({client_type}): 0 samples WARNING: No data assigned!")

    # 3. Get global test loader
    test_loader = data_manager.get_test_loader()

    # 4. Initialize Global Model
    use_lora = config.get('use_lora', False)
    if use_lora:
        print("Initializing global model (DistilBERT) with LoRA...")
        global_model = NewsClassifierModel(
            model_name=config.get('model_name', 'distilbert-base-uncased'),
            num_labels=config.get('num_labels', 4),
            use_lora=True,
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            lora_target_modules=config.get('lora_target_modules', None)
        )
    else:
        print("Initializing global model (DistilBERT) [Full Fine-tuning]...")
        global_model = NewsClassifierModel(
            model_name=config.get('model_name', 'distilbert-base-uncased'),
            num_labels=config.get('num_labels', 4),
            use_lora=False
        )

    # 5. Initialize Server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        defense_threshold=config['defense_threshold'],
        total_rounds=config['num_rounds'],
        server_lr=config['server_lr'],
        tolerance_factor=config['tolerance_factor'],
        d_T=config['d_T'],
        gamma=config['gamma'],
        similarity_alpha=config['similarity_alpha'],
        defense_high_rejection_threshold=config['defense_high_rejection_threshold'],
        defense_threshold_decay=config['defense_threshold_decay']
    )

    # 6. Create Clients
    print("\nCreating federated learning clients...")
    num_attackers = config.get('num_attackers', 0)  # Allow 0 attackers for baseline experiment
    
    for client_id in range(config['num_clients']):
        # Determine if benign or attacker
        # Logic: Last 'num_attackers' clients are attackers
        # If num_attackers=0, all clients are benign (baseline experiment)
        if client_id < (config['num_clients'] - num_attackers):
            # --- Benign Client ---
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]
            
            # Create static dataloader for benign client
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer)
            client_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            print(f"  Client {client_id}: BENIGN ({len(client_indices[client_id])} samples)")
            
            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config['alpha'],
                data_indices=client_indices[client_id],
                grad_clip_norm=config['grad_clip_norm']
            )
        else:
            # --- Attacker Client ---
            print(f"  Client {client_id}: ATTACKER (VGAE Enabled)")
            # Use actual assigned data size for claimed_data_size (for fair weighted aggregation)
            # Note: Attackers are data-agnostic (don't use data for training), but use assigned
            # data size for aggregation weight to maintain realistic attack scenario
            actual_data_size = len(client_indices[client_id])
            # Allow config override if attacker wants to claim different size (for attack experiments)
            config_claimed = config.get('attacker_claimed_data_size', None)
            if config_claimed is None:
                # Use actual assigned data size (recommended for realistic scenario)
                claimed_data_size = actual_data_size
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
            else:
                # Override with config value (for attack experiments)
                claimed_data_size = config_claimed
                print(f"    WARNING: Override: Claimed data size D'_j(t): {claimed_data_size} (actual: {actual_data_size})")
            
            client = AttackerClient(
                client_id=client_id,
                model=global_model,
                data_manager=data_manager,
                data_indices=client_indices[client_id],
                lr=config['client_lr'],
                local_epochs=config['local_epochs'],
                alpha=config['alpha'],
                dim_reduction_size=config['dim_reduction_size'],
                vgae_epochs=config['vgae_epochs'],
                vgae_lr=config['vgae_lr'],
                graph_threshold=config['graph_threshold'],
                proxy_step=config['proxy_step'],
                claimed_data_size=claimed_data_size,
                proxy_sample_size=config['proxy_sample_size'],
                proxy_max_batches_opt=config['proxy_max_batches_opt'],
                proxy_max_batches_eval=config['proxy_max_batches_eval'],
                vgae_hidden_dim=config['vgae_hidden_dim'],
                vgae_latent_dim=config['vgae_latent_dim'],
                vgae_dropout=config['vgae_dropout'],
                proxy_steps=config['proxy_steps'],
                gsp_perturbation_scale=config['gsp_perturbation_scale'],
                opt_init_perturbation_scale=config['opt_init_perturbation_scale'],
                grad_clip_norm=config['grad_clip_norm']
            )

        server.register_client(client)
    
    return server, results_dir

# Run the experiment
def run_experiment(config):
    server, results_dir = setup_experiment(config)

    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_clean = server.evaluate()
    print(f"Initial Performance - Clean Accuracy: {initial_clean:.4f}")

    print("\n" + "=" * 50)
    print("Starting Federated Learning Rounds")
    print("=" * 50)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'acc_diff': [],
        'rejection_rate': [],
        'agg_update_norm': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['rejection_rate'].append(round_log['defense'].get('rejection_rate', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['defense'].get('aggregated_update_norm', 0.0))
            
            # Memory cleanup after each round
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    results_data = {
        'config': config,
        'results': server.log_data,
        'progressive_metrics': progressive_metrics,
        'local_accuracies': server.history['local_accuracies']  # Include local accuracies
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    if config.get('generate_plots', True):
        print("\n" + "=" * 60)
        print("Generating Visualization Plots")
        print("=" * 60)
        
        visualizer = ExperimentVisualizer(results_dir=results_dir)
        
        # Extract attacker IDs
        attacker_ids = [client.client_id for client in server.clients 
                       if getattr(client, 'is_attacker', False)]
        
        # Generate all figures
        # Check if baseline experiment results exist for Figure 5
        baseline_path = results_dir / f"{config['experiment_name']}_no_attack_results.json"
        baseline_local_accs = None
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_local_accs = baseline_data.get('local_accuracies', None)
                    if baseline_local_accs:
                        print("  Found baseline experiment data for Figure 5")
            except Exception as e:
                print(f"  WARNING: Could not load baseline data: {e}")
        
        visualizer.generate_all_figures(
            server_log_data=server.log_data,
            local_accuracies=server.history['local_accuracies'],
            attacker_ids=attacker_ids,
            experiment_name=config['experiment_name'],
            baseline_local_accuracies=baseline_local_accs,
            num_rounds=config['num_rounds'],
            attack_start_round=config['attack_start_round']
        )
    
    return server.log_data, progressive_metrics

# Simple analysis
def analyze_results(metrics):
    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    
    rounds = metrics['rounds']
    if not rounds:
        print("No rounds completed.")
        return

    clean = metrics['clean_acc']

    print(f"Total Rounds: {len(rounds)}")
    print(f"Final Clean Accuracy: {clean[-1]:.4f}")
    if len(clean) > 1:
        print(f"Best Clean Accuracy: {max(clean):.4f}")
        print(f"Accuracy Change: {clean[-1] - clean[0]:+.4f}")

def run_no_attack_experiment(config_base: Dict) -> Dict:
    """
    Run a baseline experiment WITHOUT any attackers (for Figure 5 comparison).
        
    Args:
        config_base: Base configuration dictionary
        
    Returns:
        Dictionary with experiment results
    """
    # Create a copy of config with no attackers
    config = config_base.copy()
    config['experiment_name'] = config_base.get('experiment_name', 'baseline') + '_no_attack'
    num_benign_in_attack = config_base['num_clients'] - config_base.get('num_attackers', 0)
    if 'num_benign_clients' in config_base and config_base['num_benign_clients'] is not None:
        config['num_clients'] = config_base['num_benign_clients']
    else:
        config['num_clients'] = num_benign_in_attack
    
    config['num_attackers'] = 0  # No attackers
    
    print("\n" + "=" * 60)
    print("Running BASELINE Experiment (NO ATTACK)")
    print("=" * 60)
    print(f"Baseline will use {config['num_clients']} benign clients")
    print(f"(matching {num_benign_in_attack} benign clients in attack experiment)")
    print("This ensures fair comparison by controlling variables.")
    print("=" * 60)
    
    return run_experiment(config)


def main():
    config = {
        # ========== Experiment Configuration ==========
        'experiment_name': 'vgae_grmp_attack',  # Name for result files and logs
        'seed': 42,  # Random seed for reproducibility (int)
        
        # ========== Federated Learning Setup ==========
        'num_clients': 6,  # Total number of federated learning clients (int)
        'num_attackers': 2,  # Number of attacker clients (int, must be < num_clients)
        'num_benign_clients': None,  # Optional: Explicit number of benign clients for baseline experiment
                                     # If None, baseline will use (num_clients - num_attackers) to ensure fair comparison
                                     # If set, baseline experiment will use exactly this many benign clients
        'num_rounds': 50,  # Total number of federated learning rounds (int)
        
        # ========== Training Mode Configuration ==========
        'use_lora': False,  # True for LoRA fine-tuning, False for full fine-tuning
        # LoRA parameters (only used when use_lora=True)
        'lora_r': 16,  # LoRA rank (controls the rank of low-rank matrices)
        'lora_alpha': 32,  # LoRA alpha (scaling factor, typically 2*r)
        'lora_dropout': 0.1,  # LoRA dropout rate
        'lora_target_modules': None,  # None = use default for DistilBERT (["q_lin", "k_lin", "v_lin", "out_lin"])
        # Model configuration
        'model_name': 'distilbert-base-uncased',  # Model name or path
        'num_labels': 4,  # Number of classification labels
        
        # ========== Training Hyperparameters ==========
        'client_lr': 2e-5,  # Learning rate for local client training (float)
        'server_lr': 1.0,  # Server learning rate for model aggregation (fixed at 1.0)
        'batch_size': 128,  # Batch size for local training (int)
        'test_batch_size': 512,  # Batch size for test/validation data loaders (int)
        
        'local_epochs': 5,  # Number of local training epochs per round (int, per paper Section IV)
        'alpha': 0.05,  # Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (float)
        
        # ========== Data Distribution ==========
        'dirichlet_alpha': 0.3,  # Make data less extreme non-IID (higher alpha = more balanced)
        # 'dataset_size_limit': None,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)
        'dataset_size_limit': 20000,  # Limit dataset size for faster experimentation (None = use FULL AG News dataset per paper, int = limit training samples)

        # ========== Attack Configuration ==========
        'attack_start_round': 0,  # Round when attack phase starts (int, now all rounds use complete poisoning)
        
        # ========== Formula 4 Constraint Parameters ==========
        'd_T': 0.6,  # Tighter proximity for camouflage (closer to global)
        'gamma': 5.0,  # Tighter aggregation distance budget
        
        # ========== VGAE Training Parameters ==========
        # Reference paper: input_dim=5, hidden1_dim=32, hidden2_dim=16, num_epoch=10, lr=0.01
        # Note: dim_reduction_size should be <= total trainable parameters
        # - Full fine-tuning: ~67M parameters, dim_reduction_size=10000 is fine
        # - LoRA (r=16): ~0.5-1M parameters, dim_reduction_size will be auto-adjusted if > LoRA params
        # Auto-adjustment: If dim_reduction_size > actual LoRA params, it will be set to 80% of LoRA params
        'dim_reduction_size': 10000,  # Reduced dimensionality of LLM parameters (auto-adjusted for LoRA if needed)
        'vgae_epochs': 10,  # Number of epochs for VGAE training (reference: 10)
        'vgae_lr': 0.01,  # Learning rate for VGAE optimizer (reference: 0.01)
        'vgae_hidden_dim': 32,  # VGAE hidden layer dimension (per paper: hidden1_dim=32)
        'vgae_latent_dim': 16,  # VGAE latent space dimension (per paper: hidden2_dim=16)
        'vgae_dropout': 0.0,  # VGAE dropout rate (float, 0.0-1.0)
        
        # ========== Attack Optimization Parameters ==========
        'proxy_step': 0.1,  # Step size for gradient-free ascent toward global-loss proxy
        'proxy_steps': 20,  # Number of optimization steps for attack objective (int)
        'gsp_perturbation_scale': 0.01,  # Perturbation scale for GSP attack diversity (float)
        'opt_init_perturbation_scale': 0.001,  # Perturbation scale for optimization initialization (float)
        'grad_clip_norm': 1.0,  # Gradient clipping norm for training stability (float)
        # 'attacker_claimed_data_size': None,  # If None, uses actual assigned data size (recommended for realistic scenario)
        # If set to a value, overrides actual data size (for attack experiments where attacker claims more data)
        'attacker_claimed_data_size': None,  # None = use actual assigned data size (recommended)
        
        # ========== Proxy Loss Estimation Parameters ==========
        'proxy_sample_size': 512,  # Number of samples in proxy dataset for F(w'_g) estimation (int)
                                   # Increased from 128 to 512 for better accuracy (4 batches with test_batch_size=128)
        'proxy_max_batches_opt': 2,  # Max batches for proxy loss in optimization loop (int)
                                      # Used during gradient-based optimization (20 steps per round)
        'proxy_max_batches_eval': 4,  # Max batches for proxy loss in final evaluation (int)
                                       # Used for final attack objective logging (1 call per round)
        
        # ========== Graph Construction Parameters ==========
        'graph_threshold': 0.5,  # Threshold for graph adjacency matrix binarization in VGAE (float, 0.0-1.0)
        
        # ========== Defense Mechanism Parameters ==========
        'defense_threshold': 0,  # Base threshold for defense mechanism (float, lower = more strict)
        'tolerance_factor': 3.0,  # Tolerance factor for defense mechanism (float, higher = more lenient)
        'similarity_alpha': 0.5,  # Weight for pairwise similarities in mixed similarity computation (float, 0.0-1.0)
        'defense_high_rejection_threshold': 0.4,  # High rejection rate threshold for adaptive defense (float, 0.0-1.0)
        'defense_threshold_decay': 0.9,  # Decay factor for defense threshold when high rejection detected (float, 0.0-1.0)
        
        # ========== Visualization ==========
        'generate_plots': True,  # Whether to generate visualization plots (bool)
        'run_both_experiments': False,  # Set to True to run baseline + attack (for Figure 5)
        'run_attack_only': False,  # Set to True to only run attack experiment
    }



    # Option 1: Run attack experiment only
    if config.get('run_attack_only', False):
        print("Running GRMP Attack with VGAE...")
        results, metrics = run_experiment(config)
        analyze_results(metrics)
    
    # Option 2: Run both baseline (no attack) and attack experiments
    elif config.get('run_both_experiments', False):
        print("\n" + "=" * 60)
        print("Running COMPLETE Experiment Suite")
        print("=" * 60)
        print("This will run:")
        print("  1. Baseline experiment (no attack) - for Figure 5")
        print("  2. Attack experiment (with GRMP) - for Figures 3, 4, 6")
        print("=" * 60)
        
        # Run baseline (no attack) experiment
        baseline_results, baseline_metrics = run_no_attack_experiment(config)
        
        # Run attack experiment
        print("\n" + "=" * 60)
        print("Now running ATTACK experiment...")
        print("=" * 60)
        attack_results, attack_metrics = run_experiment(config)
        
        # Generate combined visualizations
        if config.get('generate_plots', True):
            from visualization import ExperimentVisualizer
            from pathlib import Path
            
            results_dir = Path("results")
            visualizer = ExperimentVisualizer(results_dir=results_dir)
            
            # Extract attacker IDs from attack experiment
            # (We need to reload the server or pass it differently)
            # For now, we'll use the config to determine attackers
            num_clients = config['num_clients']
            num_attackers = config['num_attackers']
            attacker_ids = list(range(num_clients - num_attackers, num_clients))
            
            print("\n" + "=" * 60)
            print("Generating Combined Visualizations")
            print("=" * 60)
            
            # Load baseline results for Figure 5
            baseline_path = results_dir / f"{config['experiment_name']}_no_attack_results.json"
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_local_accs = baseline_data.get('local_accuracies', {})
                    
                    # Generate Figure 5 from baseline
                    print("\nGenerating Figure 5: Local Accuracy (No Attack)...")
                    baseline_rounds = list(range(1, len(baseline_results) + 1))
                    visualizer.plot_figure5_local_accuracy_no_attack(
                        baseline_local_accs, baseline_rounds,
                        save_path=results_dir / f"{config['experiment_name']}_figure5.png"
                    )
            
            # Generate attack experiment figures (3, 4, 6)
            # Note: We need server object to get local_accuracies, so we'll regenerate
            # For now, load from saved results
            attack_path = results_dir / f"{config['experiment_name']}_results.json"
            if attack_path.exists():
                with open(attack_path, 'r') as f:
                    attack_data = json.load(f)
                    attack_local_accs = attack_data.get('local_accuracies', {})
                    
                    # Generate Figures 3, 4, 6
                    visualizer.generate_all_figures(
                        server_log_data=attack_results,
                        local_accuracies=attack_local_accs,
                        attacker_ids=attacker_ids,
                        experiment_name=config['experiment_name']
                    )
        
        analyze_results(attack_metrics)
    
    # Option 3: Default - run attack experiment only
    else:
        print("Running GRMP Attack with VGAE...")
        results, metrics = run_experiment(config)
        analyze_results(metrics)
        
        # Note: Figure 5 requires a separate baseline experiment
        # Set 'run_both_experiments': True to generate all figures
        if config.get('generate_plots', True):
            print("\n" + "=" * 60)
            print("NOTE: Figure 5 (No Attack) requires a baseline experiment.")
            print("To generate Figure 5, set 'run_both_experiments': True in config")
            print("or run a separate experiment with 'num_attackers': 0")
            print("=" * 60)

if __name__ == "__main__":
    main()