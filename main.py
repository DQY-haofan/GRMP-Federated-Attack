# main.py
# This script sets up and runs a federated learning experiment (benign fine-tuning and optional AugMP clients).

import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from typing import Dict, List, Optional, Sequence

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server
from visualization import ExperimentVisualizer
from fed_checkpoint import save_global_model_checkpoint

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
    # dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers' — select dataset; num_labels and max_length must match (see config below)
    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'],
        test_seed=config['seed'],
        dataset_size_limit=config['dataset_size_limit'],
        batch_size=config['batch_size'],
        test_batch_size=config['test_batch_size'],
        model_name=config.get('model_name', 'distilbert-base-uncased'),
        max_length=config.get('max_length', 128),
        dataset=config.get('dataset', 'ag_news')
    )

    # 2. Partition data among clients
    # Supports both IID and Non-IID distributions based on config
    data_distribution = config.get('data_distribution', 'non-iid').lower()
    indices = np.arange(len(data_manager.train_texts))
    labels = np.array(data_manager.train_labels)
    num_labels = config.get('num_labels', 4)
    num_clients = config['num_clients']
    num_attackers = config.get('num_attackers', 0)
    num_benign = num_clients - num_attackers
    
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    
    client_indices = {i: [] for i in range(num_clients)}
    
    if data_distribution == 'iid':
        # ========== IID Distribution: Uniform Random Partition ==========
        # Each client gets approximately equal number of samples with similar label distribution
        print("\nPartitioning data (IID distribution)...")
        
        # Shuffle all indices
        all_indices = indices.copy()
        rng.shuffle(all_indices)
        
        # Calculate samples per client (approximately equal)
        total_samples = len(all_indices)
        base_samples = total_samples // num_clients
        remainder = total_samples % num_clients
        
        # Assign samples to each client
        start_idx = 0
        for client_id in range(num_clients):
            # First 'remainder' clients get one extra sample
            extra = 1 if client_id < remainder else 0
            end_idx = start_idx + base_samples + extra
            client_indices[client_id] = all_indices[start_idx:end_idx].tolist()
            start_idx = end_idx
        
        # Print distribution statistics
        print(f"  IID distribution (uniform random partition)")
        for client_id in range(num_clients):
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

        if num_benign < num_clients:
            print("\n  [Note] Attackers are assigned only data *quantities* (sizes) for the experimental setup. "
                  "In reality, attackers do NOT perform local training and do NOT use these local data "
                  "(dataset-free). They also do NOT access other local agents' data.")
    
    else:
        # ========== Non-IID Distribution: Dirichlet-based Partition ==========
        # Per paper: "heterogeneous IoA system" with heterogeneous data distributions
        print("\nPartitioning data (Non-IID distribution)...")
        
        # Use Dirichlet distribution to create heterogeneous data
        # Each client gets data with different label distributions
        dirichlet_alpha = config['dirichlet_alpha']
        
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
            proportions = rng.dirichlet([dirichlet_alpha] * num_clients)
            proportions = np.cumsum(proportions)
            proportions[-1] = 1.0  # Ensure last is exactly 1.0
            
            # Assign samples based on proportions
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = int(len(label_list) * proportions[client_id])
                client_indices[client_id].extend(label_list[start_idx:end_idx].tolist())
                start_idx = end_idx
        
        # Shuffle within each client to mix labels (but distribution remains non-IID)
        for client_id in range(num_clients):
            client_list = np.array(client_indices[client_id])
            rng.shuffle(client_list)
            client_indices[client_id] = client_list.tolist()
        
        # Print distribution statistics
        print(f"  Non-IID distribution (Dirichlet alpha={dirichlet_alpha})")
        for client_id in range(num_clients):
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

        # Clarification: attackers are dataset-free
        if num_benign < num_clients:
            print("\n  [Note] Attackers are assigned only data *quantities* (sizes) following the non-IID distribution, "
                  "for experimental setup. In reality, attackers do NOT perform local training and do NOT use "
                  "these local data (dataset-free). They also do NOT access other local agents' data.")

    # 3. Get global test loader
    test_loader = data_manager.get_test_loader()

    # 4. Initialize Global Model
    use_lora = config.get('use_lora', False)
    model_name = config.get('model_name', 'distilbert-base-uncased')
    if use_lora:
        print(f"Initializing global model ({model_name}) with LoRA...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=True,
            lora_r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            lora_dropout=config.get('lora_dropout', 0.1),
            lora_target_modules=config.get('lora_target_modules', None)
        )
    else:
        print(f"Initializing global model ({model_name}) [Full Fine-tuning]...")
        global_model = NewsClassifierModel(
            model_name=model_name,
            num_labels=config.get('num_labels', 4),
            use_lora=False
        )

    # 5. Initialize Server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        total_rounds=config['num_rounds'],
        server_lr=config['server_lr'],
        dist_bound=config.get('dist_bound', config.get('d_T', 0.5)),  # Renamed from d_T
        similarity_mode=config.get('server_similarity_mode', 'local_vs_global')
    )
    # Manual cosine similarity bounds (None = use benign min/max)
    server.sim_bound_low = config.get('sim_bound_low', None)
    server.sim_bound_up = config.get('sim_bound_up', None)

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
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer, 
                                  max_length=config.get('max_length', 128))
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
            attack_method = config.get('attack_method', 'AugMP')
            if attack_method == 'GRMP':
                attack_method = 'AugMP'  # legacy config alias for proposed method
            
            # Use actual assigned data size for claimed_data_size (for fair weighted aggregation)
            # Note: Attackers are data-agnostic (don't use data for training), but use assigned
            # data size for aggregation weight to maintain realistic attack scenario
            actual_data_size = len(client_indices[client_id])
            # Allow config override if attacker wants to claim different size (for attack experiments)
            config_claimed = config.get('attacker_claimed_data_size', None)
            if config_claimed is None:
                # Use actual assigned data size (recommended for realistic scenario)
                claimed_data_size = actual_data_size
            else:
                # Override with config value (for attack experiments)
                claimed_data_size = config_claimed
            
            # Create attacker based on attack_method
            if attack_method == 'ALIE':
                # ========== ALIE Attack Client ==========
                from attack_baseline_alie import ALIEAttackerClient
                print(f"  Client {client_id}: ATTACKER (ALIE Attack)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                
                # Get ALIE-specific parameters
                alie_z_max = config.get('alie_z_max', None)
                alie_attack_start_round = config.get('alie_attack_start_round', None)
                
                client = ALIEAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    num_clients=config['num_clients'],
                    num_attackers=config['num_attackers'],
                    z_max=alie_z_max,
                    attack_start_round=alie_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0)
                )
            elif attack_method == 'SignFlipping':
                # ========== Sign-Flipping Attack Client (ICML '18: g^byz = -scale * g_own) ==========
                from attack_baseline_sign_flipping import SignFlippingAttackerClient
                print(f"  Client {client_id}: ATTACKER (Sign-Flipping Attack, ICML '18)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                # Build DataLoader for attacker so it can compute g_own (same as benign client)
                client_texts_sf = [data_manager.train_texts[i] for i in client_indices[client_id]]
                client_labels_sf = [data_manager.train_labels[i] for i in client_indices[client_id]]
                dataset_sf = NewsDataset(client_texts_sf, client_labels_sf, data_manager.tokenizer,
                                         max_length=config.get('max_length', 128))
                client_loader_sf = DataLoader(dataset_sf, batch_size=config['batch_size'], shuffle=True)
                sign_flip_scale = config.get('sign_flip_scale', 10.0)
                sign_flip_attack_start_round = config.get('sign_flip_attack_start_round', None)
                client = SignFlippingAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    data_loader=client_loader_sf,
                    sign_flip_scale=sign_flip_scale,
                    attack_start_round=sign_flip_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0)
                )
            elif attack_method == 'Gaussian':
                # ========== Gaussian (Random Model Poisoning) Attack - USENIX Security '20 ==========
                from attack_baseline_gaussian import GaussianAttackerClient
                print(f"  Client {client_id}: ATTACKER (Gaussian Attack, USENIX Security '20)")
                print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                gaussian_attack_start_round = config.get('gaussian_attack_start_round', None)
                gaussian_std_scale = config.get('gaussian_std_scale', 1.0)
                if gaussian_std_scale != 1.0:
                    print(f"    Gaussian std_scale: {gaussian_std_scale} (noise range expanded for FedAvg)")
                client = GaussianAttackerClient(
                    client_id=client_id,
                    model=global_model,
                    data_manager=data_manager,
                    data_indices=client_indices[client_id],
                    lr=config['client_lr'],
                    local_epochs=config['local_epochs'],
                    alpha=config['alpha'],
                    attack_start_round=gaussian_attack_start_round,
                    claimed_data_size=claimed_data_size,
                    grad_clip_norm=config.get('grad_clip_norm', 1.0),
                    gaussian_std_scale=gaussian_std_scale
                )
            else:
                # ========== AugMP client (default proposed method; legacy attack_method='GRMP' normalized above) ==========
                print(f"  Client {client_id}: AugMP participant (VGAE enabled)")
                if config_claimed is None:
                    print(f"    Claimed data size D'_j(t): {claimed_data_size} (matches assigned data)")
                else:
                    print(f"    WARNING: Override: Claimed data size D'_j(t): {claimed_data_size} (actual: {actual_data_size})")
                
                use_proxy = config.get('attacker_use_proxy_data', True)
                if not use_proxy:
                    print(f"    Attacker proxy data disabled (attacker_use_proxy_data=False); no dataset access.")
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
                vgae_kl_weight=config['vgae_kl_weight'],
                proxy_steps=config['proxy_steps'],
                grad_clip_norm=config['grad_clip_norm'],
                proxy_grad_clip_norm=config.get('attacker_proxy_grad_clip_norm', 1.0),
                early_stop_constraint_stability_steps=config.get('early_stop_constraint_stability_steps', 3),
                use_proxy_data=use_proxy
            )
            
            # Set Lagrangian Dual parameters (if using)
            if config.get('use_lagrangian_dual', False):
                client.set_lagrangian_params(
                    use_lagrangian_dual=config['use_lagrangian_dual'],
                    lambda_dist_init=config.get('lambda_dist_init', config.get('lambda_init', 0.1)),
                    lambda_dist_lr=config.get('lambda_dist_lr', config.get('lambda_lr', 0.01)),
                    use_cosine_similarity_constraint=config.get('use_cosine_similarity_constraint', False),
                    use_pairwise_similarity_in_constraint=config.get('use_pairwise_similarity_in_constraint', False),
                    lambda_sim_low_init=config.get('lambda_sim_low_init', config.get('lambda_sim_init', 0.1)),
                    lambda_sim_up_init=config.get('lambda_sim_up_init', config.get('lambda_sim_init', 0.1)),
                    lambda_sim_low_lr=config.get('lambda_sim_low_lr', config.get('lambda_sim_lr', 0.01)),
                    lambda_sim_up_lr=config.get('lambda_sim_up_lr', config.get('lambda_sim_lr', 0.01)),
                    # ========== Augmented Lagrangian (ALM) parameters ==========
                    use_augmented_lagrangian=config.get('use_augmented_lagrangian', False),
                    lambda_update_mode=config.get('lambda_update_mode', 'classic'),
                    rho_dist_init=config.get('rho_dist_init', 1.0),
                    rho_sim_low_init=config.get('rho_sim_low_init', 1.0),
                    rho_sim_up_init=config.get('rho_sim_up_init', 1.0),
                    rho_adaptive=config.get('rho_adaptive', True),
                    rho_theta=config.get('rho_theta', 0.5),
                    rho_increase_factor=config.get('rho_increase_factor', 2.0),
                    rho_min=config.get('rho_min', 1e-3),
                    rho_max=config.get('rho_max', 1e3),
                )
                print(f"    Lagrangian Dual enabled: λ_dist(1)={config.get('lambda_dist_init', config.get('lambda_init', 0.1))}")
            else:
                print(f"    Using hard constraint mechanism (Lagrangian Dual disabled)")

        server.register_client(client)
    
    return server, results_dir


def run_downstream_task2_if_configured(config: Dict, results_dir: Path) -> None:
    """
    Optionally run Task 2 (run_downstream_generation.py) after FL when checkpoint exists.
    Controlled by config['run_downstream_after_fl'].
    """
    if not config.get("run_downstream_after_fl", False):
        return

    ckpt_dir = results_dir / config.get("global_checkpoint_subdir", "global_checkpoint")
    pt_file = ckpt_dir / "global_model.pt"
    if not pt_file.is_file():
        print(
            f"\n⚠️  Task 2 skipped: no checkpoint at {pt_file}. "
            "Set save_global_checkpoint=True and complete training, or run run_downstream_generation.py manually."
        )
        return

    repo_root = Path(__file__).resolve().parent
    raw = config.get("downstream_probes", "data/AG News Datasets/ag_news_business_30.json")
    probes = Path(raw)
    if not probes.is_absolute():
        probes = repo_root / probes
    if not probes.is_file():
        for alt in (
            repo_root / "data" / "AG News Datasets" / Path(raw).name,
            repo_root / "data" / Path(raw).name,
        ):
            if alt.is_file():
                probes = alt
                break
    if not probes.is_file():
        print(f"\n⚠️  Task 2 skipped: probes file not found: {probes}")
        return

    out_raw = config.get("downstream_output")
    if out_raw:
        out_path = Path(out_raw)
        if not out_path.is_absolute():
            out_path = results_dir / out_path
    else:
        out_path = results_dir / f"{config.get('experiment_name', 'experiment')}_downstream_gen.jsonl"

    device = config.get("downstream_device")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extra: Sequence[str] = config.get("downstream_cli_args") or []
    if isinstance(extra, str):
        extra = [extra]

    cmd: List[str] = [
        sys.executable,
        "run_downstream_generation.py",
        "--checkpoint",
        str(ckpt_dir),
        "--probes",
        str(probes),
        "--output",
        str(out_path),
        "--device",
        str(device),
    ]
    cmd.extend(str(x) for x in extra)

    print("\n" + "=" * 60)
    print("Task 2: downstream generation (run_downstream_generation.py)")
    print("=" * 60)
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
    if proc.returncode != 0:
        print(f"\n⚠️  Task 2 exited with code {proc.returncode}")
    else:
        print(f"\nTask 2 finished; JSONL: {out_path}")


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
        'agg_update_norm': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['acc_diff'].append(round_log.get('acc_diff', 0.0))
            progressive_metrics['agg_update_norm'].append(round_log['aggregation'].get('aggregated_update_norm', 0.0))
            
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

    save_global_model_checkpoint(server, config, results_dir)

    run_downstream_task2_if_configured(config, results_dir)

    # Print detailed statistics for data collection
    attacker_ids = [client.client_id for client in server.clients 
                   if getattr(client, 'is_attacker', False)]
    print_detailed_statistics(server.log_data, progressive_metrics, 
                            server.history['local_accuracies'], attacker_ids, 
                            config['experiment_name'], results_dir)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)
    
    visualizer = ExperimentVisualizer(results_dir=results_dir)
    
    # Generate all figures
    visualizer.generate_all_figures(
        server_log_data=server.log_data,
        local_accuracies=server.history['local_accuracies'],
        attacker_ids=attacker_ids,
        experiment_name=config['experiment_name'],
        num_rounds=config['num_rounds'],
        attack_start_round=config['attack_start_round'],
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers']
    )
    
    return server.log_data, progressive_metrics

# Detailed statistics printing for data collection
def print_detailed_statistics(server_log_data, progressive_metrics, local_accuracies, attacker_ids, 
                             experiment_name='experiment', results_dir=None):
    """
    Print detailed statistics for data collection and multi-run comparison.
    Outputs all key metrics in tabular format for easy copying to Excel/CSV.
    
    Args:
        server_log_data: List of round logs from server
        progressive_metrics: Dictionary with progressive metrics
        local_accuracies: Dictionary with local accuracies per client
        attacker_ids: List of attacker client IDs
        experiment_name: Name of the experiment (for file naming)
        results_dir: Path to results directory (default: Path("results"))
    """
    import csv
    from pathlib import Path
    
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)
    
    print("\n" + "=" * 80)
    print("📊 DETAILED EXPERIMENT STATISTICS FOR DATA COLLECTION")
    print("=" * 80)
    
    rounds = progressive_metrics['rounds']
    if not rounds:
        print("⚠️  No rounds completed.")
        return
    
    # Get all client IDs
    all_client_ids = set()
    for log in server_log_data:
        if 'local_accuracies' in log:
            all_client_ids.update(log['local_accuracies'].keys())
        if 'aggregation' in log and 'similarities' in log['aggregation']:
            # Infer client IDs from similarities count (if available)
            similarities = log['aggregation'].get('similarities', [])
            accepted = log['aggregation'].get('accepted_clients', [])
            all_client_ids.update(accepted)
    
    # Also include from local_accuracies history
    if local_accuracies:
        all_client_ids.update(local_accuracies.keys())
    
    all_client_ids = sorted(all_client_ids)
    attacker_ids_set = set(attacker_ids) if attacker_ids else set()
    
    # ========== 1. Global Accuracy Table ==========
    print("\n" + "-" * 80)
    print("1️⃣  GLOBAL ACCURACY (Per Round)")
    print("-" * 80)
    print(f"{'Round':<8} | {'Clean Accuracy':<15} | {'Accuracy Change':<17}")
    print("-" * 80)
    
    clean_acc = progressive_metrics['clean_acc']
    for i, r in enumerate(rounds):
        acc = clean_acc[i] if i < len(clean_acc) else 0.0
        acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
        print(f"{r:<8} | {acc:<15.6f} | {acc_change:>+17.6f}")
    
    print("-" * 80)
    if clean_acc:
        print(f"Summary: Initial={clean_acc[0]:.6f}, Final={clean_acc[-1]:.6f}, "
              f"Best={max(clean_acc):.6f}, Change={clean_acc[-1]-clean_acc[0]:+.6f}")
    
    # ========== 2. Cosine Similarity Table ==========
    print("\n" + "-" * 80)
    print("2️⃣  COSINE SIMILARITY (Per Round, Per Client)")
    print("-" * 80)
    
    # Prepare header
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        aggregation = log.get('aggregation', {})
        similarities = aggregation.get('similarities', [])
        accepted = aggregation.get('accepted_clients', [])
        
        # Create similarity map
        all_clients_round = sorted(set(accepted))
        sim_map = {}
        if len(similarities) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                sim_map[cid] = similarities[idx]
        
        # Print row
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            sim = sim_map.get(cid, 0.0)
            row += f"{sim:<14.6f} | "
        
        # Calculate mean and std for this round
        sim_values = [sim_map.get(cid, 0.0) for cid in all_client_ids if cid in sim_map]
        mean_sim = np.mean(sim_values) if sim_values else 0.0
        std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
        
        row += f"{mean_sim:<6.6f} | {std_sim:.6f}"
        print(row)
    
    print("-" * 80)
    
    # ========== 2b. Euclidean Distance Table ==========
    print("\n" + "-" * 80)
    print("2b. EUCLIDEAN DISTANCE (Per Round, Per Client)")
    print("-" * 80)
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    for log in server_log_data:
        round_num = log['round']
        aggregation = log.get('aggregation', {})
        euclidean_distances = aggregation.get('euclidean_distances', [])
        accepted = aggregation.get('accepted_clients', [])
        all_clients_round = sorted(set(accepted))
        dist_map = {}
        if len(euclidean_distances) == len(all_clients_round):
            for idx, cid in enumerate(all_clients_round):
                dist_map[cid] = euclidean_distances[idx]
        row = f"{round_num:<6} | "
        for cid in all_client_ids:
            d = dist_map.get(cid, 0.0)
            row += f"{d:<14.6f} | "
        dist_values = [dist_map.get(cid, 0.0) for cid in all_client_ids if cid in dist_map]
        mean_d = np.mean(dist_values) if dist_values else 0.0
        std_d = np.std(dist_values) if len(dist_values) > 1 else 0.0
        row += f"{mean_d:<6.6f} | {std_d:.6f}"
        print(row)
    print("-" * 80)
    
    # ========== 2c. Global Loss (Per Round) ==========
    print("\n" + "-" * 80)
    print("2c. GLOBAL LOSS (Per Round)")
    print("-" * 80)
    print(f"{'Round':<8} | {'Global Loss':<15}")
    print("-" * 80)
    for log in server_log_data:
        round_num = log['round']
        global_loss = log.get('global_loss', 0.0)
        print(f"{round_num:<8} | {global_loss:<15.6f}")
    print("-" * 80)
    
    # ========== 3. Local Accuracy Table ==========
    print("\n" + "-" * 80)
    print("3️⃣  LOCAL ACCURACY (Per Round, Per Client)")
    print("-" * 80)
    
    # Prepare header
    header = "Round | "
    for cid in all_client_ids:
        client_type = "A" if cid in attacker_ids_set else "B"
        header += f"Client{cid}({client_type}) | "
    header += "Mean | Std"
    print(header)
    print("-" * 80)
    
    for log in server_log_data:
        round_num = log['round']
        local_accs_round = log.get('local_accuracies', {})
        
        # Print row
        row = f"{round_num:<6} | "
        acc_values = []
        for cid in all_client_ids:
            acc = local_accs_round.get(cid, 0.0)
            acc_values.append(acc)
            row += f"{acc:<14.6f} | "
        
        # Calculate mean and std
        mean_acc = np.mean(acc_values) if acc_values else 0.0
        std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
        row += f"{mean_acc:<6.6f} | {std_acc:.6f}"
        print(row)
    
    print("-" * 80)
    
    # ========== 4. Save to CSV files for easy import ==========
    print("\n" + "-" * 80)
    print("💾 SAVING DATA TO CSV FILES FOR EASY COLLECTION")
    print("-" * 80)
    
    # Save Global Accuracy
    csv_path1 = results_dir / f"{experiment_name}_global_accuracy.csv"
    with open(csv_path1, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Clean_Accuracy', 'Accuracy_Change'])
        for i, r in enumerate(rounds):
            acc = clean_acc[i] if i < len(clean_acc) else 0.0
            acc_change = (clean_acc[i] - clean_acc[i-1]) if i > 0 else 0.0
            writer.writerow([r, f"{acc:.6f}", f"{acc_change:.6f}"])
    print(f"✅ Global Accuracy saved to: {csv_path1}")
    
    # Save Cosine Similarity
    csv_path2 = results_dir / f"{experiment_name}_cosine_similarity.csv"
    with open(csv_path2, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                                           for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            aggregation = log.get('aggregation', {})
            similarities = aggregation.get('similarities', [])
            accepted = aggregation.get('accepted_clients', [])
            
            all_clients_round = sorted(set(accepted))
            sim_map = {}
            if len(similarities) == len(all_clients_round):
                for idx, cid in enumerate(all_clients_round):
                    sim_map[cid] = similarities[idx]
            
            row = [round_num]
            sim_values = []
            for cid in all_client_ids:
                sim = sim_map.get(cid, 0.0)
                sim_values.append(sim)
                row.append(f"{sim:.6f}")
            
            mean_sim = np.mean(sim_values) if sim_values else 0.0
            std_sim = np.std(sim_values) if len(sim_values) > 1 else 0.0
            row.extend([f"{mean_sim:.6f}", f"{std_sim:.6f}"])
            writer.writerow(row)
    print(f"✅ Cosine Similarity saved to: {csv_path2}")
    
    # Save Local Accuracy
    csv_path3 = results_dir / f"{experiment_name}_local_accuracy.csv"
    with open(csv_path3, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['Round'] + [f"Client_{cid}_{'A' if cid in attacker_ids_set else 'B'}" 
                             for cid in all_client_ids] + ['Mean', 'Std']
        writer.writerow(header)
        
        for log in server_log_data:
            round_num = log['round']
            local_accs_round = log.get('local_accuracies', {})
            
            row = [round_num]
            acc_values = []
            for cid in all_client_ids:
                acc = local_accs_round.get(cid, 0.0)
                acc_values.append(acc)
                row.append(f"{acc:.6f}")
            
            mean_acc = np.mean(acc_values) if acc_values else 0.0
            std_acc = np.std(acc_values) if len(acc_values) > 1 else 0.0
            row.extend([f"{mean_acc:.6f}", f"{std_acc:.6f}"])
            writer.writerow(row)
    print(f"✅ Local Accuracy saved to: {csv_path3}")
    
    print("\n" + "=" * 80)
    print("✅ All statistics printed and saved to CSV files!")
    print("   You can now easily collect data from multiple runs and compare them.")
    print("=" * 80)

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

def main(config_overrides: Optional[Dict] = None):
    config = {
        # ========== Experiment Configuration ==========
        'experiment_name': 'vgae_augmp',  # Name for result files and logs
        'seed': 42069,  # Random seed for reproducibility (int), 42 is the default
        
        # ========== Federated Learning Setup ==========
        'num_clients': 7,  # Total number of federated learning clients (int)
        'num_attackers': 0,  # Number of attacker clients (int, must be < num_clients)
        'num_benign_clients': None,  # Optional: Explicit number of benign clients for baseline experiment
                                    # If None, baseline will use (num_clients - num_attackers) to ensure fair comparison
                                    # If set, baseline experiment will use exactly this many benign clients
        'num_rounds': 50,  # Total number of federated learning rounds (int)
        
        # ========== Training Hyperparameters ==========
        'client_lr': 5e-5,  # Learning rate for local client training (float)
        'server_lr': 1.0,  # Server learning rate for model aggregation (fixed at 1.0)
        'batch_size': 128,  # Batch size for local training (int)
        'test_batch_size': 256,  # Batch size for test/validation data loaders (int)
        'local_epochs': 2,  # Number of local training epochs per round (int, per paper Section IV)
        'grad_clip_norm': 1.0,  # Benign client grad clipping. Decoder models: Pythia-160m try 0.5 if nan; Qwen2.5-0.5B typically stable at 1.0
        'alpha': 0.0,  # FedProx proximal coefficient μ: loss += (μ/2)*||w - w_global||². Set 0 for standard FedAvg, >0 to penalize local drift from global model (helps Non-IID stability)
        
        # ========== Dataset Configuration ==========
        # Choose dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers' — set num_labels and max_length accordingly
        # Dataset 1: AG News
        'dataset': 'ag_news',  # news classification (4 classes)
        'num_labels': 4,       # AG News: 4 | IMDB: 2 | DBpedia: 14 | Yahoo Answers: 10
        'max_length': 128,     # AG News: 128 | IMDB: 512/256 | DBpedia: 512 | Yahoo Answers: 256
        # -------------------------------------------
        # Dataset 2: IMDB
        # 'dataset': 'imdb',   # sentiment (2 classes)
        # 'num_labels': 2,
        # 'max_length': 512,
        # -------------------------------------------
        # Dataset 3: DBpedia (14 classes, 560K train / 70K test)
        # 'dataset': 'dbpedia',   # topic classification (14 classes)
        # 'num_labels': 14,
        # 'max_length': 512,
        # -------------------------------------------
        # Dataset 4: Yahoo Answers (10 classes, 1.4M train / 60K test)
        # 'dataset': 'yahoo_answers',   # topic classification (10 classes, yassiracharki/Yahoo_Answers_10_categories_for_NLP)
        # 'num_labels': 10,       # Yahoo Answers: 10 classes
        # 'max_length': 128,      # Yahoo Answers: 256 (Q&A text, similar length to AG News)
        
        # ========== Data Distribution ==========
        'data_distribution': 'non-iid',  # 'iid' for uniform random, 'non-iid' for Dirichlet-based heterogeneous distribution
        'dirichlet_alpha': 0.3,  # Only used when data_distribution='non-iid'. Lower = more heterogeneous, higher = more balanced
        # 'dataset_size_limit': None,  # Limit dataset size (None = full dataset). AG News: ~120K; IMDB: 25K; DBpedia: 560K; Yahoo Answers: 1.4M
        'dataset_size_limit': 20000,  # Limit for faster experimentation. When set: train ≤ limit, test ≤ limit × 0.15 (same rule for all datasets)

        # ========== Training Mode Configuration ==========
        'use_lora': True,  # True for LoRA fine-tuning, False for full fine-tuning
        # LoRA parameters (only used when use_lora=True)
        # NOTE: Lower r values = faster training but potentially less capacity
        # Recommended: r=8 for speed, r=16 for better performance (default)
        'lora_r': 8,  # LoRA rank (controls the rank of low-rank matrices). r=8 for speed, r=16/32 for better capacity
        'lora_alpha': 16,  # LoRA alpha (scaling factor, typically 2*r). Must match r: alpha=2*r
        'lora_dropout': 0.1,  # LoRA dropout rate
        'lora_target_modules': None,  # None = use default for DistilBERT (["q_lin", "k_lin", "v_lin", "out_lin"])
        
        # Model configuration
        # Supported models:
        # Encoder-only (BERT-style): 'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'microsoft/deberta-v3-base'
        # 'model_name': 'distilbert-base-uncased',  # distilbert 67M
        # # -------------------------------------------
        # Decoder-only (GPT-style): 'gpt2', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-1b', 'facebook/opt-125m', 'Qwen/Qwen2.5-0.5B'
        # 'model_name': 'gpt2',                      # GPT-2 124M — stable decoder baseline
        # 'model_name': 'EleutherAI/pythia-160m',    # Pythia-160M (may need grad_clip_norm=0.5)
        # 'model_name': 'facebook/opt-125m',         # OPT-125M (Meta)
        'model_name': 'Qwen/Qwen2.5-0.5B',  # Qwen2.5-0.5B ~494M (Alibaba, LLaMA-style arch, Apache 2.0) — use BASE for fine-tuning (or Qwen/Qwen2.5-0.5B-Instruct)
        # num_labels and max_length: set above in Dataset Configuration based on chosen dataset
        

        # ========== Attack Configuration ==========
        'attack_method': 'AugMP',  # Proposed: 'AugMP' (legacy alias 'GRMP'). Baselines: 'ALIE', 'SignFlipping', 'Gaussian'
        'attack_start_round': 0,  # Round when attack phase starts (int, now all rounds use complete poisoning)
        
        # ========== ALIE Attack Parameters (only used when attack_method='ALIE') ==========
        'alie_z_max': None,  # NeurIPS '19: z-score multiplier for ALIE. None = auto-compute based on num_clients and num_attackers
        'alie_attack_start_round': None,  # Round to start ALIE attack (None = start immediately, overrides attack_start_round)
        # ========== Sign-Flipping Attack Parameters (only used when attack_method='SignFlipping') ==========
        'sign_flip_scale': 10.0,  # ICML '18: malicious = -scale * g_own (own update). Paper uses 10.
        'sign_flip_attack_start_round': None,  # Round to start Sign-Flipping attack (None = start immediately)
        # ========== Gaussian Attack Parameters (only used when attack_method='Gaussian') ==========
        'gaussian_attack_start_round': None,  # USENIX Security '20: Round to start Gaussian attack (None = start immediately)
        'gaussian_std_scale': 5.0,  # Scale factor for noise std: attack_vec ~ N(mean, (scale*std)²). scale>1 expands noise to increase impact (FedAvg). 1.0=original Fang et al.

        # ========== VGAE Training Parameters ==========
        # Reference paper: input_dim=5, hidden1_dim=32, hidden2_dim=16, num_epoch=10, lr=0.01
        # Note: dim_reduction_size should be <= total trainable parameters
        'dim_reduction_size': 1000,  # Reduced dimensionality of LLM parameters (auto-adjusted for LoRA if needed)
        'vgae_epochs': 20,  # Number of epochs for VGAE training (reference: 20)
        'vgae_lr': 0.01,  # Learning rate for VGAE optimizer (reference: 0.01)
        'vgae_hidden_dim': 64,  # VGAE hidden layer dimension (per paper: hidden1_dim=32)
        'vgae_latent_dim': 32,  # VGAE latent space dimension (per paper: hidden2_dim=16)
        'vgae_dropout': 0,  # VGAE encoder dropout rate (0=no dropout, higher=more regularization to prevent overfitting)
        'vgae_kl_weight': 0.1,  # KL divergence weight in VGAE loss: L = L_recon + kl_weight * KL(q||p). Higher=stronger latent regularization
        # ========== Graph Construction Parameters ==========
        'graph_threshold': 0.5,  # Cosine similarity threshold for adjacency matrix: A[i,j]=1 if sim(Δ_i,Δ_j)>threshold, else 0. Higher=sparser graph

        # ========== AugMP proxy optimization parameters ==========
        'proxy_step': 0.001,  # Step size for gradient-free ascent toward global-loss proxy
        'proxy_steps': 200,  # Number of optimization steps for AugMP proxy objective (int)
        'attacker_proxy_grad_clip_norm': 1.0,  # AugMP proxy-parameter update only; separate from benign training
        'attacker_claimed_data_size': None,  # None = use actual assigned data size
        'early_stop_constraint_stability_steps': 1,  # Early stopping: stop after N consecutive steps satisfying constraint (int)

        # ========== Formula 4 Constraint Parameters ==========
        'dist_bound': None,  # Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ dist_bound (None = use benign max distance)
        'sim_bound_low': None,  # Manual lower bound for cosine similarity (None = use benign min). e.g. 0.0 to require non-negative similarity
        'sim_bound_up': None,   # Manual upper bound for cosine similarity (None = use benign mean)
        # Server cosine similarity mode: 'local_vs_global' (each client vs Δ_g) | 'pairwise' (local vs local, report mean to others) | 'both'
        'server_similarity_mode': 'pairwise',  # Use 'pairwise' to avoid self-comparison; set to 'local_vs_global' to match AugMP constraint definition

        # ========== Lagrangian Dual Parameters ==========
        'use_lagrangian_dual': True,  # Whether to use Lagrangian Dual mechanism (bool, True/False)
        # Distance constraint multiplier parameters
        'lambda_dist_init': 0.1,  # Initial λ_dist(t) value for distance constraint: dist(Δ_att, Δ_g) ≤ dist_bound
        'lambda_dist_lr': 0.001,    # Learning rate for λ_dist(t) update (dual ascent step size)
        
        # ========== Cosine Similarity Constraint Parameters (TWO-SIDED with TWO multipliers) False by default ==========
        'use_cosine_similarity_constraint': True,  # Whether to enable cosine similarity constraints (bool, True/False) False by default! open both to use pairwise sim
        'use_pairwise_similarity_in_constraint': True,  # When True and similarity constraint on: use pairwise sim (align with server_similarity_mode='pairwise') open both to use pairwise sim
        'lambda_sim_low_init': 0.1,  # Initial λ_sim_low(t) value for lower bound constraint: sim_bound_low <= sim_att
        'lambda_sim_up_init': 0.1,   # Initial λ_sim_up(t) value for upper bound constraint: sim_att <= sim_bound_up
        'lambda_sim_low_lr': 0.001,    # Learning rate for λ_sim_low(t) update
        'lambda_sim_up_lr': 0.001,     # Learning rate for λ_sim_up(t) update

        # ========== Augmented Lagrangian Method (ALM) Parameters ==========
        # Standard ALM adds quadratic penalties: (ρ/2) * g(x)^2 for each inequality constraint g(x) ≤ 0.
        'use_augmented_lagrangian': True,   # Enable Augmented Lagrangian (requires use_lagrangian_dual=True)
        'lambda_update_mode': 'alm',    # Dual variable update: "classic"=λ += lr*g (fixed step), "alm"=λ += ρ*g (penalty-scaled step, standard ALM)
        # Penalty parameters ρ (per-constraint): controls quadratic penalty strength (ρ/2)*max(0,g)^2 in ALM objective
        'rho_dist_init': 1.0,
        'rho_sim_low_init': 1.0,
        'rho_sim_up_init': 1.0,
        # Adaptive ρ update (monotone increase)
        'rho_adaptive': True,
        'rho_theta': 0.5,            # If σ_k > theta * σ_{k-1} then increase ρ
        'rho_increase_factor': 2.0,
        'rho_min': 1e-4,
        'rho_max': 1e4,
        # ========== Proxy Loss Estimation Parameters ==========
        'attacker_use_proxy_data': True,  # If True, AugMP client uses proxy set to estimate F(w'_g); if False, no data access (constraint-only optimization)
        'proxy_sample_size': 256,  # Number of samples in proxy dataset for F(w'_g) estimation (int)
                                # Increased from 128 to 512 for better accuracy (4 batches with test_batch_size=128)
        'proxy_max_batches_opt': 1,  # Max batches per _proxy_global_loss call in optimization loop (int)
                                # Only has effect when proxy set has >1 batch (proxy_sample_size > test_batch_size).
        'proxy_max_batches_eval': 1,  # Max batches per _proxy_global_loss call in final evaluation (int)

        # ========== Global checkpoint (for downstream generation / transfer experiments) ==========
        'save_global_checkpoint': True,  # True: save server.global_model after FL under results_dir/global_checkpoint_subdir
        'global_checkpoint_subdir': 'global_checkpoint',  # Subfolder name under results/ (same run uses results_dir from setup)
        # ========== Task 2: optional downstream causal generation (same run as FL) ==========
        'run_downstream_after_fl': True,  # True: subprocess run_downstream_generation.py after checkpoint save
        'downstream_probes': 'data/AG News Datasets/ag_news_business_30.json',  # relative to repo root
        'downstream_output': None,  # None -> results/<experiment_name>_downstream_gen.jsonl; else path (relative to results/ if not absolute)
        'downstream_device': None,  # None -> cuda if available else cpu
        # Extra CLI tokens for run_downstream_generation.py (SeqCLS classify + CausalLM explain)
        'downstream_cli_args': [
            '--stable',
        ],

    }
    if config_overrides:
        config.update(config_overrides)

    # Run experiment (attack if num_attackers > 0, baseline if num_attackers == 0)
    if config.get('num_attackers', 0) > 0:
        attack_method = config.get('attack_method', 'AugMP')
        if attack_method == 'GRMP':
            attack_method = 'AugMP'  # legacy config alias for proposed method
        if attack_method == 'ALIE':
            print("Running ALIE Attack (Model Poisoning Baseline)...")
        elif attack_method == 'SignFlipping':
            print("Running Sign-Flipping Attack (Model Poisoning Baseline)...")
        elif attack_method == 'Gaussian':
            print("Running Gaussian Attack (Random Model Poisoning Baseline)...")
        else:
            print("Running AugMP (graph-augmented model manipulation) with VGAE...")
    else:
        print("Running Baseline Experiment (No Attack)...")
    
    results, metrics = run_experiment(config)
    analyze_results(metrics)
        

if __name__ == "__main__":
    main()
    