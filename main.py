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

# Import our custom modules
from models import NewsClassifierModel
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server

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
    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'], 
        poison_rate=config['poison_rate']
    )

    # 2. Partition data among clients
    print("\nPartitioning data...")
    indices = np.arange(len(data_manager.train_texts))
    # Fixed shuffle for consistent partitioning across runs
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(indices)

    samples_per_client = len(indices) // config['num_clients']
    client_indices = {}

    for i in range(config['num_clients']):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < config['num_clients'] - 1 else len(indices)
        client_indices[i] = indices[start_idx:end_idx].tolist()

    # 3. Get global test loaders
    test_loader = data_manager.get_test_loader()
    attack_test_loader = data_manager.get_attack_test_loader()

    # 4. Initialize Global Model
    print("Initializing global model (DistilBERT)...")
    global_model = NewsClassifierModel()

    # 5. Initialize Server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        attack_test_loader=attack_test_loader,
        defense_threshold=config['defense_threshold'],
        total_rounds=config['num_rounds'],
        server_lr=config.get('server_lr', 1.0)
    )

    # 6. Create Clients
    print("\nCreating federated learning clients...")
    for client_id in range(config['num_clients']):
        # Determine if benign or attacker
        # Logic: Last 'num_attackers' clients are attackers
        if client_id < (config['num_clients'] - config['num_attackers']):
            # --- Benign Client ---
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]
            
            # Create static dataloader for benign client
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer)
            client_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs']
            )
        else:
            # --- Attacker Client ---
            print(f"  Client {client_id}: ATTACKER (VGAE Enabled)")
            client = AttackerClient(
                client_id=client_id,
                model=global_model,
                data_manager=data_manager,
                data_indices=client_indices[client_id],
                lr=config['client_lr'],
                local_epochs=config['local_epochs']
            )
            # Configure VGAE specific parameters
            # Note: base_amplification is deprecated in VGAE version
            client.dim_reduction_size = config.get('dim_reduction_size', 5000)

        server.register_client(client)
    
    return server, results_dir

# Run the experiment
def run_experiment(config):
    server, results_dir = setup_experiment(config)

    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_clean, initial_asr = server.evaluate()
    print(f"Initial Performance - Clean: {initial_clean:.4f}, ASR: {initial_asr:.4f}")

    print("\n" + "=" * 50)
    print("Starting Federated Learning Rounds")
    print("=" * 50)

    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'attack_asr': [],
        'defense_threshold': []
    }

    try:
        for round_num in range(config['num_rounds']):
            round_log = server.run_round(round_num)

            # Track metrics
            progressive_metrics['rounds'].append(round_num + 1)
            progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
            progressive_metrics['attack_asr'].append(round_log['attack_success_rate'])
            progressive_metrics['defense_threshold'].append(round_log['defense']['threshold'])
            
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
        'progressive_metrics': progressive_metrics
    }

    results_path = results_dir / f"{config['experiment_name']}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
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
    asr = metrics['attack_asr']

    print(f"Total Rounds: {len(rounds)}")
    print(f"Final Clean Accuracy: {clean[-1]:.4f}")
    print(f"Final Attack Success Rate: {asr[-1]:.4f}")
    print(f"Peak ASR: {max(asr):.4f}")

def main():
    config = {
        'experiment_name': 'vgae_grmp_attack',
        'seed': 42,
        'num_clients': 6,
        'num_attackers': 2, 
        'num_rounds': 20,
        'client_lr': 2e-5,
        'server_lr': 0.8,
        'batch_size': 16,
        'local_epochs': 2,
        'poison_rate': 3.0,
        'dim_reduction_size': 10000,
        'defense_threshold': 0.070
    }

    print("Running GRMP Attack with VGAE...")
    results, metrics = run_experiment(config)
    analyze_results(metrics)

if __name__ == "__main__":
    main()