# main.py - 支持TPU/GPU的Progressive GRMP Attack实验

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from models import NewsClassifierModel, VGAE
from data_loader import DataManager, NewsDataset
from client import BenignClient, AttackerClient
from server import Server
from device_utils import device_manager
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# 在Colab中安装必要的依赖
def setup_colab():
    """在Colab环境中设置TPU/GPU"""
    try:
        import google.colab
        IN_COLAB = True
        print("🔧 Detected Colab environment")

        # 如果使用TPU，需要安装额外依赖
        if device_manager.is_tpu():
            print("📦 Installing TPU dependencies...")
            import subprocess
            subprocess.run(['pip', 'install', 'cloud-tpu-client==0.10', 'torch_xla'], check=True)

    except ImportError:
        IN_COLAB = False
        print("💻 Running in local environment")

    return IN_COLAB


def setup_experiment(config):
    """Initialize experiment components with TPU/GPU support"""
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if device_manager.is_gpu():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Initialize data manager
    print("\n" + "=" * 50)
    print("Setting up Progressive GRMP Attack Experiment")
    print(f"Device: {device_manager.device_type.upper()}")
    print("=" * 50)

    data_manager = DataManager(
        num_clients=config['num_clients'],
        num_attackers=config['num_attackers'],
        poison_rate=config['poison_rate']
    )

    # Get data loaders and partition indices
    print("\nPartitioning data among clients...")

    # First, get the partition indices
    indices = np.arange(len(data_manager.train_texts))
    np.random.shuffle(indices)
    samples_per_client = len(indices) // config['num_clients']

    # Store indices for each client
    client_indices = {}
    for i in range(config['num_clients']):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < config['num_clients'] - 1 else len(indices)
        client_indices[i] = indices[start_idx:end_idx].tolist()

    # Create initial data loaders with appropriate batch size for device
    batch_size = config.get('batch_size', 16)
    if device_manager.is_tpu():
        # TPU performs better with larger batch sizes
        batch_size = batch_size * 2
        print(f"📊 Using batch size {batch_size} for TPU")

    test_loader = data_manager.get_test_loader()
    attack_test_loader = data_manager.get_attack_test_loader()

    # Initialize global model
    print("\nInitializing global model...")
    global_model = NewsClassifierModel()

    # Initialize server
    server = Server(
        global_model=global_model,
        test_loader=test_loader,
        attack_test_loader=attack_test_loader,
        defense_threshold=config['defense_threshold'],
        total_rounds=config['num_rounds']
    )

    # Create clients
    print("\nCreating federated learning clients...")
    for client_id in range(config['num_clients']):
        if client_id < (config['num_clients'] - config['num_attackers']):
            # Benign client - create normal dataloader
            client_texts = [data_manager.train_texts[i] for i in client_indices[client_id]]
            client_labels = [data_manager.train_labels[i] for i in client_indices[client_id]]

            # Print distribution
            client_dist = np.bincount(client_labels, minlength=4)
            print(f"Client {client_id} (Benign) - Distribution: "
                  f"{dict(zip(['World', 'Sports', 'Business', 'Sci/Tech'], client_dist))}")

            # Create dataset and loader
            dataset = NewsDataset(client_texts, client_labels, data_manager.tokenizer)
            client_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            client = BenignClient(
                client_id=client_id,
                model=global_model,
                data_loader=client_loader,
                lr=config['client_lr'],
                local_epochs=config['local_epochs']
            )
        else:
            # Attacker client - will create dynamic dataloaders
            print(f"Client {client_id} (Attacker) - Will use progressive poisoning")

            client = AttackerClient(
                client_id=client_id,
                model=global_model,
                data_manager=data_manager,
                data_indices=client_indices[client_id],
                lr=config['client_lr'],
                local_epochs=config['local_epochs']
            )

            # Set base amplification factor
            client.base_amplification = config.get('base_amplification_factor', 3.0)
            client.progressive_enabled = config.get('progressive_attack', True)

        server.register_client(client)

    return server, results_dir, config


def run_experiment(config):
    """Run the progressive GRMP attack experiment with TPU/GPU optimization"""
    # Setup
    server, results_dir, config = setup_experiment(config)

    # Initial evaluation
    print("\nEvaluating initial model...")
    initial_clean, initial_asr = server.evaluate()
    print(f"Initial Performance - Clean: {initial_clean:.4f}, ASR: {initial_asr:.4f}")

    # Run federated learning rounds
    print("\n" + "=" * 50)
    print("Starting Progressive Federated Learning Attack")
    print(f"Using {device_manager.device_type.upper()} acceleration")
    print("=" * 50)

    # Track progressive metrics
    progressive_metrics = {
        'rounds': [],
        'clean_acc': [],
        'attack_asr': [],
        'detection_rate': [],
        'device_type': device_manager.device_type
    }

    # Training time tracking
    import time
    start_time = time.time()

    for round_num in range(config['num_rounds']):
        round_start = time.time()
        round_log = server.run_round(round_num)
        round_time = time.time() - round_start

        # Track metrics
        progressive_metrics['rounds'].append(round_num + 1)
        progressive_metrics['clean_acc'].append(round_log['clean_accuracy'])
        progressive_metrics['attack_asr'].append(round_log['attack_success_rate'])
        progressive_metrics['detection_rate'].append(round_log.get('detection_rate', 0))

        print(f"⏱️  Round {round_num + 1} completed in {round_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n✅ Total training time: {total_time:.2f}s on {device_manager.device_type.upper()}")

    # Save results
    results_data = {
        'config': config,
        'results': server.log_data,
        'progressive_metrics': progressive_metrics,
        'device_info': {
            'type': device_manager.device_type,
            'total_time': total_time,
            'avg_round_time': total_time / config['num_rounds']
        }
    }

    device_suffix = device_manager.device_type
    results_path = results_dir / f"progressive_grmp_{config['experiment_name']}_{device_suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n📊 Results saved to: {results_path}")
    return server.log_data, progressive_metrics


def analyze_progressive_results(results, metrics, config):
    """Analyze progressive attack results with device performance info"""
    print("\n" + "=" * 50)
    print("Progressive GRMP Attack Analysis")
    print("=" * 50)

    # Configuration summary
    print("\nAttack Configuration:")
    print(f"  Device: {metrics['device_type'].upper()}")
    print(f"  Total Clients: {config['num_clients']}")
    print(f"  Attackers: {config['num_attackers']} ({config['num_attackers'] / config['num_clients'] * 100:.0f}%)")
    print(f"  Base Poison Rate: {config['poison_rate'] * 100:.0f}%")
    print(f"  Total Rounds: {config['num_rounds']}")
    print(f"  Progressive Attack: {'Enabled' if config.get('progressive_attack', True) else 'Disabled'}")

    # Device-specific optimizations applied
    if device_manager.is_tpu():
        print("\n🚀 TPU Optimizations Applied:")
        print("  - Parallel data loading")
        print("  - XLA JIT compilation")
        print("  - Increased batch size")
    elif device_manager.is_gpu():
        print("\n🚀 GPU Optimizations Applied:")
        print("  - Mixed precision training (AMP)")
        print("  - CUDNN acceleration")
        print("  - Optimized memory allocation")

    # Stage-wise analysis
    print("\nProgressive Attack Stages:")
    stages = [
        (0, 5, "Early (Trust Building)"),
        (5, 10, "Growing (Increasing Impact)"),
        (10, 15, "Mature (Strong Attack)"),
        (15, 100, "Full Force (Maximum Impact)")
    ]

    for start, end, name in stages:
        stage_rounds = [i for i, r in enumerate(metrics['rounds']) if start < r <= min(end, config['num_rounds'])]
        if stage_rounds:
            avg_acc = np.mean([metrics['clean_acc'][i] for i in stage_rounds])
            avg_asr = np.mean([metrics['attack_asr'][i] for i in stage_rounds])
            avg_detect = np.mean([metrics['detection_rate'][i] for i in stage_rounds])

            print(f"\n  {name}:")
            print(f"    Avg Clean Accuracy: {avg_acc:.4f}")
            print(f"    Avg Attack Success: {avg_asr:.4f}")
            print(f"    Avg Detection Rate: {avg_detect:.1%}")

    # Overall performance
    final_round = results[-1]
    print(f"\nFinal Performance:")
    print(f"  Clean Accuracy: {final_round['clean_accuracy']:.4f}")
    print(f"  Attack Success Rate: {final_round['attack_success_rate']:.4f}")
    print(f"  Accuracy Drop: {(results[0]['clean_accuracy'] - final_round['clean_accuracy']) * 100:.2f}%")

    # Attack effectiveness
    max_asr = max(metrics['attack_asr'])
    max_asr_round = metrics['attack_asr'].index(max_asr) + 1
    print(f"\nAttack Effectiveness:")
    print(f"  Peak ASR: {max_asr:.4f} (Round {max_asr_round})")
    print(f"  Average Detection Rate: {np.mean(metrics['detection_rate']):.1%}")

    # Success milestones
    print(f"\nAttack Milestones:")
    for threshold in [0.1, 0.25, 0.5, 0.75]:
        rounds_above = [r for r, asr in zip(metrics['rounds'], metrics['attack_asr']) if asr >= threshold]
        if rounds_above:
            print(f"  ASR ≥ {threshold * 100:.0f}%: First achieved in Round {rounds_above[0]}")
        else:
            print(f"  ASR ≥ {threshold * 100:.0f}%: Not achieved")


def main():
    """Main experiment with TPU/GPU support"""
    # Setup Colab if needed
    IN_COLAB = setup_colab()

    # Configuration
    config = {
        'experiment_name': 'progressive_semantic_poisoning',
        'seed': 42,
        'num_clients': 4,
        'num_attackers': 1,  # 25% attackers
        'num_rounds': 5,  # Adjust based on device speed
        'client_lr': 1e-4,
        'poison_rate': 0.8,  # Base rate (will be adjusted progressively)
        'defense_threshold': 0.4,
        'local_epochs': 2,
        'base_amplification_factor': 3.0,
        'progressive_attack': True,  # Enable progressive strategy
        'batch_size': 16  # Will be adjusted for TPU
    }

    # Adjust rounds based on device
    if device_manager.is_tpu():
        config['num_rounds'] = 20  # TPU can handle more rounds efficiently
    elif device_manager.is_gpu():
        config['num_rounds'] = 15
    else:
        config['num_rounds'] = 5  # Fewer rounds for CPU

    print("Progressive GRMP (Graph Representation-based Model Poisoning) Attack")
    print("Target: AG News Classification - Business+Finance → Sports")
    print("Strategy: Gradual poisoning intensity to evade detection")
    print(f"🚀 Accelerator: {device_manager.device_type.upper()}")

    # Run experiment
    results, metrics = run_experiment(config)

    # Analyze results
    analyze_progressive_results(results, metrics, config)

    print("\n" + "=" * 50)
    print("Progressive attack experiment completed!")
    print("=" * 50)

    # Colab-specific: Save results to Google Drive if available
    if IN_COLAB:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            import shutil
            shutil.copytree('results', '/content/drive/MyDrive/grmp_results', dirs_exist_ok=True)
            print("\n📁 Results saved to Google Drive!")
        except Exception as e:
            print(f"\n⚠️  Could not save to Google Drive: {e}")


if __name__ == "__main__":
    main()
