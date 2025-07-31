# server.py This module implements the Server class for federated learning, including model aggregation and defense mechanisms against GRMP attacks.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from client import BenignClient, AttackerClient
import torch.nn.functional as F

class Server:
    """Server class for federated learning with GRMP attack defense"""
    def __init__(self, global_model: nn.Module, test_loader, attack_test_loader,
                defense_threshold=0.4, total_rounds=20, server_lr=0.8, tolerance_factor=2):
        self.global_model = copy.deepcopy(global_model)
        self.test_loader = test_loader
        self.attack_test_loader = attack_test_loader
        self.defense_threshold = defense_threshold
        self.total_rounds = total_rounds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        self.clients = []
        self.log_data = []

        # Additional stability parameters
        self.server_lr = server_lr  # Server learning rate (inertia)
        self.tolerance_factor = tolerance_factor  # Defense tolerance level

        # Track historical data for adaptive adjustments
        self.history = {
            'asr': [],  # Attack Success Rate
            'clean_acc': [],  # Clean accuracy
            'rejection_rates': []  # Client rejection rates
        }

    def register_client(self, client):
        """Register a client to the server."""
        self.clients.append(client)

    def broadcast_model(self):
        """Broadcast the global model to all clients."""
        global_params = self.global_model.get_flat_params()
        for client in self.clients:
            client.model.set_flat_params(global_params.clone())
            client.reset_optimizer()

    def _compute_similarities(self, updates: List[torch.Tensor]) -> np.ndarray:
        """Compute mixed similarities - combining pairwise similarities and similarities with the mean."""
        n_updates = len(updates)

        print("  📊 Using mixed similarity computation")

        # Step 1: Compute pairwise similarities
        pairwise_sims = []
        for i in range(n_updates):
            other_sims = []
            for j in range(n_updates):
                if i != j:
                    sim = torch.cosine_similarity(
                        updates[i].unsqueeze(0),
                        updates[j].unsqueeze(0)
                    ).item()
                    other_sims.append(sim)
            # Use average instead of median (more lenient)
            avg_sim = np.mean(other_sims) if other_sims else 0
            pairwise_sims.append(avg_sim)

        # Step 2: Compute similarities with the mean update
        avg_update = torch.stack(updates).mean(dim=0)
        avg_sims = []
        for update in updates:
            sim = torch.cosine_similarity(
                update.unsqueeze(0),
                avg_update.unsqueeze(0)
            ).item()
            avg_sims.append(sim)

        # Step 3: Combine both similarities (weighted average)
        alpha = 0.7  # Weight for pairwise similarities
        similarities = []
        for i in range(n_updates):
            mixed_sim = alpha * pairwise_sims[i] + (1 - alpha) * avg_sims[i]
            similarities.append(mixed_sim)

        similarities = np.array(similarities)

        # Print information
        print(f"  📈 Mixed Similarity - Mean: {similarities.mean():.3f}, "
              f"Std Dev: {similarities.std():.3f}")

        # Display similarity for each client
        num_attackers = 2
        for i, sim in enumerate(similarities):
            if i >= n_updates - num_attackers:
                print(f"    Client {i} (Attacker): {sim:.3f}")
            else:
                print(f"    Client {i} (Benign): {sim:.3f}")

        return similarities

    def aggregate_updates(self, updates: List[torch.Tensor],
                          client_ids: List[int]) -> Dict:
        """
        Aggregate updates - Enhanced stability version.
        Uses a more lenient defense mechanism and smooth update strategy.
        """
        similarities = self._compute_similarities(updates)

        # Compute dynamic threshold (more lenient)
        mean_sim = similarities.mean()
        std_sim = similarities.std()

        # Apply tolerance_factor to make the threshold more lenient
        dynamic_threshold = max(self.defense_threshold,
                                mean_sim - self.tolerance_factor * std_sim)

        # Adaptive adjustment: If rejection rates are too high, further lower the threshold
        if len(self.history['rejection_rates']) > 0:
            recent_rejection_rate = np.mean(self.history['rejection_rates'][-3:])
            if recent_rejection_rate > 0.4:  # If more than 40% are rejected
                dynamic_threshold *= 0.9  # Reduce threshold by 10%
                print(f"  ⚠️ High rejection rate detected. Lowering threshold to: {dynamic_threshold:.3f}")

        accepted_indices = []
        rejected_indices = []

        for i, sim in enumerate(similarities):
            if sim >= dynamic_threshold:
                accepted_indices.append(i)
            else:
                rejected_indices.append(i)

        # Record rejection rate
        rejection_rate = len(rejected_indices) / len(updates)
        self.history['rejection_rates'].append(rejection_rate)

        defense_log = {
            'similarities': similarities.tolist(),
            'accepted_clients': [client_ids[i] for i in accepted_indices],
            'rejected_clients': [client_ids[i] for i in rejected_indices],
            'threshold': dynamic_threshold,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'tolerance_factor': self.tolerance_factor,
            'rejection_rate': rejection_rate
        }

        # Aggregate updates from accepted clients
        if accepted_indices:
            accepted_updates = [updates[i] for i in accepted_indices]
            aggregated_update = torch.stack(accepted_updates).mean(dim=0)

            # Smooth the global model update using server learning rate (key improvement)
            current_params = self.global_model.get_flat_params()
            new_params = current_params + self.server_lr * aggregated_update
            self.global_model.set_flat_params(new_params)

            print(f"  📊 Update Stats: Accepted {len(accepted_indices)}/{len(updates)} updates")
            print(f"  🔧 Server Learning Rate: {self.server_lr} (Smooth updates)")
        else:
            print("  ⚠️ Warning: No updates were accepted this round!")

        return defense_log

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate the global model's performance."""
        self.global_model.eval()

        # Evaluate clean accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.global_model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        clean_accuracy = correct / total if total > 0 else 0

        # Evaluate Attack Success Rate (ASR)
        attack_success = 0
        attack_total = 0

        if self.attack_test_loader:
            with torch.no_grad():
                for batch in self.attack_test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = self.global_model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs, dim=1)

                    attack_success += (predictions == 1).sum().item()
                    attack_total += len(predictions)

        attack_success_rate = attack_success / attack_total if attack_total > 0 else 0

        # Record historical metrics
        self.history['asr'].append(attack_success_rate)
        self.history['clean_acc'].append(clean_accuracy)

        return clean_accuracy, attack_success_rate

    def adaptive_adjustment(self, round_num: int):
        """Adaptively adjust parameters based on historical performance."""
        if len(self.history['asr']) < 2:
            return

        # Compute ASR changes
        asr_change = self.history['asr'][-1] - self.history['asr'][-2]
        current_asr = self.history['asr'][-1]

        # Adjust server learning rate if ASR fluctuations are too high
        if abs(asr_change) > 0.40:  # Fluctuation exceeds 40%
            self.server_lr = max(0.5, self.server_lr * 0.9)  # Reduce learning rate
            print(f"  🔄 Large fluctuation detected. Lowering server learning rate to: {self.server_lr:.2f}")
        elif abs(asr_change) < 0.05 and round_num > 5:  # If stable, increase learning rate
            self.server_lr = min(0.95, self.server_lr * 1.1)
            print(f"  🔄 System stable. Increasing server learning rate to: {self.server_lr:.2f}")

    def run_round(self, round_num: int) -> Dict:
        """Execute one round of federated learning - stable version."""
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # Adaptive adjustment
        self.adaptive_adjustment(round_num)

        # Display current stage
        if round_num < 5:
            stage = "🌱 Early Stage (Building trust)"
        elif round_num < 10:
            stage = "🌿 Growth Stage (Gradual enhancement)"
        elif round_num < 15:
            stage = "🌳 Mature Stage (Stable attacks)"
        else:
            stage = "🔥 Late Stage (Sustained pressure)"

        print(f"Attack Stage: {stage}")
        print(f"Current Parameters: server_lr={self.server_lr:.2f}, tolerance={self.tolerance_factor:.1f}")
        print(f"{'=' * 60}")

        # Broadcast the model
        print("📡 Broadcasting the global model...")
        self.broadcast_model()

        # Phase 1: Preparation
        print("\n🔧 Phase 1: Client Preparation")
        for client in self.clients:
            client.set_round(round_num)
            if isinstance(client, AttackerClient):
                client.prepare_for_round(round_num)

        # Phase 2: Local Training
        print("\n💪 Phase 2: Local Training")
        initial_updates = {}
        for client in self.clients:
            update = client.local_train()
            initial_updates[client.client_id] = update
            print(f"  ✓ Client {client.client_id} completed training")

        # Phase 3: Attacker Camouflage
        print("\n🎭 Phase 3: Attacker Camouflage")
        benign_updates = []
        for client_id, update in initial_updates.items():
            if client_id < (len(self.clients) - sum(1 for c in self.clients if isinstance(c, AttackerClient))):
                benign_updates.append(update)

        final_updates = {}
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if isinstance(client, AttackerClient):
                client.receive_benign_updates(benign_updates)
                final_updates[client_id] = client.camouflage_update(update)
            else:
                final_updates[client_id] = update

        # Phase 4: Defense and Aggregation
        print("\n🛡️ Phase 4: Defense and Aggregation")
        final_update_list = [final_updates[cid] for cid in sorted(final_updates.keys())]
        client_id_list = sorted(final_updates.keys())

        defense_log = self.aggregate_updates(final_update_list, client_id_list)

        # Evaluate the global model
        clean_acc, attack_asr = self.evaluate()

        # Defense analysis
        print(f"\n📈 Defense Analysis:")
        print(f"  Dynamic Threshold: {defense_log['threshold']:.4f}")
        print(f"  Rejection Rate: {defense_log['rejection_rate']:.1%}")

        # Create log for the current round
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'attack_success_rate': attack_asr,
            'defense': defense_log,
            'stage': stage,
            'server_lr': self.server_lr
        }

        self.log_data.append(round_log)

        # Display results
        print(f"\n📊 Round {round_num + 1} Results:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        print(f"  Attack Success Rate: {attack_asr:.4f}")

        # Analyze ASR changes
        if len(self.history['asr']) > 1:
            asr_change = attack_asr - self.history['asr'][-2]
            if abs(asr_change) > 0.1:
                print(f"  ⚠️ ASR Change: {asr_change:+.2%}")

        return round_log