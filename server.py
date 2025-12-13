# server.py
# This module implements the Server class for federated learning, including model aggregation and defense mechanisms against GRMP attacks.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from client import BenignClient, AttackerClient
import torch.nn.functional as F


class Server:
    """Server class for federated learning with GRMP attack defense"""
    def __init__(self, global_model: nn.Module, test_loader,
                defense_threshold=0.4, total_rounds=20, server_lr=0.8, tolerance_factor=2,
                d_T=0.5, gamma=10.0, similarity_alpha=0.7):
        self.global_model = copy.deepcopy(global_model)
        self.test_loader = test_loader
        self.defense_threshold = defense_threshold
        self.total_rounds = total_rounds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        self.clients = []
        self.log_data = []

        # Additional stability parameters
        self.server_lr = server_lr  # Server learning rate (inertia)
        self.tolerance_factor = tolerance_factor  # Defense tolerance level
        
        # Formula 4 constraint parameters (passed to attackers)
        self.d_T = d_T  # Distance threshold for constraint (4b)
        self.gamma = gamma  # Upper bound for constraint (4c)
        self.similarity_alpha = similarity_alpha  # Weight for pairwise similarities

        # Track historical data for adaptive adjustments
        self.history = {
            'clean_acc': [],  # Clean accuracy
            'rejection_rates': [],  # Client rejection rates
            'local_accuracies': {}  # Local accuracies per client per round {client_id: [acc1, acc2, ...]}
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
        similarities = []
        for i in range(n_updates):
            mixed_sim = self.similarity_alpha * pairwise_sims[i] + (1 - self.similarity_alpha) * avg_sims[i]
            similarities.append(mixed_sim)

        similarities = np.array(similarities)

        # Print information
        print(f"  📈 Mixed Similarity - Mean: {similarities.mean():.3f}, "
              f"Std Dev: {similarities.std():.3f}")

        # Display similarity for each client
        # Note: similarities are ordered by updates, which match client_ids order from aggregate_updates
        attacker_ids = {client.client_id for client in self.clients if getattr(client, 'is_attacker', False)}
        for i, sim in enumerate(similarities):
            if hasattr(self, '_sorted_client_ids') and i < len(self._sorted_client_ids):
                client_id = self._sorted_client_ids[i]
                client = next((c for c in self.clients if c.client_id == client_id), None)
                if client:
                    client_type = "Attacker" if getattr(client, 'is_attacker', False) else "Benign"
                    print(f"    Client {client_id} ({client_type}): {sim:.3f}")
                else:
                    print(f"    Client {client_id}: {sim:.3f}")
            else:
                print(f"    Update {i}: {sim:.3f}")

        return similarities

    def aggregate_updates(self, updates: List[torch.Tensor],
                          client_ids: List[int]) -> Dict:
        # Store client_ids for similarity display
        self._current_client_ids = client_ids
        self._sorted_client_ids = client_ids
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

        # Aggregate updates from accepted clients
        aggregated_update_norm = 0.0
        if accepted_indices:
            accepted_updates = [updates[i] for i in accepted_indices]
            # Weighted aggregation by claimed data sizes (paper: D_i/D(t)).
            # For benign clients, try to use their data_indices length if available.
            weights = []
            for i in accepted_indices:
                cid = client_ids[i]
                client = self.clients[cid]
                if getattr(client, 'is_attacker', False):
                    w = getattr(client, 'claimed_data_size', 1.0)
                else:
                    w = len(getattr(client, 'data_indices', [])) or 1.0
                weights.append(w)
            weight_tensor = torch.tensor(weights, device=self.device, dtype=accepted_updates[0].dtype)
            weight_tensor = weight_tensor / weight_tensor.sum()
            stacked = torch.stack(accepted_updates)
            aggregated_update = (stacked * weight_tensor.view(-1, 1)).sum(dim=0)
            aggregated_update_norm = torch.norm(aggregated_update).item()

            # Smooth the global model update using server learning rate (key improvement)
            current_params = self.global_model.get_flat_params()
            new_params = current_params + self.server_lr * aggregated_update
            self.global_model.set_flat_params(new_params)

            print(f"  📊 Update Stats: Accepted {len(accepted_indices)}/{len(updates)} updates")
            print(f"  🔧 Server Learning Rate: {self.server_lr} (Smooth updates)")
            print(f"  📐 Aggregated update norm: {aggregated_update_norm:.6f}")
        else:
            print("  ⚠️ Warning: No updates were accepted this round!")

        defense_log = {
            'similarities': similarities.tolist(),
            'accepted_clients': [client_ids[i] for i in accepted_indices],
            'rejected_clients': [client_ids[i] for i in rejected_indices],
            'threshold': dynamic_threshold,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'tolerance_factor': self.tolerance_factor,
            'rejection_rate': rejection_rate,
            'aggregated_update_norm': aggregated_update_norm
        }

        return defense_log

    def evaluate_local_accuracy(self, client) -> float:
        """
        Evaluate local model accuracy for a specific client.
        Uses the global test set for fair comparison across clients.
        """
        client.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Use global test loader for fair comparison (same test set for all clients)
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = client.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def evaluate(self) -> float:
        """
        Evaluate the global model's performance.
        
        Returns:
            Clean accuracy (float) on the test set
        """
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

        # Record historical metrics
        self.history['clean_acc'].append(clean_accuracy)

        return clean_accuracy

    def adaptive_adjustment(self, round_num: int):
        """Adaptively adjust parameters based on historical performance."""
        # Fixed server_lr (no adaptive change)
        pass

    def run_round(self, round_num: int) -> Dict:
        """Execute one round of federated learning - stable version."""
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # Adaptive adjustment
        self.adaptive_adjustment(round_num)

        # Display current parameters (no hard-coded stage logic)
        print(f"Current Parameters: server_lr={self.server_lr:.2f}, tolerance={self.tolerance_factor:.1f}")
        print(f"{'=' * 60}")

        # Broadcast the model
        print("📡 Broadcasting the global model...")
        self.broadcast_model()
        
        # Set global model params and constraint parameters for attackers (Formula 4)
        global_params = self.global_model.get_flat_params()
        for client in self.clients:
            if isinstance(client, AttackerClient):
                client.set_global_model_params(global_params)
                # Set constraint parameters: d_T and gamma
                # d_T: distance threshold for proximity constraint (4b)
                # gamma: upper bound for aggregation distance constraint (4c)
                client.set_constraint_params(d_T=self.d_T, gamma=self.gamma)

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
            client = self.clients[client_id]
            if not getattr(client, 'is_attacker', False):
                benign_updates.append(update)
        
        print(f"  Captured {len(benign_updates)} benign updates for camouflage.")
        
        final_updates = {}
        for client_id, update in initial_updates.items():
            client = self.clients[client_id]
            if getattr(client, 'is_attacker', False):
                print(f"  ⚠️ Triggering camouflage logic for Client {client_id}")
                client.receive_benign_updates(benign_updates)
                final_updates[client_id] = client.camouflage_update(update)
            else:
                final_updates[client_id] = update

        # Phase 4: Defense and Aggregation
        print("\n🛡️ Phase 4: Defense and Aggregation")
        # Ensure deterministic order of keys
        sorted_client_ids = sorted(final_updates.keys())
        final_update_list = [final_updates[cid] for cid in sorted_client_ids]
        
        defense_log = self.aggregate_updates(final_update_list, sorted_client_ids)

        # Evaluate the global model
        clean_acc = self.evaluate()
        
        # Evaluate local accuracies for each client
        local_accs_this_round = {}
        for client in self.clients:
            try:
                local_acc = self.evaluate_local_accuracy(client)
                local_accs_this_round[client.client_id] = local_acc
                
                # Update history
                if client.client_id not in self.history['local_accuracies']:
                    self.history['local_accuracies'][client.client_id] = []
                self.history['local_accuracies'][client.client_id].append(local_acc)
            except Exception as e:
                # Skip if evaluation fails (e.g., empty data loader)
                print(f"  ⚠️  Could not evaluate local accuracy for client {client.client_id}: {e}")

        # Defense analysis
        print(f"\n📈 Defense Analysis:")
        print(f"  Dynamic Threshold: {defense_log['threshold']:.4f}")
        print(f"  Rejection Rate: {defense_log['rejection_rate']:.1%}")

        # Create log for the current round
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'acc_diff': (abs(clean_acc - self.history['clean_acc'][-2])
                         if len(self.history['clean_acc']) > 1 else 0.0),
            'defense': defense_log,
            'server_lr': self.server_lr,
            'local_accuracies': local_accs_this_round  # Add local accuracies
        }

        self.log_data.append(round_log)

        # Display results
        print(f"\n📊 Round {round_num + 1} Results:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        # Show performance change
        if len(self.history['clean_acc']) > 1:
            prev_clean = self.history['clean_acc'][-2]
            delta_prev = clean_acc - prev_clean
            best_clean = max(self.history['clean_acc'])
            delta_best = clean_acc - best_clean
            print(f"  ΔClean vs prev: {delta_prev:+.4f}")
            print(f"  ΔClean vs best: {delta_best:+.4f}")

        return round_log