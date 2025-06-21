# server.py - 支持TPU/GPU的版本
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from client import BenignClient, AttackerClient
from device_utils import device_manager


class Server:
    """Federated learning server with TPU/GPU support and defense mechanisms"""

    def __init__(self, global_model: nn.Module, test_loader, attack_test_loader,
                 defense_threshold=0.5, total_rounds=20):
        self.global_model = copy.deepcopy(global_model)
        self.test_loader = test_loader
        self.attack_test_loader = attack_test_loader
        self.defense_threshold = defense_threshold
        self.total_rounds = total_rounds
        self.device = device_manager.get_device()
        self.global_model = device_manager.move_to_device(self.global_model)
        self.clients = []
        self.log_data = []

        # Track attack progression
        self.attack_progression = {
            'poison_rates': [],
            'amplification_factors': [],
            'detection_rates': []
        }

        # Mixed precision for GPU
        self.use_amp = device_manager.is_gpu()

    def register_client(self, client):
        """Register a client with the server"""
        self.clients.append(client)

    def broadcast_model(self):
        """Send global model to all clients and reset their optimizers"""
        global_params = self.global_model.get_flat_params()

        for client in self.clients:
            # 确保参数在正确的设备上
            client_params = global_params.clone()
            if hasattr(client, 'device'):
                client_params = client_params.to(client.device)
            client.model.set_flat_params(client_params)
            client.reset_optimizer()

    def _compute_similarities(self, updates: List[torch.Tensor]) -> np.ndarray:
        """Compute cosine similarities between updates and their average"""
        # 确保所有更新在同一设备上
        updates = [u.to(self.device) for u in updates]
        update_matrix = torch.stack(updates)
        avg_update = update_matrix.mean(dim=0)

        similarities = []
        for update in updates:
            sim = torch.cosine_similarity(
                update.unsqueeze(0),
                avg_update.unsqueeze(0)
            ).item()
            similarities.append(sim)

        return np.array(similarities)

    def aggregate_updates(self, updates: List[torch.Tensor],
                          client_ids: List[int]) -> Dict:
        """
        Aggregate updates with defense mechanism
        Supports distributed aggregation for TPU
        """
        # 确保所有更新在服务器设备上
        updates = [u.to(self.device) for u in updates]

        similarities = self._compute_similarities(updates)

        # Dynamic threshold based on similarity distribution
        mean_sim = similarities.mean()
        std_sim = similarities.std()
        dynamic_threshold = max(self.defense_threshold, mean_sim - std_sim)

        accepted_indices = []
        rejected_indices = []

        for i, sim in enumerate(similarities):
            if sim >= dynamic_threshold:
                accepted_indices.append(i)
            else:
                rejected_indices.append(i)

        defense_log = {
            'similarities': similarities.tolist(),
            'accepted_clients': [client_ids[i] for i in accepted_indices],
            'rejected_clients': [client_ids[i] for i in rejected_indices],
            'threshold': dynamic_threshold,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim
        }

        # Aggregate accepted updates
        if accepted_indices:
            accepted_updates = [updates[i] for i in accepted_indices]
            aggregated_update = torch.stack(accepted_updates).mean(dim=0)

            # TPU分布式聚合
            if device_manager.is_tpu():
                aggregated_update = device_manager.reduce_mean(aggregated_update)

            # Apply aggregated update to global model
            current_params = self.global_model.get_flat_params()
            new_params = current_params + aggregated_update
            self.global_model.set_flat_params(new_params)
        else:
            print("WARNING: No updates accepted in this round!")

        return defense_log

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model performance with TPU/GPU optimization
        1. Clean accuracy on full test set
        2. Attack Success Rate (ASR) on targeted samples
        """
        self.global_model.eval()

        # 创建并行数据加载器（TPU优化）
        if device_manager.is_tpu():
            test_para_loader = device_manager.create_parallel_loader(self.test_loader)
            test_loader = test_para_loader.per_device_loader(self.device)
        else:
            test_loader = self.test_loader

        # Evaluate clean accuracy
        correct = 0
        total = 0
        class_predictions = {0: 0, 1: 0, 2: 0, 3: 0}

        with torch.no_grad():
            # 混合精度评估（GPU优化）
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.global_model(input_ids, attention_mask)
                        predictions = torch.argmax(outputs, dim=1)

                        for pred in predictions:
                            class_predictions[pred.item()] = class_predictions.get(pred.item(), 0) + 1

                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
            else:
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.global_model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs, dim=1)

                    for pred in predictions:
                        class_predictions[pred.item()] = class_predictions.get(pred.item(), 0) + 1

                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                    # TPU同步
                    if device_manager.is_tpu():
                        device_manager.mark_step()

        clean_accuracy = correct / total if total > 0 else 0

        # Evaluate Attack Success Rate
        attack_success = 0
        attack_total = 0

        if self.attack_test_loader:
            # 创建并行数据加载器（TPU优化）
            if device_manager.is_tpu():
                attack_para_loader = device_manager.create_parallel_loader(self.attack_test_loader)
                attack_loader = attack_para_loader.per_device_loader(self.device)
            else:
                attack_loader = self.attack_test_loader

            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        for batch in attack_loader:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)

                            outputs = self.global_model(input_ids, attention_mask)
                            predictions = torch.argmax(outputs, dim=1)

                            attack_success += (predictions == 1).sum().item()
                            attack_total += len(predictions)
                else:
                    for batch in attack_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)

                        outputs = self.global_model(input_ids, attention_mask)
                        predictions = torch.argmax(outputs, dim=1)

                        attack_success += (predictions == 1).sum().item()
                        attack_total += len(predictions)

                        if device_manager.is_tpu():
                            device_manager.mark_step()

        attack_success_rate = attack_success / attack_total if attack_total > 0 else 0

        print(f"\nEvaluation Results:")
        print(f"  Clean Accuracy: {clean_accuracy:.4f} ({correct}/{total})")
        print(f"  Class predictions: World={class_predictions[0]}, Sports={class_predictions[1]}, "
              f"Business={class_predictions[2]}, Sci/Tech={class_predictions[3]}")
        print(f"  Attack Success Rate: {attack_success_rate:.4f} ({attack_success}/{attack_total})")

        return clean_accuracy, attack_success_rate

    def run_round(self, round_num: int) -> Dict:
        """
        Run one round of federated learning with TPU/GPU optimization
        """
        print(f"\n{'=' * 50}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # Show progressive attack stage
        if round_num < 5:
            print("Attack Stage: 🌱 Early (Building Trust)")
        elif round_num < 10:
            print("Attack Stage: 🌿 Growing (Increasing Impact)")
        elif round_num < 15:
            print("Attack Stage: 🌳 Mature (Strong Attack)")
        else:
            print("Attack Stage: 🔥 Full Force (Maximum Impact)")
        print(f"Device: {device_manager.device_type.upper()}")
        print(f"{'=' * 50}")

        # Broadcast model
        print("Broadcasting global model to clients...")
        self.broadcast_model()

        # Phase 1: Prepare clients for this round
        print("\nPhase 1: Preparing clients for round", round_num + 1)
        for client in self.clients:
            client.set_round(round_num)

            if isinstance(client, AttackerClient):
                client.prepare_for_round(round_num)
                print(f"  Attacker {client.client_id} prepared with progressive strategy")

        # Phase 2: All clients perform local training
        print("\nPhase 2: All clients perform local training")
        initial_updates = {}

        # 批量处理客户端更新（TPU优化）
        for client in self.clients:
            update = client.local_train()
            initial_updates[client.client_id] = update
            print(f"  Client {client.client_id} completed training")

        # Phase 3: Attackers camouflage their updates
        print("\nPhase 3: Attackers apply progressive GRMP camouflage")

        # Collect benign updates
        benign_updates = []
        benign_client_ids = []

        for client_id, update in initial_updates.items():
            if client_id < (len(self.clients) - sum(1 for c in self.clients if isinstance(c, AttackerClient))):
                benign_updates.append(update)
                benign_client_ids.append(client_id)

        # Process final updates
        final_updates = {}

        for client_id, update in initial_updates.items():
            client = self.clients[client_id]

            if isinstance(client, AttackerClient):
                client.receive_benign_updates(benign_updates)
                final_updates[client_id] = client.camouflage_update(update)
                print(f"  Attacker {client_id} generated progressive GRMP update")
            else:
                final_updates[client_id] = update

        # Phase 4: Defense and aggregation
        print("\nPhase 4: Server performs defense and aggregation")
        final_update_list = [final_updates[cid] for cid in sorted(final_updates.keys())]
        client_id_list = sorted(final_updates.keys())
        defense_log = self.aggregate_updates(final_update_list, client_id_list)

        # Evaluate
        clean_acc, attack_asr = self.evaluate()

        # Enhanced defense analysis with progressive tracking
        print(f"\nDefense Analysis:")
        print(f"  Dynamic Threshold: {defense_log['threshold']:.4f} "
              f"(mean: {defense_log['mean_similarity']:.4f}, std: {defense_log['std_similarity']:.4f})")

        num_attackers = sum(1 for c in self.clients if isinstance(c, AttackerClient))
        attacker_detected = 0

        for client_id, sim in zip(client_id_list, defense_log['similarities']):
            client_type = "ATTACKER" if client_id >= (len(self.clients) - num_attackers) else "Benign"
            status = "✓ Accepted" if client_id in defense_log['accepted_clients'] else "✗ REJECTED"

            if client_type == "ATTACKER" and status == "✗ REJECTED":
                attacker_detected += 1

            print(f"  Client {client_id} ({client_type}): Similarity={sim:.4f} {status}")

        # Track detection rate
        detection_rate = attacker_detected / num_attackers if num_attackers > 0 else 0
        self.attack_progression['detection_rates'].append(detection_rate)

        # Create enhanced round log
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'attack_success_rate': attack_asr,
            'defense': defense_log,
            'progressive_stage': self._get_stage_name(round_num),
            'detection_rate': detection_rate,
            'device': device_manager.device_type
        }

        self.log_data.append(round_log)

        # Progressive attack milestone alerts
        if attack_asr > 0.1 and round_num < 10:
            print(f"\n⚠️  EARLY BREAKTHROUGH: Attack achieving {attack_asr:.1%} success!")
        elif attack_asr > 0.5:
            print(f"\n🎯 MAJOR SUCCESS: Attack reached {attack_asr:.1%} success rate!")

        print(f"\nRound Summary:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        print(f"  Attack Success Rate: {attack_asr:.4f}")
        print(f"  Detection Rate: {detection_rate:.1%}")

        return round_log

    def _get_stage_name(self, round_num: int) -> str:
        """Get progressive attack stage name"""
        if round_num < 5:
            return "Early (Trust Building)"
        elif round_num < 10:
            return "Growing (Increasing Impact)"
        elif round_num < 15:
            return "Mature (Strong Attack)"
        else:
            return "Full Force (Maximum Impact)"
