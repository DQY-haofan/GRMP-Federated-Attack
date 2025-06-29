# server.py - 稳定版本（控制波动）

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from client import BenignClient, AttackerClient
import torch.nn.functional as F  # 添加这一行

class Server:
    """联邦学习服务器 - 增强稳定性版本"""

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

        # 新增稳定性参数
        self.server_lr = server_lr  # 服务器端学习率（惯性）
        self.tolerance_factor = tolerance_factor  # 防御宽容度

        # 跟踪历史信息用于自适应
        self.history = {
            'asr': [],
            'clean_acc': [],
            'rejection_rates': []
        }

    def register_client(self, client):
        """注册客户端"""
        self.clients.append(client)

    def broadcast_model(self):
        """广播全局模型到所有客户端"""
        global_params = self.global_model.get_flat_params()
        for client in self.clients:
            client.model.set_flat_params(global_params.clone())
            client.reset_optimizer()

    def _compute_similarities(self, updates: List[torch.Tensor]) -> np.ndarray:
        """计算更新之间的余弦相似度 - 调整版本"""
        update_matrix = torch.stack(updates)
        
        # 使用加权平均而不是简单平均
        # 给予norm较大的更新更多权重（它们通常更稳定）
        norms = torch.norm(update_matrix, dim=1)
        weights = F.softmax(norms, dim=0)
        avg_update = torch.sum(update_matrix * weights.unsqueeze(1), dim=0)
        
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
        聚合更新 - 增强稳定性版本
        使用更宽容的防御机制和平滑的更新策略
        """
        similarities = self._compute_similarities(updates)

        # 计算动态阈值（更宽容）
        mean_sim = similarities.mean()
        std_sim = similarities.std()

        # 使用tolerance_factor让阈值更宽容
        dynamic_threshold = max(self.defense_threshold,
                                mean_sim - self.tolerance_factor * std_sim)

        # 自适应调整：如果拒绝率过高，进一步降低阈值
        if len(self.history['rejection_rates']) > 0:
            recent_rejection_rate = np.mean(self.history['rejection_rates'][-3:])
            if recent_rejection_rate > 0.4:  # 如果40%以上被拒绝
                dynamic_threshold *= 0.9  # 降低10%的阈值
                print(f"  ⚠️ 高拒绝率检测，降低阈值至: {dynamic_threshold:.3f}")

        accepted_indices = []
        rejected_indices = []

        for i, sim in enumerate(similarities):
            if sim >= dynamic_threshold:
                accepted_indices.append(i)
            else:
                rejected_indices.append(i)

        # 记录拒绝率
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

        # 聚合接受的更新
        if accepted_indices:
            accepted_updates = [updates[i] for i in accepted_indices]
            aggregated_update = torch.stack(accepted_updates).mean(dim=0)

            # 使用服务器学习率进行平滑更新（关键改进）
            current_params = self.global_model.get_flat_params()
            new_params = current_params + self.server_lr * aggregated_update
            self.global_model.set_flat_params(new_params)

            print(f"  📊 更新统计: 接受 {len(accepted_indices)}/{len(updates)} 个更新")
            print(f"  🔧 服务器学习率: {self.server_lr} (平滑更新)")
        else:
            print("  ⚠️ 警告: 本轮没有更新被接受!")

        return defense_log

    def evaluate(self) -> Tuple[float, float]:
        """评估模型性能"""
        self.global_model.eval()

        # 评估clean accuracy
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

        # 评估Attack Success Rate
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

        # 记录历史
        self.history['asr'].append(attack_success_rate)
        self.history['clean_acc'].append(clean_accuracy)

        return clean_accuracy, attack_success_rate

    def adaptive_adjustment(self, round_num: int):
        """根据历史表现自适应调整参数"""
        if len(self.history['asr']) < 2:
            return

        # 计算ASR变化
        asr_change = self.history['asr'][-1] - self.history['asr'][-2]
        current_asr = self.history['asr'][-1]

        # 如果ASR波动过大，调整服务器学习率
        if abs(asr_change) > 0.60:  # 波动超过15%
            self.server_lr = max(0.5, self.server_lr * 0.9)  # 降低学习率
            print(f"  🔄 检测到大幅波动，降低服务器学习率至: {self.server_lr:.2f}")
        elif abs(asr_change) < 0.05 and round_num > 5:  # 稳定后可以加速
            self.server_lr = min(0.95, self.server_lr * 1.2)
            print(f"  🔄 系统稳定，提高服务器学习率至: {self.server_lr:.2f}")

    def run_round(self, round_num: int) -> Dict:
        """执行一轮联邦学习 - 稳定版本"""
        print(f"\n{'=' * 60}")
        print(f"Round {round_num + 1}/{self.total_rounds}")

        # 自适应调整
        self.adaptive_adjustment(round_num)

        # 显示当前阶段
        if round_num < 5:
            stage = "🌱 早期 (建立信任)"
        elif round_num < 10:
            stage = "🌿 成长期 (逐步增强)"
        elif round_num < 15:
            stage = "🌳 成熟期 (稳定攻击)"
        else:
            stage = "🔥 后期 (持续压力)"

        print(f"攻击阶段: {stage}")
        print(f"当前参数: server_lr={self.server_lr:.2f}, tolerance={self.tolerance_factor:.1f}")
        print(f"{'=' * 60}")

        # 广播模型
        print("📡 广播全局模型...")
        self.broadcast_model()

        # Phase 1: 准备
        print("\n🔧 Phase 1: 客户端准备")
        for client in self.clients:
            client.set_round(round_num)
            if isinstance(client, AttackerClient):
                client.prepare_for_round(round_num)

        # Phase 2: 本地训练
        print("\n💪 Phase 2: 本地训练")
        initial_updates = {}
        for client in self.clients:
            update = client.local_train()
            initial_updates[client.client_id] = update
            print(f"  ✓ 客户端 {client.client_id} 完成训练")

        # Phase 3: 攻击者伪装
        print("\n🎭 Phase 3: 攻击者伪装")
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

        # Phase 4: 防御和聚合
        print("\n🛡️ Phase 4: 防御和聚合")
        final_update_list = [final_updates[cid] for cid in sorted(final_updates.keys())]
        client_id_list = sorted(final_updates.keys())

        defense_log = self.aggregate_updates(final_update_list, client_id_list)

        # 评估
        clean_acc, attack_asr = self.evaluate()

        # 分析
        print(f"\n📈 防御分析:")
        print(f"  动态阈值: {defense_log['threshold']:.4f}")
        print(f"  拒绝率: {defense_log['rejection_rate']:.1%}")

        # 创建轮次日志
        round_log = {
            'round': round_num + 1,
            'clean_accuracy': clean_acc,
            'attack_success_rate': attack_asr,
            'defense': defense_log,
            'stage': stage,
            'server_lr': self.server_lr
        }

        self.log_data.append(round_log)

        # 显示结果
        print(f"\n📊 Round {round_num + 1} 结果:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        print(f"  Attack Success Rate: {attack_asr:.4f}")

        # ASR变化分析
        if len(self.history['asr']) > 1:
            asr_change = attack_asr - self.history['asr'][-2]
            if abs(asr_change) > 0.1:
                print(f"  ⚠️ ASR变化: {asr_change:+.2%}")

        return round_log
