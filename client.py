# client.py - 稳定版本（带动量机制）

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE


class Client:
    """联邦学习客户端基类"""

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr=0.001, local_epochs=2):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.current_round = 0

    def reset_optimizer(self):
        """重置优化器"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """设置当前轮次"""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """计算模型更新"""
        current_params = self.model.get_flat_params()
        return current_params - initial_params


class BenignClient(Client):
    """良性客户端"""

    def prepare_for_round(self, round_num: int):
        """良性客户端不需要特殊准备"""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """执行本地训练 - 添加近端正则化"""
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # 近端正则化系数
        mu = 0.01  # 控制更新不要偏离初始模型太远
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(self.data_loader,
                    desc=f'Client {self.client_id} - Epoch {epoch + 1}/{epochs}',
                    leave=False)
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                ce_loss = nn.CrossEntropyLoss()(outputs, labels)
                
                # 添加近端正则化项
                current_params = self.model.get_flat_params()
                proximal_term = mu * torch.norm(current_params - initial_params) ** 2
                
                loss = ce_loss + proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        return self.get_model_update(initial_params)

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """良性客户端不使用此方法"""
        pass


class AttackerClient(Client):
    """恶意客户端 - 稳定版本"""

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr=0.001, local_epochs=2):
        self.data_manager = data_manager
        self.data_indices = data_indices

        dummy_loader = data_manager.get_attacker_data_loader(client_id, data_indices, 0)
        super().__init__(client_id, model, dummy_loader, lr, local_epochs)

        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []

        # 渐进式攻击参数（调整为更温和）
        self.base_amplification = 1.2  # 降低基础放大因子
        self.progressive_enabled = True
        self.beta = 0.2

        # 动量机制（关键改进）
        self.momentum = 0.7  # 保持70%的历史攻击方向
        self.prev_update = None
        self.prev_amplification = None

        # 自适应参数
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        self.similarity_target = 0.35  # 目标相似度（模仿良性客户端）
        self.similarity_std = 0.08     # 相似度标准差

    def prepare_for_round(self, round_num: int):
        """为新轮次准备"""
        self.set_round(round_num)
        self.data_loader = self.data_manager.get_attacker_data_loader(
            self.client_id, self.data_indices, round_num
        )

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """接收良性更新"""
        self.benign_updates = updates

    def local_train(self, epochs=None) -> torch.Tensor:
        """执行本地训练 - 温和版本"""
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()

        # 渐进式学习率调整（更温和）
        if self.progressive_enabled:
            if self.current_round < 3:
                effective_lr = self.lr * 0.6
            elif self.current_round < 7:
                effective_lr = self.lr * 0.7
            else:
                effective_lr = self.lr * 1.1  # 不要过度增加

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = effective_lr

        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            pbar = tqdm(self.data_loader,
                        desc=f'Attacker {self.client_id} - Round {self.current_round} - Epoch {epoch + 1}/{epochs}',
                        leave=False)

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': loss.item()})

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches

        poisoned_update = self.get_model_update(initial_params)

        # 应用动量（关键稳定性改进）
        if self.prev_update is not None:
            poisoned_update = self.momentum * self.prev_update + (1 - self.momentum) * poisoned_update
            print(f"    Attacker {self.client_id}: 应用动量 (momentum={self.momentum})")

        self.prev_update = poisoned_update.clone()

        return poisoned_update

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """伪装更新 - 温和稳定版本"""
        if not self.benign_updates:
            print(f"    Attacker {self.client_id}: 无良性更新可用")
            return poisoned_update
        # 计算良性更新的统计特性
        benign_mean = torch.stack(self.benign_updates).mean(dim=0)
        benign_std = torch.stack(self.benign_updates).std(dim=0).mean().item()
        
        # 计算当前与良性均值的相似度
        current_sim = torch.cosine_similarity(
            poisoned_update.unsqueeze(0),
            benign_mean.unsqueeze(0)
        ).item()

        # 渐进式放大因子（更温和）
        if self.progressive_enabled:
            if self.current_round < 3:
                amplification_factor = self.base_amplification * 0.8
                self.beta = 0.5  # 降低from 0.6
            elif self.current_round < 5:
                amplification_factor = self.base_amplification * 0.9
                self.beta = 0.5  # 降低from 0.7
            elif self.current_round < 8:
                amplification_factor = self.base_amplification * 0.9
                self.beta = 0.5  # 降低from 0.8
            else:
                amplification_factor = self.base_amplification * 0.9
                self.beta = 0.5  # 降低from 0.9
        else:
            amplification_factor = self.base_amplification

        # 添加随机波动来模拟IID特性
        random_factor = self.get_iid_random_factor()
        amplification_factor = amplification_factor * random_factor
      

        # 平滑放大因子变化
        if self.prev_amplification is not None:
            amplification_factor = 0.7 * self.prev_amplification + 0.3 * amplification_factor
        self.prev_amplification = amplification_factor

        print(f"    Attacker {self.client_id} - Round {self.current_round}: "
              f"温和GRMP (amp={amplification_factor:.1f}, beta={self.beta:.1f})")

        # Step 1: 放大毒药信号
        v_malicious = poisoned_update * amplification_factor

        # Step 2: 找到最接近的良性邻居
        best_neighbor = None
        max_sim = -1
        for benign_update in self.benign_updates:
            sim = torch.cosine_similarity(v_malicious.unsqueeze(0), benign_update.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_neighbor = benign_update

        if best_neighbor is None:
            return v_malicious

        # Step 3: 正交分解
        dot_product = torch.dot(v_malicious, v_malicious)
        if dot_product == 0:
            return v_malicious

        proj_v_malicious = (torch.dot(best_neighbor, v_malicious) / dot_product) * v_malicious
        v_orthogonal = best_neighbor * 0.3 - proj_v_malicious

        # Step 4: 构建最终更新
        camouflaged_update = v_malicious + self.beta * v_orthogonal

        # 日志记录
        original_norm = torch.norm(poisoned_update).item()
        final_norm = torch.norm(camouflaged_update).item()

        benign_mean = torch.stack(self.benign_updates).mean(dim=0)
        final_sim_with_mean = torch.cosine_similarity(
            camouflaged_update.unsqueeze(0),
            benign_mean.unsqueeze(0)
        ).item()

        direction_preservation = torch.cosine_similarity(
            v_malicious.unsqueeze(0),
            camouflaged_update.unsqueeze(0)
        ).item()

        print(f"    规范: {original_norm:.4f} -> {final_norm:.4f}")
        print(f"    方向保持: {direction_preservation:.4f}")
        print(f"    最终相似度: {final_sim_with_mean:.4f}")

        return camouflaged_update
    
    
    def get_iid_random_factor(self, base_value=1.0):
        """生成符合IID特性的噪声因子"""
        # 使用Beta分布生成有界噪声（更自然）
        alpha, beta = 2, 2  # 形状参数
        beta_sample = np.random.beta(alpha, beta)
        
        # 映射到[-0.3, 0.3]范围
        noise = (beta_sample - 0.8) * 1
        
        # 添加小幅高斯噪声
        gaussian_noise = np.random.normal(0, 0.1)
        
        # 客户端特定的偏移（但要小）
        client_offset = 0.02 * np.sin(self.client_id * 3.14)
        
        # 轮次相关的波动
        round_wave = 0.03 * np.sin(self.current_round * 0.8 + self.client_id)
        
        total_noise = noise + gaussian_noise + client_offset + round_wave
        return base_value + total_noise

    def _construct_graph(self, updates: List[torch.Tensor]) -> tuple:
        """构建图结构"""
        n_updates = len(updates)
        max_features = 5000

        truncated_updates = [u[:max_features] for u in updates]
        feature_matrix = torch.stack(truncated_updates)

        adj_matrix = torch.zeros(n_updates, n_updates)

        for i in range(n_updates):
            for j in range(n_updates):
                if i != j:
                    sim = torch.cosine_similarity(
                        truncated_updates[i].unsqueeze(0),
                        truncated_updates[j].unsqueeze(0)
                    )
                    adj_matrix[i, j] = sim if sim > 0.5 else 0

        return adj_matrix, feature_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=10):
        """训练VGAE"""
        if self.vgae is None:
            input_dim = feature_matrix.shape[1]
            self.vgae = VGAE(input_dim, hidden_dim=128, latent_dim=64).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=0.01)

        adj_matrix = adj_matrix.to(self.device)
        feature_matrix = feature_matrix.to(self.device)

        self.vgae.train()
        for epoch in range(epochs):
            self.vgae_optimizer.zero_grad()

            adj_reconstructed, mu, logvar = self.vgae(feature_matrix, adj_matrix)

            loss = self.vgae.loss_function(adj_reconstructed, adj_matrix, mu, logvar)

            loss.backward()
            self.vgae_optimizer.step()
