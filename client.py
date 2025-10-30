# client.py provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE


# Client class for federated learning
class Client:

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
        """Reset the optimizer."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Calculate the model update."""
        current_params = self.model.get_flat_params()
        return current_params - initial_params



    def evaluate_local(self, loader=None) -> Dict[str, float]:
        eval_loader = loader if loader is not None else self.data_loader
        m_eval = copy.deepcopy(self.model).to(self.device)
        m_eval.eval()
        correct = total = 0
        loss_sum = 0.0; n_batches = 0
        ce = nn.CrossEntropyLoss(reduction='mean')
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = m_eval(input_ids, attention_mask)
                loss = ce(logits, labels)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loss_sum += loss.item(); n_batches += 1
        del m_eval  # 释放副本
        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / n_batches if n_batches > 0 else 0.0
        return {'acc': acc, 'loss': avg_loss, 'num_samples': total}



# BenignClient class for benign clients
class BenignClient(Client):

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - includes proximal regularization."""
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # Proximal regularization coefficient
        mu = 0.01  # Controls how far updates can deviate from the initial model

        
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
                
                # Add proximal regularization term
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
        # Benign clients do not use this method
        pass


# AttackerClient class for clients that perform attacks
class AttackerClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                data_indices, lr=0.001, local_epochs=2):
        self.data_manager = data_manager
        self.data_indices = data_indices

        dummy_loader = data_manager.get_attacker_data_loader(client_id, data_indices, 0)
        super().__init__(client_id, model, dummy_loader, lr, local_epochs)

        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []

        # Progressive attack parameters (adjusted for more subtle behavior)
        self.base_amplification = 1.2
        self.progressive_enabled = True
        self.beta = 0.2

        # Momentum mechanism (key improvement)
        self.momentum = 0.7 # Retain 70% of the historical attack direction
        self.prev_update = None
        self.prev_amplification = None

        # Adaptive parameters
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        self.similarity_target = 0.35  # Target similarity (mimicking benign clients)
        self.similarity_std = 0.08     # Standard deviation for similarity

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)
        self.data_loader = self.data_manager.get_attacker_data_loader(
            self.client_id, self.data_indices, round_num
        )

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Receive updates from benign clients."""
        self.benign_updates = updates

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - mild version."""
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()

        # Progressive learning rate adjustment (milder version)
        if self.progressive_enabled:
            if self.current_round < 3:
                effective_lr = self.lr * 0.6
            elif self.current_round < 7:
                effective_lr = self.lr * 0.7
            else:
                effective_lr = self.lr * 1.1

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = effective_lr

        # Training loop
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


        # Apply momentum (key stability improvement)
        if self.prev_update is not None:
            poisoned_update = self.momentum * self.prev_update + (1 - self.momentum) * poisoned_update
            print(f"    Attacker {self.client_id}: Applying momentum (momentum={self.momentum})")

        self.prev_update = poisoned_update.clone()

        return poisoned_update

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """Camouflage the update - balanced version (make attacker updates closer to benign users)."""
        if not self.benign_updates:
            print(f"    Attacker {self.client_id}: No benign updates available")
            return poisoned_update

        # Calculate statistical properties of benign updates
        benign_tensor = torch.stack(self.benign_updates)
        benign_mean = benign_tensor.mean(dim=0)
        benign_std = benign_tensor.std(dim=0)
        
        # Calculate similarity distribution among benign users
        benign_similarities = []
        for i, update in enumerate(self.benign_updates):
            sim = torch.cosine_similarity(update.unsqueeze(0), benign_mean.unsqueeze(0)).item()
            benign_similarities.append(sim)
        
        benign_sim_mean = np.mean(benign_similarities)
        benign_sim_std = np.std(benign_similarities)
        
        print(f"    Benign user similarity: {benign_sim_mean:.3f} ± {benign_sim_std:.3f}")

        # Progressive amplification factor
        if self.progressive_enabled:
            if self.current_round < 5:
                amplification_factor = self.base_amplification * 0.4
                self.beta = 0.3
                target_similarity = benign_sim_mean - 0.5 * benign_sim_std
            elif self.current_round < 10:
                amplification_factor = self.base_amplification * 0.6
                self.beta = 0.4
                target_similarity = benign_sim_mean - 1 * benign_sim_std
            elif self.current_round < 15:
                amplification_factor = self.base_amplification * 0.8
                self.beta = 0.5
                target_similarity = benign_sim_mean - 1.5 * benign_sim_std
            else:
                amplification_factor = self.base_amplification * 1.0
                self.beta = 0.6
                target_similarity = benign_sim_mean - 0.5 * benign_sim_std
        else:
            amplification_factor = self.base_amplification
            target_similarity = benign_sim_mean

        # Smooth the amplification factor changes (retain the original smoothing mechanism)
        if self.prev_amplification is not None:
            amplification_factor = 0.7 * self.prev_amplification + 0.3 * amplification_factor
        self.prev_amplification = amplification_factor

        # Step 1: Amplify the poisoned signal
        v_malicious = poisoned_update * amplification_factor

        # Step 2: Test the current similarity
        current_sim = torch.cosine_similarity(v_malicious.unsqueeze(0), benign_mean.unsqueeze(0)).item()
        
        # Step 3: Adaptively adjust to achieve the target similarity
        if current_sim > target_similarity + 0.15:
            noise_factor = (current_sim - target_similarity) / 2
            noise = torch.randn_like(v_malicious) * benign_std * noise_factor
            v_malicious = v_malicious + noise
            print(f"    Reduce similarity: {current_sim:.3f} -> ", end='')
            current_sim = torch.cosine_similarity(v_malicious.unsqueeze(0), benign_mean.unsqueeze(0)).item()
            print(f"{current_sim:.3f}")
        
        # Step 4: Find the closest benign neighbor (retain original logic)
        best_neighbor = None
        neighbor_sims = []
        for benign_update in self.benign_updates:
            sim = torch.cosine_similarity(v_malicious.unsqueeze(0), benign_update.unsqueeze(0)).item()
            neighbor_sims.append(sim)
            if best_neighbor is None or sim > max(neighbor_sims[:-1]):
                best_neighbor = benign_update

        if best_neighbor is None:
            return v_malicious

        # Step 5: Orthogonal decomposition
        dot_product = torch.dot(v_malicious, v_malicious)
        if dot_product == 0:
            return v_malicious

        proj_v_malicious = (torch.dot(best_neighbor, v_malicious) / dot_product) * v_malicious
        v_orthogonal = best_neighbor - proj_v_malicious

        # Step 6: Dynamically adjust beta to control final similarity
        # Construct a candidate update
        candidate_update = v_malicious + self.beta * v_orthogonal
        candidate_sim = torch.cosine_similarity(candidate_update.unsqueeze(0), benign_mean.unsqueeze(0)).item()
        
        # 如If the candidate similarity is still too high, adjust beta
        if candidate_sim > target_similarity + 0.1:
            # Perform binary search to find a suitable beta
            beta_low, beta_high = 0.0, self.beta
            for _ in range(5):
                beta_mid = (beta_low + beta_high) / 2
                test_update = v_malicious + beta_mid * v_orthogonal
                test_sim = torch.cosine_similarity(test_update.unsqueeze(0), benign_mean.unsqueeze(0)).item()
                
                if test_sim > target_similarity:
                    beta_high = beta_mid
                else:
                    beta_low = beta_mid
            
            adjusted_beta = (beta_low + beta_high) / 2
            camouflaged_update = v_malicious + adjusted_beta * v_orthogonal
            print(f"    Adjusting beta: {self.beta:.2f} -> {adjusted_beta:.2f}")
        else:
            camouflaged_update = candidate_update

        # Step 7: Add slight benign noise (make it more like benign updates)
        benign_noise = torch.randn_like(camouflaged_update) * benign_std * 0.05
        camouflaged_update = camouflaged_update + benign_noise

        # Logging
        original_norm = torch.norm(poisoned_update).item()
        final_norm = torch.norm(camouflaged_update).item()
        
        final_sim_with_mean = torch.cosine_similarity(
            camouflaged_update.unsqueeze(0),
            benign_mean.unsqueeze(0)
        ).item()
        
        direction_preservation = torch.cosine_similarity(
            v_malicious.unsqueeze(0),
            camouflaged_update.unsqueeze(0)
        ).item()

        print(f"    Attacker {self.client_id} - Round {self.current_round}:")
        print(f"    Amplification factor 放大因子: {amplification_factor:.1f}, beta: {self.beta:.1f}")
        print(f"    Norm 规范: {original_norm:.4f} -> {final_norm:.4f}")
        print(f"    Direction preservation 方向保持: {direction_preservation:.4f}")
        print(f"    Final similarity 最终相似度: {final_sim_with_mean:.4f} (目标: {target_similarity:.3f})")
        
        # Log the difference from the benign distribution
        sim_diff = abs(final_sim_with_mean - benign_sim_mean)
        print(f"    Difference from benign mean 与良性均值差异: {sim_diff:.3f}")

        return camouflaged_update
    
    
    def get_iid_random_factor(self, base_value=1.0):
        """Generate noise factors that follow IID characteristics"""
        # Generate bounded noise using a Beta distribution
        alpha, beta = 2, 2  # Shape parameters
        beta_sample = np.random.beta(alpha, beta)
        
        # Map the sample to the range
        noise = (beta_sample - 0.8) * 1
        
        # Add small Gaussian noise
        gaussian_noise = np.random.normal(0, 0.1)
        
        # Add client-specific offset
        client_offset = 0.02 * np.sin(self.client_id * 3.14)
        
        # Add round-related fluctuations
        round_wave = 0.03 * np.sin(self.current_round * 0.8 + self.client_id)
        
        total_noise = noise + gaussian_noise + client_offset + round_wave
        return base_value + total_noise

    def _construct_graph(self, updates: List[torch.Tensor]) -> tuple:
        """Construct a graph structure for updates"""
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
        """Train the VGAE model on the constructed graph."""
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