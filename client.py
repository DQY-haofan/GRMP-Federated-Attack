# client.py
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
        self.is_attacker = False

    def reset_optimizer(self):
        """Reset the optimizer."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Calculate the model update (Current - Initial)."""
        current_params = self.model.get_flat_params()
        return current_params - initial_params

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


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
        
        # Proximal regularization coefficient (prevents model drifting too far)
        mu = 0.01

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
                # NewsClassifierModel returns logits directly
                logits = outputs
                
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                
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
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        
        # Attack hyperparameters
        self.base_amplification = 1.2
        self.vgae_lambda = 0.5 
        
        # --- 关键修改：降维维度 ---
        # 如果你的显存足够大 (e.g., A100 80G)，可以尝试调大这个值 (如 10000)
        # 对于 DistilBERT，5000 是一个兼顾速度和显存的安全值
        self.dim_reduction_size = 10000
        self.feature_indices = None

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)
        # Update dataloader with progressive poisoning logic
        self.data_loader = self.data_manager.get_attacker_data_loader(
            self.client_id, self.data_indices, round_num
        )

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Receive updates from benign clients."""
        # Store detached copies to avoid graph retention issues
        self.benign_updates = [u.detach().clone() for u in updates]

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training to get the initial malicious update."""
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # 1. Standard training on poisoned data
        for epoch in range(epochs):
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Get raw malicious update
        poisoned_update = self.get_model_update(initial_params)
        
        # 2. Apply VGAE-based camouflage
        final_update = self.camouflage_update(poisoned_update)
        
        return final_update

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Randomly selects indices to slice the high-dimensional vector.
        """
        stacked_updates = torch.stack(updates)
        total_dim = stacked_updates.shape[1]
        
        # 如果更新维度小于降维目标，则不降维
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # 每一轮攻击开始时，固定一组特征索引，保证这一轮内的训练一致性
        if self.feature_indices is None or not fix_indices:
            # Randomly select indices
            self.feature_indices = torch.randperm(total_dim)[:self.dim_reduction_size].to(self.device)
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _construct_graph(self, reduced_features: torch.Tensor):
        """
        Construct graph from REDUCED features.
        """
        # Normalize for cosine similarity
        norm_features = F.normalize(reduced_features, p=2, dim=1)
        
        # Compute adjacency matrix
        similarity_matrix = torch.mm(norm_features, norm_features.t())
        
        # Remove self-loops and binarize
        adj_matrix = similarity_matrix.clone()
        adj_matrix.fill_diagonal_(0)
        
        # Threshold 0.5 is empirical
        adj_matrix = (adj_matrix > 0.5).float() 
        
        return adj_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=30):
        """Train the VGAE model."""
        input_dim = feature_matrix.shape[1]
        
        # Initialize VGAE if dimensions match (lazy init)
        if self.vgae is None or self.vgae.gc1.weight.shape[0] != input_dim:
            # print(f"    [Attacker] Initializing VGAE with input_dim={input_dim}")
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=256, latent_dim=64).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=0.01)

        self.vgae.train()
        
        for _ in range(epochs):
            self.vgae_optimizer.zero_grad()
            
            # Forward pass
            adj_recon, mu, logvar = self.vgae(feature_matrix, adj_matrix)
            
            # Loss calculation
            loss = self.vgae.loss_function(adj_recon, adj_matrix, mu, logvar)
            
            loss.backward()
            self.vgae_optimizer.step()

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Main logic: Use VGAE to guide the modification of poisoned_update.
        """
        if not self.benign_updates:
            # Early round or no benign updates captured yet
            return poisoned_update

        # Reset feature indices for this camouflage session (new random projection)
        self.feature_indices = None
        
        # 1. Prepare Target (Malicious Update)
        target_update = poisoned_update.clone().detach().to(self.device)
        target_update.requires_grad_(True)
        
        # 2. Prepare Data (Benign + Malicious)
        # Note: We must concat first, THEN reduce dimension to ensure consistency
        with torch.no_grad():
            all_updates = self.benign_updates + [target_update]
            # Reduce dimensionality here (creates feature_indices)
            reduced_features = self._get_reduced_features(all_updates, fix_indices=False)
            adj_matrix = self._construct_graph(reduced_features)
        
        # 3. Train VGAE to learn the benign manifold structure
        # print(f"    [Attacker {self.client_id}] Training VGAE...")
        self._train_vgae(adj_matrix, reduced_features, epochs=20)
        
        # 4. Adversarial Optimization
        # Optimize 'target_update' so its latent representation looks 'benign'
        
        # Freeze VGAE
        for param in self.vgae.parameters():
            param.requires_grad = False
            
        optimizer_attack = optim.Adam([target_update], lr=0.1) # Higher LR for attack
        
        benign_indices = list(range(len(self.benign_updates)))
        malicious_index = len(self.benign_updates)

        # print(f"    [Attacker {self.client_id}] Optimizing Malicious Update...")
        
        for step in range(30):
            optimizer_attack.zero_grad()
            
            # Important: Must use the SAME reduction indices as training
            # We reconstruct the list with the current (grad-enabled) target_update
            current_list = [u.detach() for u in self.benign_updates] + [target_update]
            # Manual stacking to allow gradient flow from target_update
            # Since _get_reduced_features does torch.stack internally, we need to be careful
            # Let's do manual slicing here to keep gradients:
            
            stacked_current = torch.stack(current_list)
            if self.feature_indices is not None:
                current_features = torch.index_select(stacked_current, 1, self.feature_indices)
            else:
                current_features = stacked_current
            
            # Encode
            mu, _ = self.vgae.encode(current_features, adj_matrix)
            
            # Loss 1: Latent Distance (Make malicious look like benign center)
            benign_mu = mu[benign_indices]
            malicious_mu = mu[malicious_index]
            center_benign = torch.mean(benign_mu, dim=0)
            
            loss_latent = torch.norm(malicious_mu - center_benign) ** 2
            
            # Loss 2: Preservation (Don't lose the attack efficacy)
            # Use L2 distance in original high-dim space
            loss_preservation = torch.norm(target_update - poisoned_update.detach()) ** 2
            
            # Combined Loss
            total_loss = loss_latent + 0.5 * loss_preservation
            
            total_loss.backward()
            optimizer_attack.step()
            
        # Unfreeze VGAE
        for param in self.vgae.parameters():
            param.requires_grad = True
            
        # print(f"    [Attacker {self.client_id}] Camouflage Complete. Loss: {total_loss.item():.4f}")
        
        return target_update.detach()