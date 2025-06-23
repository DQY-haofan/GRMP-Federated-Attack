# client.py - 最终版本（正交噪声伪装）

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE


class Client:
    """Base class for federated learning clients"""

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
        """Re-initializes the optimizer. Should be called at the start of each round."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_round(self, round_num: int):
        """Set current round number for progressive strategies"""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Compute difference between current and initial parameters"""
        current_params = self.model.get_flat_params()
        return current_params - initial_params


class BenignClient(Client):
    """Benign client that performs honest training"""

    def prepare_for_round(self, round_num: int):
        """Benign clients don't need special preparation"""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training and return model update"""
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            pbar = tqdm(self.data_loader,
                        desc=f'Client {self.client_id} - Epoch {epoch + 1}/{epochs}',
                        leave=False)

            for batch in pbar:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': loss.item()})

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"    Client {self.client_id} - Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        return self.get_model_update(initial_params)

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Benign clients don't use this, but need the method for compatibility"""
        pass


class AttackerClient(Client):
    """Malicious client implementing GRMP with progressive attack strategy"""

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr=0.001, local_epochs=2):
        # Store data manager and indices for dynamic data loading
        self.data_manager = data_manager
        self.data_indices = data_indices

        # Initialize with dummy dataloader
        dummy_loader = data_manager.get_attacker_data_loader(client_id, data_indices, 0)
        super().__init__(client_id, model, dummy_loader, lr, local_epochs)

        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []

        # Progressive attack parameters
        self.base_amplification = 2.0
        self.progressive_enabled = True
        self.beta = 0.5  # New parameter for orthogonal noise strength

    def prepare_for_round(self, round_num: int):
        """Prepare attacker for new round with progressive poisoning"""
        self.set_round(round_num)

        # Get new dataloader with round-appropriate poisoning
        self.data_loader = self.data_manager.get_attacker_data_loader(
            self.client_id, self.data_indices, round_num
        )

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Store benign updates for graph construction"""
        self.benign_updates = updates

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform training with progressive attack intensity"""
        if epochs is None:
            epochs = self.local_epochs

        self.model.train()
        initial_params = self.model.get_flat_params().clone()

        # Progressive learning rate adjustment
        if self.progressive_enabled:
            if self.current_round < 5:
                effective_lr = self.lr * 0.5
            elif self.current_round < 10:
                effective_lr = self.lr * 0.8
            else:
                effective_lr = self.lr * 1.2

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
                print(f"    Attacker {self.client_id} - Round {self.current_round} - "
                      f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        poisoned_update = self.get_model_update(initial_params)
        return poisoned_update

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        [FINAL VERSION] New camouflage strategy: Direction Preservation with Orthogonal Noise.
        This decouples stealthiness from toxicity by preserving the malicious direction
        while adding benign-looking noise in orthogonal dimensions.
        """
        if not self.benign_updates:
            print(f"    Attacker {self.client_id}: No benign updates available")
            return poisoned_update

        # Progressive amplification and beta based on round
        if self.progressive_enabled:
            if self.current_round < 5:
                amplification_factor = self.base_amplification * 0.5
                self.beta = 0.3  # Less noise in early rounds
            elif self.current_round < 10:
                amplification_factor = self.base_amplification * 0.8
                self.beta = 0.5  # Moderate noise
            elif self.current_round < 15:
                amplification_factor = self.base_amplification * 1.2
                self.beta = 0.7  # More noise for better camouflage
            else:
                amplification_factor = self.base_amplification * 1.5
                self.beta = 0.8  # Maximum noise in later rounds
        else:
            amplification_factor = self.base_amplification

        print(f"    Attacker {self.client_id} - Round {self.current_round}: "
              f"Orthogonal GRMP (amp={amplification_factor:.1f}, beta={self.beta:.1f})")

        # Step 1: Amplify the poison signal to set the desired toxicity
        v_malicious = poisoned_update * amplification_factor
        print(f"    Amplifying poison signal with factor={amplification_factor}")
        
        # Step 2: Find the closest benign neighbor to use as a 'style reference'
        best_neighbor = None
        max_sim = -1
        for benign_update in self.benign_updates:
            sim = torch.cosine_similarity(v_malicious.unsqueeze(0), benign_update.unsqueeze(0)).item()
            if sim > max_sim:
                max_sim = sim
                best_neighbor = benign_update
                
        if best_neighbor is None:
            return v_malicious  # Failsafe

        # Step 3: Decompose the neighbor vector using Gram-Schmidt orthogonalization
        # This separates the neighbor into components parallel and orthogonal to our malicious direction
        
        # Avoid division by zero
        dot_product = torch.dot(v_malicious, v_malicious)
        if dot_product == 0:
            return v_malicious
            
        # Projection of best_neighbor onto v_malicious (the parallel component)
        proj_v_malicious = (torch.dot(best_neighbor, v_malicious) / dot_product) * v_malicious
        
        # The orthogonal component - this is our "benign noise" that won't interfere with attack direction
        v_orthogonal = best_neighbor - proj_v_malicious

        # Step 4: Construct the final update by adding scaled orthogonal noise
        camouflaged_update = v_malicious + self.beta * v_orthogonal

        # --- Logging for analysis ---
        original_norm = torch.norm(poisoned_update).item()
        amplified_norm = torch.norm(v_malicious).item()
        final_norm = torch.norm(camouflaged_update).item()
        
        benign_mean = torch.stack(self.benign_updates).mean(dim=0)
        final_sim_with_mean = torch.cosine_similarity(
            camouflaged_update.unsqueeze(0), 
            benign_mean.unsqueeze(0)
        ).item()
        
        # Calculate how much of the original direction is preserved
        direction_preservation = torch.cosine_similarity(
            v_malicious.unsqueeze(0),
            camouflaged_update.unsqueeze(0)
        ).item()
        
        print(f"    Original norm: {original_norm:.4f} -> Amplified: {amplified_norm:.4f} -> Final: {final_norm:.4f}")
        print(f"    Direction preservation: {direction_preservation:.4f}")
        print(f"    Final similarity with benign mean: {final_sim_with_mean:.4f}")

        return camouflaged_update

    def _construct_graph(self, updates: List[torch.Tensor]) -> tuple:
        """Construct adjacency matrix and feature matrix from updates"""
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
        """Train VGAE to learn benign update distribution"""
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
