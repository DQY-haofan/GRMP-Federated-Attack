# attack_baseline_alie.py
# ALIE (A Little Is Enough) Attack Implementation - NeurIPS '19
# 
# This module implements the ALIE attack as a Model Poisoning baseline.
# It is completely isolated from the GRMP attack implementation.

import torch
import numpy as np
from typing import List, Optional
from scipy.stats import norm

# Import base Client class
from client import Client


class ALIEAttackerClient(Client):
    """
    ALIE (A Little Is Enough) Attack - NeurIPS '19
    
    Simple statistical attack that generates malicious updates based on
    mean and standard deviation of benign updates.
    
    Paper: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
    URL: https://proceedings.neurips.cc/paper_files/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html
    
    Formula: attack_vec = mean(benign_updates) + z_max * std(benign_updates)
    
    where z_max is computed to ensure statistical plausibility:
        s = floor(num_clients/2 + 1) - num_attackers
        cdf_value = (num_clients - num_attackers - s) / (num_clients - num_attackers)
        z_max = norm.ppf(cdf_value)
    
    This attack is:
    - Data-agnostic: Does not modify training data
    - Statistically plausible: Based on benign update statistics
    - Simple and efficient: No optimization required
    """
    
    def __init__(self, client_id: int, model, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 num_clients: int, num_attackers: int,
                 z_max: Optional[float] = None,
                 attack_start_round: Optional[int] = None,
                 claimed_data_size: float = 1.0,
                 grad_clip_norm: float = 1.0):
        """
        Initialize ALIE attacker client.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance (for proxy loader, if needed)
            data_indices: List of data indices assigned to this client (not used for training)
            lr: Learning rate (not used, but required for Client base class)
            local_epochs: Number of local epochs (not used, but required for Client base class)
            alpha: Proximal regularization coefficient (not used, but required for Client base class)
            num_clients: Total number of federated learning clients
            num_attackers: Number of attacker clients
            z_max: Z-score multiplier. If None, computed automatically based on num_clients and num_attackers
            attack_start_round: Round number to start attack (None = start immediately)
            claimed_data_size: Data size to claim for weighted aggregation
            grad_clip_norm: Gradient clipping norm (not used, but kept for interface compatibility)
        """
        # Initialize base Client class
        # Note: data_loader=None because ALIE attackers don't train
        super().__init__(client_id, model, data_loader=None, lr=lr, local_epochs=local_epochs, alpha=alpha)
        
        # Mark as attacker
        self.is_attacker = True
        self.attack_method = "ALIE"
        
        # Store configuration
        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.z_max = z_max
        self.attack_start_round = attack_start_round
        self.claimed_data_size = claimed_data_size
        self.grad_clip_norm = grad_clip_norm  # Not used, but kept for compatibility
        
        # Store benign updates (received from server)
        self.benign_updates = []
        self.benign_update_client_ids = []
        
        # Optional: proxy loader for evaluation (not used in ALIE, but kept for interface compatibility)
        self.proxy_loader = data_manager.get_proxy_eval_loader(sample_size=512) if data_manager else None
        
        # Track data indices (not used for training, but kept for compatibility)
        self.data_indices = data_indices or []
        
        # Get model parameter count for dimension validation (works on CPU model)
        # In LoRA mode, this will be the LoRA parameter count
        # In full fine-tuning mode, this will be all parameters
        self._flat_numel = int(self.model.get_flat_params().numel())
        
        # Check if model is in LoRA mode
        self.use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
    
    def receive_benign_updates(self, updates: List[torch.Tensor], 
                              client_ids: Optional[List[int]] = None):
        """
        Receive updates from benign clients.
        
        Args:
            updates: List of benign client updates
            client_ids: Optional list of client IDs corresponding to each update
        """
        # Store detached copies on CPU to save GPU memory
        self.benign_updates = [u.detach().clone().cpu() for u in updates]
        
        # Store corresponding client IDs
        if client_ids is not None:
            self.benign_update_client_ids = client_ids.copy()
        else:
            # Fallback: use indices as client IDs
            self.benign_update_client_ids = list(range(len(updates)))
    
    def local_train(self, epochs=None) -> torch.Tensor:
        """
        ALIE attackers do not perform local training (data-agnostic attack).
        
        Returns:
            Zero update tensor with correct dimension (LoRA params in LoRA mode, all params in full fine-tuning mode)
        """
        # Return zero update with correct dimension
        # get_flat_params() automatically handles LoRA vs full fine-tuning
        return torch.zeros(self.model.get_flat_params().numel())
    
    def prepare_for_round(self, round_num: int):
        """
        Prepare for a new training round.
        
        Args:
            round_num: Current round number
        """
        self.set_round(round_num)
    
    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Generate ALIE attack update.
        
        This method implements the core ALIE attack logic:
        1. Check if attack should start (based on attack_start_round)
        2. Compute z_max if not provided
        3. Calculate mean and std of benign updates
        4. Generate attack vector: attack_vec = mean + z_max * std
        
        Args:
            poisoned_update: Zero update (attackers don't train, so this is always zero)
        
        Returns:
            Malicious update vector generated using ALIE method
        """
        # Check if attack should start
        if self.attack_start_round is not None:
            if self.current_round < self.attack_start_round:
                return poisoned_update
        
        # Check if we have benign updates
        if not self.benign_updates:
            print(f"    [ALIE Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update
        
        # Validate benign update dimensions (critical for LoRA mode)
        for idx, update in enumerate(self.benign_updates):
            update_dim = int(update.numel())
            if update_dim != self._flat_numel:
                raise RuntimeError(
                    f"[ALIE Attacker {self.client_id}] Benign update dimension mismatch: "
                    f"update[{idx}] has {update_dim} params, expected {self._flat_numel} "
                    f"(LoRA mode: {self.use_lora}). "
                    f"This may indicate a mismatch between benign client updates and attacker model configuration."
                )
        
        # Compute z_max if not provided
        if self.z_max is None:
            # Standard ALIE z_max calculation
            s = np.floor(self.num_clients / 2 + 1) - self.num_attackers
            cdf_value = (self.num_clients - self.num_attackers - s) / \
                       (self.num_clients - self.num_attackers)
            
            # Ensure cdf_value is in valid range [0, 1]
            cdf_value = max(0.0, min(1.0, cdf_value))
            
            # Compute z_max using inverse CDF of standard normal distribution
            z_max = norm.ppf(cdf_value)
        else:
            z_max = self.z_max
        
        # Convert benign updates to numpy array
        # Ensure all updates are flattened and on CPU
        benign_np = []
        for update in self.benign_updates:
            # Flatten and convert to numpy
            update_flat = update.cpu().numpy().flatten()
            benign_np.append(update_flat)
        
        benign_array = np.array(benign_np)  # Shape: (num_benign, param_dim)
        
        # Calculate mean and standard deviation
        mean = np.mean(benign_array, axis=0)
        std = np.std(benign_array, axis=0, ddof=0)  # Population std (ddof=0)
        
        # Handle zero standard deviation (numerical stability)
        # If std is zero, set to small epsilon to avoid division issues
        std = np.where(std == 0, 1e-8, std)
        
        # Generate attack vector: attack_vec = mean + z_max * std
        attack_vec = mean + z_max * std
        
        # Convert back to torch.Tensor
        malicious_update = torch.from_numpy(attack_vec).float()
        
        # Validate attack vector dimension (critical for LoRA mode)
        attack_dim = int(malicious_update.numel())
        if attack_dim != self._flat_numel:
            raise RuntimeError(
                f"[ALIE Attacker {self.client_id}] Attack vector dimension mismatch: "
                f"generated {attack_dim} params, expected {self._flat_numel} "
                f"(LoRA mode: {self.use_lora}). "
                f"This indicates a bug in attack vector generation."
            )
        
        # Log attack generation info
        mean_norm = np.linalg.norm(mean)
        std_norm = np.linalg.norm(std)
        attack_norm = np.linalg.norm(attack_vec)
        
        lora_info = f"LoRA" if self.use_lora else "Full"
        print(f"    [ALIE Attacker {self.client_id}] Generated attack ({lora_info} mode): "
              f"z_max={z_max:.4f}, mean_norm={mean_norm:.4f}, "
              f"std_norm={std_norm:.4f}, attack_norm={attack_norm:.4f}, "
              f"num_benign={len(self.benign_updates)}, param_dim={self._flat_numel}")
        
        return malicious_update
    
    def receive_attacker_updates(self, updates: List[torch.Tensor], 
                                client_ids: List[int], 
                                data_sizes: Optional[dict] = None):
        """
        Receive updates from other attackers (for interface compatibility).
        
        ALIE attack does not use other attackers' updates, but we implement
        this method for interface compatibility with the server.
        
        Args:
            updates: List of other attacker updates (not used)
            client_ids: List of attacker client IDs (not used)
            data_sizes: Dictionary of attacker data sizes (not used)
        """
        # ALIE does not coordinate with other attackers
        # This method is kept for interface compatibility only
        pass
    
    def set_global_model_params(self, global_params: torch.Tensor):
        """
        Set global model parameters (for interface compatibility).
        
        ALIE attack does not use global model parameters, but we implement
        this method for interface compatibility.
        
        Args:
            global_params: Global model parameters (not used)
        """
        # ALIE does not need global model params
        # This method is kept for interface compatibility only
        pass
    
    def set_constraint_params(self, dist_bound: Optional[float] = None,
                              sim_bound_low: Optional[float] = None,
                              sim_bound_up: Optional[float] = None,
                              total_data_size: Optional[float] = None,
                              benign_data_sizes: Optional[dict] = None):
        """Set constraint parameters (for interface compatibility). ALIE does not use constraints."""
        pass
    
    def set_lagrangian_params(self, **kwargs):
        """
        Set Lagrangian parameters (for interface compatibility).
        
        ALIE attack does not use Lagrangian optimization, but we implement
        this method for interface compatibility.
        
        Args:
            **kwargs: Lagrangian parameters (not used)
        """
        # ALIE does not use Lagrangian optimization
        # This method is kept for interface compatibility only
        pass