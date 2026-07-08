# attack_baseline_gaussian.py
# Gaussian (Random Model Poisoning) Attack Implementation - USENIX Security '20
#
# This module implements the Gaussian attack as a Model Poisoning baseline.
# Attackers can collect benign updates; for each parameter j, we estimate
# a Gaussian distribution from benign updates and sample from it.
#
# Paper: "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
# Fang et al., USENIX Security 2020
# URL: https://www.usenix.org/conference/usenixsecurity20/presentation/fang
#
# Logic: For each jth parameter, estimate mean_j, std_j from benign updates;
#        malicious_update[j] = sample from N(mean_j, std_j²).
# This attack is used to show that random model poisoning does NOT effectively
# attack Byzantine-robust aggregation rules (lower-bound baseline).

import torch
import numpy as np
from typing import List, Optional

from client import Client


class GaussianAttackerClient(Client):
    """
    Gaussian (Random Model Poisoning) Attack - USENIX Security '20
    
    Randomly crafts malicious updates by sampling each parameter from
    a Gaussian distribution estimated from benign updates.
    
    Paper: "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    Fang et al., USENIX Security 2020
    
    Formula: For each j, attack_vec[j] ~ N(mean_j, std_j²)
    where mean_j = mean(benign_updates[:, j]), std_j = std(benign_updates[:, j])
    
    This attack is:
    - Data-agnostic: Does not modify training data
    - Random: No optimization, pure random sampling from estimated distribution
    - Lower-bound baseline: Paper shows it does NOT effectively attack
      Byzantine-robust aggregation (Krum, trimmed mean, median)
    """

    def __init__(self, client_id: int, model, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 attack_start_round: Optional[int] = None,
                 claimed_data_size: float = 1.0,
                 grad_clip_norm: float = 1.0,
                 gaussian_std_scale: float = 1.0):
        """
        Initialize Gaussian attacker client.

        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance (for interface compatibility)
            data_indices: List of data indices assigned (not used for training)
            lr: Learning rate (not used, but required for Client base class)
            local_epochs: Number of local epochs (not used, but required)
            alpha: Proximal coefficient (not used, but required)
            attack_start_round: Round to start attack (None = start immediately)
            claimed_data_size: Data size to claim for weighted aggregation
            grad_clip_norm: Gradient clipping norm (not used, interface compatibility)
            gaussian_std_scale: Scale factor for noise std. attack_vec ~ N(mean, (scale*std)²).
                               scale=1.0: original Fang et al. definition.
                               scale>1.0: expands noise range to increase attack impact (FedAvg).
        """
        super().__init__(
            client_id=client_id,
            model=model,
            data_loader=None,
            lr=lr,
            local_epochs=local_epochs,
            alpha=alpha
        )

        self.is_attacker = True
        self.attack_method = "Gaussian"
        self.attack_start_round = attack_start_round
        self.claimed_data_size = claimed_data_size
        self.grad_clip_norm = grad_clip_norm
        self.gaussian_std_scale = max(0.0, float(gaussian_std_scale))

        self.benign_updates: List[torch.Tensor] = []
        self.benign_update_client_ids: List[int] = []
        self.data_indices = data_indices or []

        self._flat_numel = int(self.model.get_flat_params().numel())
        self.use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora

    def receive_benign_updates(self, updates: List[torch.Tensor],
                              client_ids: Optional[List[int]] = None):
        """
        Receive updates from benign clients.

        Args:
            updates: List of benign client updates
            client_ids: Optional list of client IDs corresponding to each update
        """
        self.benign_updates = [u.detach().clone().cpu() for u in updates]
        if client_ids is not None:
            self.benign_update_client_ids = client_ids.copy()
        else:
            self.benign_update_client_ids = list(range(len(updates)))

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Gaussian attackers do not perform local training (data-agnostic attack).

        Returns:
            Zero update tensor with correct dimension
        """
        return torch.zeros(self.model.get_flat_params().numel())

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Generate Gaussian attack update.

        For each parameter j: sample from N(mean_j, std_j²) where mean_j and
        std_j are estimated from benign updates.

        Args:
            poisoned_update: Zero update (attackers don't train)

        Returns:
            Malicious update vector (random sample from estimated Gaussian)
        """
        if self.attack_start_round is not None:
            if self.current_round < self.attack_start_round:
                return poisoned_update

        if not self.benign_updates:
            print(f"    [Gaussian Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update

        # Validate dimensions (critical for LoRA mode)
        for idx, update in enumerate(self.benign_updates):
            update_dim = int(update.numel())
            if update_dim != self._flat_numel:
                raise RuntimeError(
                    f"[Gaussian Attacker {self.client_id}] Benign update dimension mismatch: "
                    f"update[{idx}] has {update_dim} params, expected {self._flat_numel} "
                    f"(LoRA mode: {self.use_lora})."
                )

        # Stack benign updates: (num_benign, param_dim)
        benign_np = np.array([u.cpu().numpy().flatten() for u in self.benign_updates])

        mean = np.mean(benign_np, axis=0)
        std = np.std(benign_np, axis=0, ddof=0)
        std = np.where(std == 0, 1e-8, std)  # Numerical stability

        # Sample from N(mean, (scale*std)²) per parameter.
        # scale=1.0: original Fang et al. definition.
        # scale>1.0: expands noise range to increase attack impact (FedAvg).
        effective_std = std * self.gaussian_std_scale
        attack_vec = np.random.normal(mean, effective_std)

        malicious_update = torch.from_numpy(attack_vec).float()

        if int(malicious_update.numel()) != self._flat_numel:
            raise RuntimeError(
                f"[Gaussian Attacker {self.client_id}] Attack vector dimension mismatch: "
                f"generated {malicious_update.numel()} params, expected {self._flat_numel}."
            )

        mean_norm = float(np.linalg.norm(mean))
        std_norm = float(np.linalg.norm(std))
        effective_std_norm = float(np.linalg.norm(effective_std))
        attack_norm = float(np.linalg.norm(attack_vec))
        lora_info = "LoRA" if self.use_lora else "Full"
        scale_info = f", std_scale={self.gaussian_std_scale}" if self.gaussian_std_scale != 1.0 else ""
        print(
            f"    [Gaussian Attacker {self.client_id}] Generated attack ({lora_info}): "
            f"mean_norm={mean_norm:.4f}, effective_std_norm={effective_std_norm:.4f}, "
            f"attack_norm={attack_norm:.4f}, num_benign={len(self.benign_updates)}{scale_info}"
        )

        return malicious_update

    def receive_attacker_updates(self, updates: List[torch.Tensor],
                                client_ids: List[int],
                                data_sizes: Optional[dict] = None):
        """Interface compatibility; Gaussian does not use other attackers' updates."""
        pass

    def set_global_model_params(self, global_params: torch.Tensor):
        """Interface compatibility; not used."""
        pass

    def set_constraint_params(self, dist_bound: Optional[float] = None,
                             sim_bound_low: Optional[float] = None,
                             sim_bound_up: Optional[float] = None,
                             total_data_size: Optional[float] = None,
                             benign_data_sizes: Optional[dict] = None):
        """Interface compatibility; not used."""
        pass

    def set_lagrangian_params(self, **kwargs):
        """Interface compatibility; not used."""
        pass