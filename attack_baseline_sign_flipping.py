# attack_baseline_sign_flipping.py
# Sign-Flipping Attack Implementation - Model Poisoning Baseline (ICML '18)
#
# This module implements the Sign-flipping attack as defined in:
#   "Asynchronous Byzantine Machine Learning (the case of SGD)", Damaskinos et al., ICML 2018.
#   Malicious update: g^byz = -scale * g_own, where g_own is the gradient/update computed by
#   this worker on its own data (same training as benign). Paper uses scale=10.
# It is completely isolated from the GRMP and ALIE implementations.

import torch
from typing import List, Optional

from client import BenignClient


class SignFlippingAttackerClient(BenignClient):
    """
    Sign-Flipping Attack - ICML '18 Byzantine Baseline
    
    Malicious update = -scale * own_update, where own_update is the update this client
    would have sent if honest (computed by local training on its assigned data).
    Paper: "We test Kardam against a baseline Byzantine behavior (3 out of 10 workers
    send g^byz_p = −10 gp)" — Damaskinos et al., ICML 2018.
    
    This attack:
    - Uses real local training (inherited from BenignClient) to get g_own
    - Flips and scales: sends -sign_flip_scale * g_own (default scale=10 per paper)
    """

    def __init__(self, client_id: int, model, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 data_loader,  # Required: same as benign, for computing g_own (ICML '18)
                 sign_flip_scale: float = 10.0,
                 attack_start_round: Optional[int] = None,
                 claimed_data_size: float = 1.0,
                 grad_clip_norm: float = 1.0):
        """
        Initialize Sign-Flipping attacker client (ICML '18: g^byz = -scale * g_own).
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance (kept for API compatibility with main)
            data_indices: List of data indices assigned (used for aggregation weight)
            lr: Learning rate for local training (used to compute g_own)
            local_epochs: Number of local training epochs per round
            alpha: Proximal coefficient (FedProx, used in local training)
            data_loader: DataLoader for this client's assigned data (required for g_own)
            sign_flip_scale: Scale for flip; malicious = -scale * g_own. Paper uses 10.
            attack_start_round: Round to start attack (None = start immediately)
            claimed_data_size: Data size to claim for weighted aggregation
            grad_clip_norm: Gradient clipping norm for local training
        """
        super().__init__(
            client_id=client_id,
            model=model,
            data_loader=data_loader,
            lr=lr,
            local_epochs=local_epochs,
            alpha=alpha,
            data_indices=data_indices,
            grad_clip_norm=grad_clip_norm
        )
        self.is_attacker = True
        self.attack_method = "SignFlipping"
        self.sign_flip_scale = sign_flip_scale
        self.attack_start_round = attack_start_round
        self.claimed_data_size = claimed_data_size

    def prepare_for_round(self, round_num: int):
        """Prepare for a new round."""
        self.set_round(round_num)

    # local_train: inherited from BenignClient — computes real g_own on own data

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        ICML '18: malicious = -scale * g_own. Here poisoned_update is g_own from local_train.
        
        Args:
            poisoned_update: The own update from local_train (g_own).
        
        Returns:
            Malicious update: -sign_flip_scale * poisoned_update (or honest if before attack_start_round).
        """
        if self.attack_start_round is not None:
            if self.current_round < self.attack_start_round:
                return poisoned_update

        malicious = -self.sign_flip_scale * poisoned_update
        norm_own = float(torch.norm(poisoned_update).item())
        norm_mal = float(torch.norm(malicious).item())
        print(f"    [SignFlipping Attacker {self.client_id}] ICML '18: g^byz = -{self.sign_flip_scale} * g_own; "
              f"||g_own||={norm_own:.4f}, ||g^byz||={norm_mal:.4f}")
        return malicious

    def receive_benign_updates(self, updates: List[torch.Tensor],
                               client_ids: Optional[List[int]] = None):
        """Interface compatibility; attack logic does not use benign updates (ICML '18 uses only g_own)."""
        pass

    def receive_attacker_updates(self, updates: List[torch.Tensor],
                                  client_ids: List[int],
                                  data_sizes: Optional[dict] = None):
        """Interface compatibility; not used."""
        pass

    def set_global_model_params(self, global_params: torch.Tensor):
        """Interface compatibility; model is synced via broadcast_model."""
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