# client.py
# Provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from models import VGAE
from torch.nn.utils import stateless
# ============================================================================
# CRITICAL: Functional call wrapper for LoRA gradient preservation
# ============================================================================
# Purpose: Use torch.func.functional_call to preserve gradients when injecting
#          LoRA parameters into the model forward pass. This ensures the
#          computational graph remains intact from proxy_param to loss.
#
# API Note: torch.func.functional_call (PyTorch 2.0+) takes (params, buffers) tuple
#           stateless.functional_call (PyTorch < 2.0) only takes params dict
# ============================================================================
try:
    from torch.func import functional_call as _torch_func_call
    def functional_call(model, params_buffers, args=(), kwargs=None):
        """
        Wrapper for torch.func.functional_call (PyTorch 2.0+).
        
        Args:
            model: PyTorch model
            params_buffers: Tuple of (params_dict, buffers_dict)
            args: Positional arguments for forward pass
            kwargs: Keyword arguments for forward pass
            
        Returns:
            Model output with preserved gradients
        """
        params, buffers = params_buffers
        return _torch_func_call(model, (params, buffers), args=args, kwargs=kwargs or {})
except ImportError:
    # Fallback for older PyTorch versions (< 2.0)
    from torch.nn.utils.stateless import functional_call as _stateless_call
    def functional_call(model, params_buffers, args=(), kwargs=None):
        """
        Fallback wrapper for torch.nn.utils.stateless.functional_call (PyTorch < 2.0).
        
        Note: stateless.functional_call doesn't support buffers parameter, so we only
              pass params. This is acceptable because buffers are typically constant
              and don't need to be injected for gradient preservation.
        """
        params, buffers = params_buffers
        return _stateless_call(model, params, args=args, kwargs=kwargs or {})

# Client class for federated learning
class Client:

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_loader: DataLoader for local training data
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient μ (FedProx standard: min_w F_k(w) + (μ/2) * ||w - w_t||²)
                   Note: α corresponds to μ in FedProx paper (Li et al., 2020)
        
        Note: All parameters must be explicitly provided. Default values are removed to prevent
        inconsistencies with config settings. See main.py for proper usage.
        
        Memory optimization: Model is kept in CPU by default to save GPU memory.
        It will be moved to GPU only during training (for benign clients) or proxy loss calculation (for attackers).
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        # Keep model in CPU initially to save GPU memory
        # Will be moved to GPU only when needed (training or proxy loss calculation)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha  # Regularization coefficient α ∈ [0,1] from paper formula (1)
        # CRITICAL: Use explicit cuda:0 instead of 'cuda' to ensure device consistency
        # This prevents issues where 'cuda' and 'cuda:0' are treated as different devices
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        # Do NOT move model to GPU here - will be moved on-demand
        self.optimizer = None  # Will be created when needed
        self.current_round = 0
        self.is_attacker = False
        self._model_on_gpu = False  # Track if model is currently on GPU

    def reset_optimizer(self):
        """Reset the optimizer. Only valid when model is on GPU."""
        if self._model_on_gpu:
            # Only optimize trainable parameters (important for LoRA)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.lr)
        else:
            self.optimizer = None

    def set_round(self, round_num: int):
        """Set the current training round."""
        self.current_round = round_num

    def get_model_update(self, initial_params: torch.Tensor) -> torch.Tensor:
        """
        Calculate the model update (Current - Initial).
        
        Args:
            initial_params: Initial model parameters (flattened)
            
        Returns:
            Model update tensor (flattened, on CPU)
        
        Note: Works on both CPU and GPU models. Returns CPU tensor to save GPU memory.
        """
        current_params = self.model.get_flat_params()
        # Ensure both tensors are on the same device before subtraction
        # initial_params is on CPU, so move current_params to CPU
        if current_params.device.type == 'cuda':
            current_params = current_params.cpu()
        # Ensure initial_params is also on CPU (should already be, but double-check)
        if initial_params.device.type == 'cuda':
            initial_params = initial_params.cpu()
        update = current_params - initial_params
        return update

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


# BenignClient class for benign clients
class BenignClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha,
                 data_indices=None, grad_clip_norm=1.0):
        super().__init__(client_id, model, data_loader, lr, local_epochs, alpha)
        # Track assigned data indices for proper aggregation weighting
        self.data_indices = data_indices or []
        self.grad_clip_norm = grad_clip_norm

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Perform local training with FedProx proximal regularization.
        
        Standard FedProx formula: min_w F_k(w) + (μ/2) * ||w - w_t||²
        where μ is the proximal regularization coefficient (self.alpha).
        """
        if epochs is None:
            epochs = self.local_epochs
            
        # Move model to GPU for training
        if not self._model_on_gpu:
            self.model.to(self.device)
            self._model_on_gpu = True
            # Create optimizer when model is on GPU
            # Only optimize trainable parameters (important for LoRA)
            if self.optimizer is None:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = optim.Adam(trainable_params, lr=self.lr)
            
        self.model.train()
        # Get initial params and move to CPU to save GPU memory
        initial_params = self.model.get_flat_params().clone().cpu()
        
        # Proximal regularization coefficient (FedProx standard: μ)
        # Standard formula: min_w F_k(w) + (μ/2) * ||w - w_t||²
        # Note: α (self.alpha) corresponds to μ in FedProx paper
        mu = self.alpha

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
                
                # Add proximal regularization term (FedProx standard formula: (μ/2) * ||w - w_t||²)
                # Standard FedProx: min_w F_k(w) + (μ/2) * ||w - w_t||²
                # This ensures gradient is μ * (w - w_t) without extra 2x factor
                # Move initial_params to GPU temporarily for computation
                # CRITICAL: requires_grad=True to preserve gradients for backward pass
                current_params = self.model.get_flat_params(requires_grad=True)
                initial_params_gpu = initial_params.to(self.device)
                proximal_term = (mu / 2.0) * torch.norm(current_params - initial_params_gpu) ** 2
                initial_params_gpu = None  # Release GPU reference
                
                loss = ce_loss + proximal_term
                
                if not torch.isfinite(loss).item():
                    # Skip batch on nan/inf to avoid corrupting model (e.g. Pythia-160m can be unstable)
                    import warnings
                    warnings.warn(
                        f"[Client {self.client_id}] Skipping batch: loss={loss.item()} (non-finite). "
                        "Consider lowering client_lr or grad_clip_norm for decoder models (e.g. Pythia-160m)."
                    )
                    pbar.set_postfix({'loss': 'nan(skip)'})
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate update (will be on CPU)
        update = self.get_model_update(initial_params)
        
        # Move model back to CPU to free GPU memory
        self.model.cpu()
        self._model_on_gpu = False
        # Delete optimizer to free its GPU memory (Adam states)
        del self.optimizer
        self.optimizer = None
        torch.cuda.empty_cache()  # Clear CUDA cache
        
        return update

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        # Benign clients do not use this method
        pass


# AttackerClient class for clients that perform attacks
class AttackerClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_manager,
                 data_indices, lr, local_epochs, alpha,
                 dim_reduction_size=10000,
                 vgae_epochs=20, vgae_lr=0.01, graph_threshold=0.5,
                 proxy_step=0.1,
                 claimed_data_size=1.0,
                 proxy_sample_size=512,
                 proxy_max_batches_opt=2,
                 proxy_max_batches_eval=4,
                 vgae_hidden_dim=32,
                 vgae_latent_dim=16,
                 vgae_dropout=0.0,
                 vgae_kl_weight=0.1,
                 proxy_steps=20,
                 grad_clip_norm=1.0,
                 proxy_grad_clip_norm=None,
                 early_stop_constraint_stability_steps=3,
                 use_proxy_data=True):
        """
        Initialize an attacker client with VGAE-based camouflage capabilities.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance for managing attacker data
            data_indices: List of data indices assigned to this client
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient μ (FedProx standard: (μ/2) * ||w - w_t||²)
                   Note: α corresponds to μ in FedProx paper (Li et al., 2020)
            dim_reduction_size: Dimensionality for feature reduction (default: 10000)
            vgae_epochs: Number of epochs for VGAE training (default: 20)
            vgae_lr: Learning rate for VGAE optimizer (default: 0.01)
            graph_threshold: Threshold for graph adjacency matrix binarization (default: 0.5)
            proxy_step: Step size for gradient-free ascent toward global-loss proxy (default: 0.1)
            claimed_data_size: Reported data size D'_j(t) for weighted aggregation (default: 1.0)
            proxy_sample_size: Number of samples in proxy dataset for F(w'_g) estimation (default: 512)
            proxy_max_batches_opt: Max batches for proxy loss in optimization loop (default: 2)
            proxy_max_batches_eval: Max batches for proxy loss in final evaluation (default: 4)
            vgae_hidden_dim: VGAE hidden layer dimension (default: 32, per paper)
            vgae_latent_dim: VGAE latent space dimension (default: 16, per paper)
            vgae_dropout: VGAE dropout rate (default: 0.0)
            vgae_kl_weight: Weight for KL divergence term in VGAE loss (default: 0.1)
            proxy_steps: Number of optimization steps for attack objective (default: 20)
            grad_clip_norm: Kept for compatibility; proxy step uses proxy_grad_clip_norm if set.
            proxy_grad_clip_norm: Gradient clipping for AugMP proxy parameter update only (default: None = use grad_clip_norm). Separate from benign client training.
            use_proxy_data: If True, use proxy set to estimate F(w'_g); if False, no data access (constraint-only optimization) (default: True)
        
        Note: lr, local_epochs, and alpha must be explicitly provided to ensure consistency
        with config settings. Other parameters have defaults but should be set via config in main.py.
        """
        self.data_manager = data_manager
        self.data_indices = data_indices
        
        # Store parameters first (before using them)
        self.dim_reduction_size = dim_reduction_size
        self.vgae_epochs = vgae_epochs
        self.vgae_lr = vgae_lr
        self.graph_threshold = graph_threshold
        self.proxy_step = proxy_step
        self.claimed_data_size = claimed_data_size  # For weighted aggregation (paper: D'(t))
        self.proxy_sample_size = proxy_sample_size
        self.proxy_max_batches_opt = proxy_max_batches_opt
        self.proxy_max_batches_eval = proxy_max_batches_eval
        self.vgae_hidden_dim = vgae_hidden_dim
        self.vgae_latent_dim = vgae_latent_dim
        self.vgae_dropout = vgae_dropout
        self.vgae_kl_weight = vgae_kl_weight
        self.proxy_steps = proxy_steps
        self.grad_clip_norm = grad_clip_norm
        self.proxy_grad_clip_norm = proxy_grad_clip_norm if proxy_grad_clip_norm is not None else grad_clip_norm
        self.early_stop_constraint_stability_steps = early_stop_constraint_stability_steps
        self.use_proxy_data = use_proxy_data

        dummy_loader = data_manager.get_empty_loader()
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.benign_update_client_ids = []  # Track client_id for each benign update to enable weighted average calculation
        self.feature_indices = None
        
        # Other attackers' updates (for coordinated optimization)
        self.other_attacker_updates = []
        self.other_attacker_client_ids = []
        self.other_attacker_data_sizes = {}  # {client_id: claimed_data_size}
        
        if use_proxy_data:
            self.proxy_loader = data_manager.get_proxy_eval_loader(sample_size=self.proxy_sample_size)
        else:
            self.proxy_loader = None  # No data access; optimization will use constraint terms only
        
        # Formula 4 constraints parameters
        self.dist_bound = None  # Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ dist_bound
        self.global_model_params = None  # Store global model params for constraint (4b) (will be on GPU when needed)
        # Paper Formula (2): w'_g(t) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) w_i(t) + (D'_j(t)/D(t)) w'_j(t)
        self.total_data_size = None  # D(t): Total data size for aggregation weight calculation
        self.benign_data_sizes = {}  # {client_id: D_i(t)}: Data sizes for each benign client
        
        # Manual cosine similarity bounds (None = use benign min/max)
        self.sim_bound_low = None  # If set, use as lower bound; else benign min
        self.sim_bound_up = None   # If set, use as upper bound; else benign mean
        
        # Lagrangian dual variables (λ(t) from paper)
        # Initialized in set_lagrangian_params
        self.lambda_dist = None  # λ_dist(t): Lagrangian multiplier for distance constraint (4b)
        self.use_lagrangian_dual = False  # Whether to use Lagrangian Dual mechanism
        self.lambda_dist_lr = 0.01  # Learning rate for λ_dist(t) update
        # Save initial values for reset in prepare_for_round
        self.lambda_dist_init = None  # Save initial λ_dist value for reset in prepare_for_round
        
        # Cosine similarity constraint parameters (TWO-SIDED with TWO multipliers)
        self.use_cosine_similarity_constraint = False  # Whether to use cosine similarity constraints
        self.lambda_sim_low = None  # λ_sim_low(t): Lagrangian multiplier for lower bound constraint
        self.lambda_sim_up = None  # λ_sim_up(t): Lagrangian multiplier for upper bound constraint
        self.lambda_sim_low_lr = 0.01  # Learning rate for λ_sim_low(t) update
        self.lambda_sim_up_lr = 0.01  # Learning rate for λ_sim_up(t) update
        self.lambda_sim_low_init = None  # Save initial λ_sim_low value for reset
        self.lambda_sim_up_init = None  # Save initial λ_sim_up value for reset

        # ============================================================
        # Augmented Lagrangian (ALM) penalty parameters (ρ)
        # Standard ALM uses: L_aug = f(x) + Σ_i [ λ_i g_i(x) + (ρ_i/2) g_i(x)^2 ]
        # where g_i(x) ≤ 0 are inequality constraints.
        # ============================================================
        self.use_augmented_lagrangian = False  # Whether to use Augmented Lagrangian Method (ALM)
        self.lambda_update_mode = "classic"  # "classic": λ += lr * g ; "alm": λ += ρ * g

        # ρ variables (kept on same device as optimization)
        self.rho_dist = None
        self.rho_sim_low = None
        self.rho_sim_up = None

        # Save initial values for reset in prepare_for_round
        self.rho_dist_init = None
        self.rho_sim_low_init = None
        self.rho_sim_up_init = None

        # Adaptive ρ update (monotone increase) parameters
        self.rho_adaptive = True
        self.rho_theta = 0.5  # If violation does not decrease enough: σ_k > theta * σ_{k-1} => increase ρ
        self.rho_increase_factor = 2.0
        self.rho_min = 1e-3
        self.rho_max = 1e3

        # Track previous constraint violations for adaptive ρ update (per-round)
        self._prev_violation_dist = None
        self._prev_violation_sim_low = None
        self._prev_violation_sim_up = None
        
        
        # Get model parameter count (works on CPU model)
        self._flat_numel = int(self.model.get_flat_params().numel())  # Convert to Python int
        
        # ===== CRITICAL: LoRA functional_call cache for gradient preservation =====
        # These will be initialized in _init_functional_param_cache() when needed
        self.lora_param_names: List[str] = []  # Ordered list of LoRA parameter names
        self.lora_param_shapes: Dict[str, torch.Size] = {}  # Shape for each LoRA param
        self.lora_param_numels: Dict[str, int] = {}  # Numel for each LoRA param
        self.lora_param_slices: Dict[str, slice] = {}  # Slice in flat tensor for each LoRA param
        self.base_params: Dict[str, torch.Tensor] = {}  # Frozen base parameters (detached)
        self.base_buffers: Dict[str, torch.Tensor] = {}  # Buffers (detached)
        self._functional_cache_initialized = False  # Cache initialization flag
        # ============================================================================
        
        # Validate and adjust dim_reduction_size for LoRA mode
        # In LoRA mode, if dim_reduction_size > actual LoRA params, use all LoRA params
        # Rationale: When LoRA params are already small, using all of them is more reasonable
        # than further reducing, as it preserves information and the computation is still feasible.
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            actual_lora_params = self._flat_numel
            if dim_reduction_size > actual_lora_params:
                # Auto-adjust: use all LoRA params (no further reduction needed)
                # When LoRA params are already small, using all of them is reasonable
                # and preserves more information for VGAE training
                print(f"    [Attacker {self.client_id}] Info: dim_reduction_size ({dim_reduction_size}) > LoRA params ({actual_lora_params})")
                print(f"    [Attacker {self.client_id}] Auto-adjusting dim_reduction_size to {actual_lora_params} (using all LoRA params)")
                self.dim_reduction_size = actual_lora_params
            elif dim_reduction_size == actual_lora_params:
                # Use all parameters (no reduction), which is fine
                pass
            else:
                # dim_reduction_size < actual_lora_params, which is the normal case (with reduction)
                pass

    def prepare_for_round(self, round_num: int):
        """
        Prepare for a new training round.
        
        Modification 1: Reset λ and ρ to initial values at the start of each round
        to prevent numerical instability from cross-round accumulation.
        """
        self.set_round(round_num)
        # Data-agnostic attacker keeps an empty loader
        self.data_loader = self.data_manager.get_empty_loader()

        # ===== CRITICAL: Reset functional cache for new round =====
        # Model structure may change between rounds, so cache must be reset
        self._functional_cache_initialized = False
        self.lora_param_names = []
        self.lora_param_shapes = {}
        self.lora_param_numels = {}
        self.lora_param_slices = {}
        self.base_params = {}
        self.base_buffers = {}
        # ============================================================

        # Note: d_T is used only as fallback when distance prediction is disabled or no history

        # Modification 1: Reset Lagrangian multipliers at the start of each round
        # Reason: Prevent λ from accumulating across rounds, which causes numerical instability and optimization imbalance
        
        # Reset distance constraint multiplier
        if self.use_lagrangian_dual and self.lambda_dist_init is not None:
            self.lambda_dist = torch.tensor(self.lambda_dist_init, requires_grad=False)
        
        # Reset cosine similarity constraint multipliers (TWO multipliers for two-sided constraint)
        if self.use_cosine_similarity_constraint and self.lambda_sim_low_init is not None:
            self.lambda_sim_low = torch.tensor(self.lambda_sim_low_init, requires_grad=False)
        if self.use_cosine_similarity_constraint and self.lambda_sim_up_init is not None:
            self.lambda_sim_up = torch.tensor(self.lambda_sim_up_init, requires_grad=False)

        # Reset Augmented Lagrangian penalty parameters (ρ) at the start of each round
        if self.use_lagrangian_dual and self.use_augmented_lagrangian:
            if self.rho_dist_init is not None:
                self.rho_dist = torch.tensor(self.rho_dist_init, requires_grad=False)
            if self.use_cosine_similarity_constraint:
                if self.rho_sim_low_init is not None:
                    self.rho_sim_low = torch.tensor(self.rho_sim_low_init, requires_grad=False)
                if self.rho_sim_up_init is not None:
                    self.rho_sim_up = torch.tensor(self.rho_sim_up_init, requires_grad=False)

            # Reset per-round violation history (avoid cross-round coupling)
            self._prev_violation_dist = None
            self._prev_violation_sim_low = None
            self._prev_violation_sim_up = None

    def receive_benign_updates(self, updates: List[torch.Tensor], client_ids: Optional[List[int]] = None):
        """
        Receive updates from benign clients.
        
        Args:
            updates: List of benign client updates
            client_ids: Optional list of client IDs corresponding to each update.
                       If None, indices will be used as client IDs (fallback for backward compatibility)
        """
        # Store detached copies on CPU to save GPU memory
        # Updates will be moved to GPU only when needed for VGAE processing
        self.benign_updates = [u.detach().clone().cpu() for u in updates]
        # Store corresponding client IDs for weighted average calculation
        if client_ids is not None:
            self.benign_update_client_ids = client_ids.copy()
        else:
            # Fallback: use indices as client IDs (for backward compatibility)
            # Note: This may not be accurate, but allows code to work without server changes
            self.benign_update_client_ids = list(range(len(updates)))
    
    def receive_attacker_updates(self, updates: List[torch.Tensor], client_ids: List[int], data_sizes: Dict[int, float] = None):
        """
        Receive updates from other attackers that have already completed optimization.
        These will be used in distance calculation to match Phase 4's reference point.
        
        Args:
            updates: List of attacker update tensors (already optimized)
            client_ids: List of attacker client IDs
            data_sizes: Dictionary mapping client_id to claimed_data_size (optional)
        """
        # Store detached copies on CPU to save GPU memory
        self.other_attacker_updates = [u.detach().clone().cpu() for u in updates]
        self.other_attacker_client_ids = client_ids.copy() if client_ids else []
        
        # Store data sizes for weighted aggregation
        if data_sizes is not None:
            self.other_attacker_data_sizes = data_sizes.copy()
        else:
            # Fallback: use current attacker's claimed size as estimate
            self.other_attacker_data_sizes = {cid: float(self.claimed_data_size) for cid in client_ids}

    def _select_benign_subset(self) -> List[torch.Tensor]:
        """
        Select a subset of benign updates (β selection) using 0-1 Knapsack optimization.
        
        Paper formulation (Equation 9):
        β'_{i,j}(t)^* = argmin_{β'_{i,j}(t)} Σ_{i=1}^I β'_{i,j}(t) d(w_i(t), w̄_i(t))
        s.t. Σ_{i=1}^I β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        β'_{i,j}(t) ∈ {0,1}
        
        This is a 0-1 Knapsack problem: minimize sum of selected distances
        subject to sum ≤ capacity (Γ).
        
        Note: Since we want to minimize the sum and the constraint is also on the sum,
        the optimal solution is to select as many items as possible while staying within capacity.
        We use a greedy approach to find an approximate optimal selection.
        
        Returns:
            List of selected benign updates (on CPU to save GPU memory)
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from weighted mean for all benign updates
        # Paper definition: w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) (weighted mean, not simple mean)
        # Move to GPU only for computation, then back to CPU
        benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_updates_gpu])
        
        # Compute weighted mean: w̄_i(t) = Σ (D_i/D) w_i(t)
        if self.total_data_size is not None and len(self.benign_data_sizes) > 0 and len(self.benign_update_client_ids) > 0:
            D_total = float(self.total_data_size)
            benign_mean = torch.zeros_like(benign_stack[0])
            for idx, benign_update in enumerate(self.benign_updates):
                if idx < len(self.benign_update_client_ids):
                    client_id = self.benign_update_client_ids[idx]
                    D_i = self.benign_data_sizes.get(client_id, 1.0)
                    weight = D_i / D_total
                else:
                    # Fallback: use equal weight if client_id not available
                    weight = 1.0 / len(self.benign_updates)
                benign_mean = benign_mean + weight * benign_update.to(self.device)
        else:
            # Fallback: use simple mean if data sizes not available
            benign_mean = benign_stack.mean(dim=0)
        
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        # Clean up GPU references immediately
        del benign_updates_gpu, benign_stack, benign_mean
        torch.cuda.empty_cache()
        
        # Always return all benign updates
        return self.benign_updates
    
    def _get_selected_benign_indices(self) -> List[int]:
        """
        Get indices of selected benign updates (β selection).
        This is a helper method to avoid tensor comparison issues.
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from weighted mean for all benign updates
        # Paper definition: w̄_i(t) = Σ_{i=1}^I (D_i(t)/D(t)) w_i(t) (weighted mean, not simple mean)
        # Move to GPU only for computation, then back to CPU
        benign_updates_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_updates_gpu])
        
        # Compute weighted mean: w̄_i(t) = Σ (D_i/D) w_i(t)
        if self.total_data_size is not None and len(self.benign_data_sizes) > 0 and len(self.benign_update_client_ids) > 0:
            D_total = float(self.total_data_size)
            benign_mean = torch.zeros_like(benign_stack[0])
            for idx, benign_update in enumerate(self.benign_updates):
                if idx < len(self.benign_update_client_ids):
                    client_id = self.benign_update_client_ids[idx]
                    D_i = self.benign_data_sizes.get(client_id, 1.0)
                    weight = D_i / D_total
                else:
                    # Fallback: use equal weight if client_id not available
                    weight = 1.0 / len(self.benign_updates)
                benign_mean = benign_mean + weight * benign_update.to(self.device)
        else:
            # Fallback: use simple mean if data sizes not available
            benign_mean = benign_stack.mean(dim=0)
        
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        # Clean up GPU references immediately
        del benign_updates_gpu, benign_stack, benign_mean
        torch.cuda.empty_cache()
        
        # Always return all indices
        return list(range(len(self.benign_updates)))

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Attacker does not perform local training (data-agnostic attack).
        
        Attackers are not assigned local data, so they return zero update.
        The actual attack is generated in camouflage_update using VGAE+GSP.
        
        Returns:
            Zero update tensor on CPU (to save GPU memory)
        """
        # Attackers don't have local data, return zero update
        # Model is on CPU, so initial_params is on CPU
        initial_params = self.model.get_flat_params().clone()
        return torch.zeros_like(initial_params)  # Already on CPU

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Selects indices based on update magnitude (importance) to prioritize parameters
        that change most in normal training, simulating realistic training patterns.
        
        Selection strategy:
        1. Calculate update magnitudes (absolute mean across all updates)
        2. Select top-K important parameters (2x dim_reduction_size for diversity pool)
        3. Randomly select from important pool using client_id for diversity
        
        This approach:
        - Prioritizes important parameters (classifier, attention layers)
        - Maintains diversity among different attackers
        - Simulates normal training patterns (updates important params more)
        - Aligns with FedLLM standard practices (FedPipe, FedLEASE)
        
        Args:
            updates: List of update tensors to reduce
            fix_indices: If True, reuse existing feature_indices; if False, generate new ones
            
        Returns:
            Stacked reduced features tensor of shape (I, M) where I=num_updates, M=dim_reduction_size
        """
        stacked_updates = torch.stack(updates)
        # Ensure stacked_updates has valid shape
        if len(stacked_updates.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] stacked_updates must be 2D, got shape {stacked_updates.shape}")
        shape_dim = stacked_updates.shape[1]
        if shape_dim is None:
            raise ValueError(f"[Attacker {self.client_id}] stacked_updates.shape[1] is None")
        try:
            total_dim = int(shape_dim)  # Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert shape[1]={shape_dim} to int: {e}")
        
        # If update dimension is smaller than reduction target, skip reduction
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # Fix feature indices at the start of each attack round to ensure training consistency within the round
        if self.feature_indices is None or not fix_indices:
            # Importance-based selection: prioritize parameters with larger update magnitudes
            # This simulates normal training patterns where important parameters change more
            import hashlib
            
            # Validate inputs
            if self.client_id is None:
                raise ValueError(f"client_id is None for attacker")
            if total_dim is None or total_dim <= 0:
                raise ValueError(f"total_dim is None or invalid: {total_dim}")
            
            # Step 1: Calculate update magnitudes (importance scores)
            # Use absolute mean across all benign updates to identify important parameters
            # Parameters with larger magnitudes are more important (change more in normal training)
            update_magnitudes = torch.abs(stacked_updates).mean(dim=0)  # (total_dim,)
            
            # Step 2: Select top-K important parameters (2x for diversity pool)
            # Select 2x dim_reduction_size to create a pool, then randomly select from pool
            # This balances importance with diversity among different attackers
            top_k = min(self.dim_reduction_size * 2, total_dim)
            _, top_indices_tensor = torch.topk(update_magnitudes, k=top_k)
            top_indices = top_indices_tensor.cpu().numpy()  # Convert to numpy for random selection
            
            # Step 3: Within top-K, use client_id for diversity (different attackers select different params)
            # This ensures different attackers choose different parameters from the important pool
            seed_str = f"{self.client_id}_{top_k}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
            np_rng = np.random.RandomState(seed)
            # Ensure we don't select more than available (safety check)
            num_to_select = min(self.dim_reduction_size, len(top_indices))
            permuted = np_rng.permutation(len(top_indices))[:num_to_select]
            selected_indices = top_indices[permuted]
            
            # Step 4: Create feature_indices tensor on the same device as stacked_updates
            # Ensure device consistency for index_select operation
            target_device = stacked_updates.device
            self.feature_indices = torch.tensor(selected_indices, dtype=torch.long, device=target_device)
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _flat_to_param_dict(self, flat_params: torch.Tensor, skip_dim_check: bool = False) -> Dict[str, torch.Tensor]:
        """
        Convert flat tensor to param dict for stateless.functional_call.
        
        In LoRA mode, only sets LoRA parameters (trainable parameters).
        In full fine-tuning mode, sets all parameters.
        
        Important: Handles PEFT model parameter name compatibility.
        PEFT models have nested structure (base_model.model.*), and stateless.functional_call
        may need specific parameter name formats.
        
        Args:
            flat_params: Flattened parameter tensor (LoRA params in LoRA mode, all params in full mode)
            skip_dim_check: If True, skip dimension check (for performance in loops)
        
        Returns:
            Dictionary mapping parameter names to tensors, compatible with stateless.functional_call
        """
        param_dict = {}
        offset = 0
        flat_params = flat_params.view(-1)  # Ensure 1D (O(1), just view change)
        total_numel = int(flat_params.numel())  # Convert to Python int
        
        # Check if model is in LoRA mode
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        
        # Build a mapping from parameter objects to their names
        # This is more efficient than searching each time
        param_to_name = {}
        for name, param in self.model.named_parameters():
            # In LoRA mode, only track trainable parameters
            if use_lora:
                if param.requires_grad:
                    param_to_name[param] = name
            else:
                # Full fine-tuning: track all parameters
                param_to_name[param] = name
        
        # Iterate through parameters in the same order as get_flat_params
        for param in self.model.parameters():
            # In LoRA mode, skip non-trainable parameters
            if use_lora and not param.requires_grad:
                continue
            
            # Get parameter name from pre-built mapping
            param_name = param_to_name.get(param)
            if param_name is None:
                # Parameter not in mapping (shouldn't happen, but handle gracefully)
                continue
            
            numel = int(param.numel())  # Convert to Python int
            if not skip_dim_check and offset + numel > total_numel:
                # Dimension mismatch: return empty dict to avoid errors
                print(f"    [Attacker {self.client_id}] Param dict dimension mismatch: offset {offset} + numel {numel} > total {total_numel}")
                return {}
            
            # For PEFT models, stateless.functional_call expects parameter names
            # that match the actual model structure. The names from named_parameters()
            # should already be correct, but we verify compatibility.
            param_value = flat_params[offset:offset + numel].view_as(param)
            
            # Ensure param_value is on the same device as param
            # This is important when model is on GPU but flat_params might be on different device
            if param_value.device != param.device:
                param_value = param_value.to(param.device)
            
            # Handle PEFT model parameter names (base_model.model.* format)
            # stateless.functional_call should work with the names as-is from named_parameters()
            # But if we're working with a PEFT-wrapped model, ensure the name is correct
            if use_lora and hasattr(self.model, 'model') and hasattr(self.model.model, 'base_model'):
                # This is a PEFT model - parameter names should already include base_model.model prefix
                # from named_parameters(), so use as-is
                param_dict[param_name] = param_value
            else:
                # Standard model or direct PEFT model access
                param_dict[param_name] = param_value
            
            offset += numel
        
        # Verify we used all parameters
        if not skip_dim_check and offset != total_numel:
            print(f"    [Attacker {self.client_id}] Param dict size mismatch: used {offset} params, provided {total_numel}")
            # This could indicate a serious problem - log warning but continue
        
        return param_dict

    def _device_matches(self, device1, device2):
        """
        Check if two devices are the same, handling 'cuda' vs 'cuda:0' equivalence.
        
        Args:
            device1: First device
            device2: Second device
        
        Returns:
            True if devices are the same, False otherwise
        """
        # Convert to string and normalize
        d1_str = str(device1)
        d2_str = str(device2)
        
        # Normalize 'cuda' to 'cuda:0'
        if d1_str == 'cuda':
            d1_str = 'cuda:0'
        if d2_str == 'cuda':
            d2_str = 'cuda:0'
        
        return d1_str == d2_str

    def _ensure_model_on_device(self, module, device):
        """
        Recursively ensure ALL parameters and buffers of a module are on the specified device.
        This is critical for PEFT models with nested structures.
        
        Args:
            module: The module to move
            device: Target device (will be normalized to 'cuda:0' if it's 'cuda')
        """
        # Normalize device: always use 'cuda:0' instead of 'cuda' for consistency
        device_str = str(device)
        if device_str == 'cuda':
            target_device = torch.device('cuda:0')
        elif device_str.startswith('cuda'):
            target_device = torch.device(device_str if ':' in device_str else 'cuda:0')
        else:
            target_device = device
        
        # Use named_parameters to get all parameters including nested ones
        for name, param in module.named_parameters(recurse=False):
            if not self._device_matches(param.device, target_device):
                # Force move by creating new tensor on target device
                with torch.no_grad():
                    param.data = param.data.to(target_device, non_blocking=True)
        
        for name, buffer in module.named_buffers(recurse=False):
            if not self._device_matches(buffer.device, target_device):
                # Force move buffer
                buffer.data = buffer.data.to(target_device, non_blocking=True)
        
        # Recursively process all child modules
        for child in module.children():
            self._ensure_model_on_device(child, target_device)

    def _init_functional_param_cache(self, target_device: torch.device):
        """
        Initialize cache for functional_call with full parameters (base + LoRA).
        
        CRITICAL: This must be called before using functional_call in LoRA mode.
        Caches LoRA parameter metadata and base parameters/buffers for gradient-preserving forward.
        
        What this function does:
        1. Collects LoRA parameters (trainable) in the same order as get_flat_params()
        2. Caches base parameters (frozen, detached) to avoid repeated lookups
        3. Caches buffers (detached) for functional_call
        4. Verifies dimension consistency (sum(LoRA numel) == _flat_numel)
        5. Verifies parameter name completeness (base + LoRA = all params)
        
        Args:
            target_device: Device to cache parameters on (typically GPU)
        
        Raises:
            AssertionError: If dimension consistency checks fail
            RuntimeError: If parameter name lookup fails
        """
        if self._functional_cache_initialized:
            return  # Already initialized
        
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if not use_lora:
            # Full fine-tuning mode doesn't need special caching
            self._functional_cache_initialized = True
            return
        
        # Step 1: Build LoRA parameter metadata (trainable parameters only)
        # ============================================================
        # CRITICAL: Must match get_flat_params() order EXACTLY
        # ============================================================
        # get_flat_params() in models.py uses: 
        #   for param in self.model.parameters():
        #       if param.requires_grad:
        #           lora_params.append(param.data.view(-1))
        #
        # Problem: parameters() and named_parameters() order may differ!
        # Solution: Build a mapping from param object to name, then iterate in parameters() order
        # ============================================================
        self.lora_param_names = []
        self.lora_param_shapes = {}
        self.lora_param_numels = {}
        self.lora_param_slices = {}
        offset = 0
        
        # CRITICAL: Build param -> name mapping first
        # This allows us to iterate in parameters() order (matching get_flat_params())
        # while still getting parameter names (needed for functional_call)
        param_to_name = {param: name for name, param in self.model.named_parameters()}
        
        # CRITICAL: Iterate in parameters() order (SAME as get_flat_params() in models.py)
        # This ensures exact order match, preventing parameter misalignment
        for param in self.model.parameters():
            if param.requires_grad:
                # Get parameter name from mapping
                name = param_to_name[param]
                numel = int(param.numel())
                self.lora_param_names.append(name)
                self.lora_param_shapes[name] = param.shape
                self.lora_param_numels[name] = numel
                self.lora_param_slices[name] = slice(offset, offset + numel)
                offset += numel
        
        # Step 2: Build base parameters dict (frozen parameters, detached)
        self.base_params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                # Frozen base parameter - detach and move to target device
                with torch.no_grad():
                    base_param = param.data.clone().detach().to(target_device)
                self.base_params[name] = base_param
        
        # Step 3: Build buffers dict (detached)
        self.base_buffers = {}
        for name, buffer in self.model.named_buffers():
            with torch.no_grad():
                base_buffer = buffer.data.clone().detach().to(target_device)
            self.base_buffers[name] = base_buffer
        
        # Step 4: Consistency assertions (CRITICAL)
        total_lora_numel = sum(self.lora_param_numels.values())
        if total_lora_numel != self._flat_numel:
            # Enhanced error message with diagnostic information
            model_params_info = f"Model has {len(list(self.model.named_parameters()))} total params, " \
                              f"{len([p for p in self.model.parameters() if p.requires_grad])} trainable"
            raise RuntimeError(
                f"[Attacker {self.client_id}] LoRA dimension mismatch:\n"
                f"  - Total LoRA numel (from cache): {total_lora_numel}\n"
                f"  - _flat_numel (from get_flat_params): {self._flat_numel}\n"
                f"  - LoRA param names: {self.lora_param_names}\n"
                f"  - {model_params_info}\n"
                f"This indicates parameter order mismatch between get_flat_params() and _init_functional_param_cache()."
            )
        
        all_param_names = set(dict(self.model.named_parameters()).keys())
        expected_param_names = set(self.base_params.keys()) | set(self.lora_param_names)
        if all_param_names != expected_param_names:
            missing_in_cache = all_param_names - expected_param_names
            extra_in_cache = expected_param_names - all_param_names
            raise RuntimeError(
                f"[Attacker {self.client_id}] Parameter name mismatch:\n"
                f"  - Model params: {len(all_param_names)} params\n"
                f"  - Cache params: {len(expected_param_names)} params\n"
                f"  - Missing in cache: {missing_in_cache}\n"
                f"  - Extra in cache: {extra_in_cache}\n"
                f"  - Base params: {set(self.base_params.keys())}\n"
                f"  - LoRA params: {self.lora_param_names}"
            )
        
        self._functional_cache_initialized = True
        print(f"    [Attacker {self.client_id}] Functional cache initialized: "
              f"{len(self.lora_param_names)} LoRA params, {len(self.base_params)} base params, "
              f"{len(self.base_buffers)} buffers")

    def _flat_to_lora_param_dict(self, flat_lora: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert flat LoRA tensor to parameter dict for functional_call.
        
        CRITICAL: This preserves gradients by using view/reshape operations only.
        No .data operations that would break the computational graph.
        
        How it works:
        - Uses pre-computed slices (from _init_functional_param_cache) to extract
          each parameter from the flat tensor
        - Uses .view() to reshape without copying (preserves gradients)
        - Order must match get_flat_params() to ensure correctness
        
        Args:
            flat_lora: 1D flat LoRA parameter tensor (requires_grad=True, on GPU)
                      Should have shape (self._flat_numel,)
        
        Returns:
            Dictionary mapping LoRA parameter names to shaped tensors (preserves gradients)
            Each tensor maintains requires_grad=True and gradient flow
        
        Raises:
            RuntimeError: If functional cache not initialized
        """
        if not self._functional_cache_initialized:
            raise RuntimeError(f"[Attacker {self.client_id}] Functional cache not initialized. "
                             f"Call _init_functional_param_cache() first.")
        
        flat_lora = flat_lora.view(-1)  # Ensure 1D
        
        out = {}
        for name in self.lora_param_names:
            sl = self.lora_param_slices[name]
            shape = self.lora_param_shapes[name]
            # CRITICAL: Use view/reshape to preserve gradients, no copy_() or .data assignment
            out[name] = flat_lora[sl].view(shape)
        
        return out

    def _proxy_global_loss(self, malicious_update: torch.Tensor, max_batches: int = 1, 
                           skip_dim_check: bool = False, keep_model_on_gpu: bool = False) -> torch.Tensor:
        """
        Differentiable proxy for F(w'_g): cross-entropy on a small clean subset,
        using stateless.functional_call with (w_g + malicious_update).
        
        Args:
            malicious_update: Update vector to evaluate (can be on CPU or GPU)
            max_batches: Maximum number of batches to process
            skip_dim_check: If True, skip dimension check (for performance in loops)
            keep_model_on_gpu: If True, model will NOT be moved back to CPU after computation.
                              This is critical when the returned loss will be used for backward pass.
        
        Note: 
            - If keep_model_on_gpu=False: Model will be temporarily moved to GPU, then moved back to CPU.
            - If keep_model_on_gpu=True: Model stays on GPU (important for backward pass in optimization loops).
        """
        if self.global_model_params is None or self.proxy_loader is None:
            return torch.tensor(0.0, device=self.device)

        # Normalize device: always use 'cuda:0' for consistency
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        
        # Ensure malicious_update is on GPU for computation
        if malicious_update.device.type != 'cuda':
            malicious_update = malicious_update.to(target_device)

        # Ensure shapes match: flatten to 1D and check dimension
        malicious_update = malicious_update.view(-1)  # Flatten to 1D (O(1), just view change)
        if not skip_dim_check and int(malicious_update.numel()) != self._flat_numel:
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            msg = (f"    [Attacker {self.client_id}] Proxy loss dimension mismatch: "
                   f"got {malicious_update.numel()}, expected {self._flat_numel}")
            if use_lora:
                raise RuntimeError(msg)
            print(msg)
            return torch.tensor(0.0, device=self.device)

        # Move model to GPU temporarily for proxy loss calculation
        # Normalize device: always use 'cuda:0' for consistency
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        
        model_was_on_cpu = not self._model_on_gpu
        if model_was_on_cpu:
            # Move entire model to device (including all parameters and buffers)
            # For PEFT models, this should move base model, LoRA parameters, and all buffers
            self.model.to(target_device)
            
            # CRITICAL FIX for PEFT models: Recursively ensure ALL parameters and buffers are on GPU
            # This is necessary because PEFT models have nested structures that .to() might not handle correctly
            self._ensure_model_on_device(self.model, target_device)
            
            # Double-check: Verify ALL parameters and buffers are actually on GPU
            # Use normalized device: always use 'cuda:0' for consistency
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            for name, param in self.model.named_parameters():
                if not self._device_matches(param.device, target_device):
                    print(f"    [Attacker {self.client_id}] ERROR: Parameter {name} on {param.device}, moving to {target_device}")
                    with torch.no_grad():
                        param.data = param.data.to(target_device, non_blocking=True)
            for name, buffer in self.model.named_buffers():
                if not self._device_matches(buffer.device, target_device):
                    print(f"    [Attacker {self.client_id}] ERROR: Buffer {name} on {buffer.device}, moving to {target_device}")
                    buffer.data = buffer.data.to(target_device, non_blocking=True)
            
            self._model_on_gpu = True

        try:
            # CRITICAL: Ensure global_model_params and malicious_update are on the same device
            # Both should be on target_device for proper computation
            if not self._device_matches(self.global_model_params.device, target_device):
                global_params_gpu = self.global_model_params.to(target_device)
            else:
                global_params_gpu = self.global_model_params
            
            candidate_params = global_params_gpu + malicious_update
            
            # CRITICAL: Check LoRA mode BEFORE processing to avoid unnecessary work
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            
            # ===== CRITICAL: Initialize functional cache for LoRA mode =====
            # This must be done before any parameter processing
            if use_lora:
                self._init_functional_param_cache(target_device)
            # ===================================================================
            
            # For full fine-tuning mode, prepare param_dict (LoRA mode doesn't need this)
            param_dict = {}
            if not use_lora:
                # Skip dimension check if already validated (performance optimization)
                param_dict = self._flat_to_param_dict(candidate_params, skip_dim_check=skip_dim_check)

            # CRITICAL FIX: Ensure all parameters in param_dict are on the correct device
            # Normalize device: always use 'cuda:0' for consistency  
            # Note: target_device already defined earlier, but redefining here for clarity
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            for name, value in param_dict.items():
                if not self._device_matches(value.device, target_device):
                    param_dict[name] = value.to(target_device, non_blocking=True)
            
            # EXTRA SAFETY: Before using stateless.functional_call, verify model is completely on GPU
            # Normalize device: always use 'cuda:0' for consistency
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
            # Check a sample of parameters to ensure the model is really on GPU
            try:
                sample_param = next(iter(self.model.parameters()))
                if not self._device_matches(sample_param.device, target_device):
                    print(f"    [Attacker {self.client_id}] CRITICAL: Model not fully on {target_device}, forcing move")
                    self.model.to(target_device)
                    self._ensure_model_on_device(self.model, target_device)
            except StopIteration:
                pass

            total_loss = 0.0
            batches = 0
            
            # Normalize device once for this batch loop
            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device

            for batch in self.proxy_loader:
                input_ids = batch['input_ids'].to(target_device)
                attention_mask = batch['attention_mask'].to(target_device)
                labels = batch['labels'].to(target_device)
                
                if use_lora:
                    # ============================================================================
                    # CRITICAL: LoRA mode with functional_call (NO FALLBACK)
                    # ============================================================================
                    # This path MUST preserve gradients from proxy_param to loss.
                    # PROHIBITED: NO .data operations, NO copy_(), NO no_grad() write operations
                    #
                    # Flow:
                    #   1. candidate_lora_flat = global_params + malicious_update (LoRA-only flat)
                    #   2. lora_params = _flat_to_lora_param_dict(candidate_lora_flat) [preserves gradients]
                    #   3. full_params = base_params (detached) + lora_params (with gradients)
                    #   4. functional_call(model, (full_params, full_buffers)) [preserves gradients]
                    #   5. loss = F.cross_entropy(logits, labels) [gradient flows back to proxy_param]
                    # ============================================================================
                    
                    # Step 1: Ensure candidate is LoRA-only flat (from global + malicious_update)
                    # candidate_params is already computed as: global_params_gpu + malicious_update
                    # Both global_params_gpu and malicious_update are LoRA-only flat tensors
                    candidate_lora_flat = candidate_params
                    
                    # Step 2: Convert flat LoRA to param dict (preserves gradients via view/reshape)
                    # Uses pre-computed slices from _init_functional_param_cache to extract each parameter
                    # Maintains requires_grad=True throughout, preserving computational graph
                    lora_params = self._flat_to_lora_param_dict(candidate_lora_flat)
                    
                    # Step 3: Construct full_params = base_params (constants) + lora_params (trainable)
                    # CRITICAL: base_params are detached constants (requires_grad=False),
                    #          lora_params maintain gradients (requires_grad=True)
                    full_params = dict(self.base_params)  # Shallow copy of base params (detached)
                    full_params.update(lora_params)  # Add LoRA params (with gradients)
                    
                    # Step 4: Ensure base_buffers are on correct device
                    # Buffers are constants (e.g., batch norm running means), no gradients needed
                    full_buffers = {}
                    for name, buf in self.base_buffers.items():
                        if not self._device_matches(buf.device, target_device):
                            full_buffers[name] = buf.to(target_device)
                        else:
                            full_buffers[name] = buf
                    
                    # Step 5: Use functional_call for forward pass (preserves gradients)
                    # CRITICAL: This is the ONLY valid path - no fallback allowed
                    # functional_call injects parameters without breaking computational graph
                    # If this fails, it indicates a configuration error, not a recoverable issue
                    try:
                        logits = functional_call(
                            self.model,
                            (full_params, full_buffers),
                            args=(),
                            kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
                        )
                    except (RuntimeError, KeyError, TypeError) as e:
                        # FATAL ERROR: functional_call failure indicates configuration problem
                        # Possible causes:
                        #   1. Parameter names don't match (check _init_functional_param_cache)
                        #   2. Missing parameters/buffers in full_params/full_buffers
                        #   3. Device mismatch between params and model
                        #   4. Model structure incompatibility with functional_call
                        error_msg = (
                            f"[Attacker {self.client_id}] FATAL: functional_call failed in LoRA mode: {e}\n"
                            f"This indicates a configuration error - functional_call MUST work.\n"
                            f"Check: (1) Parameter names match, (2) All params/buffers present, "
                            f"(3) Device consistency, (4) Model structure compatibility."
                        )
                        raise RuntimeError(error_msg) from e
                
                else:
                    # For full fine-tuning, try stateless.functional_call first
                    try:
                        # Final verification before calling stateless.functional_call
                        self._ensure_model_on_device(self.model, target_device)
                        self.model.to(target_device)

                        logits = stateless.functional_call(
                            self.model,
                            param_dict,
                            args=(),
                            kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
                        )
                    except (RuntimeError, KeyError) as e:
                        # If stateless.functional_call fails (e.g., parameter name mismatch in PEFT),
                        # try using the model directly with temporarily set parameters
                        # This is a fallback for PEFT model compatibility
                        print(f"    [Attacker {self.client_id}] Warning: stateless.functional_call failed: {e}")
                        print(f"    [Attacker {self.client_id}] Attempting fallback method...")
                        
                        # Fallback: temporarily set parameters, run forward, then restore
                        original_params = {}
                        try:
                            # Use normalized device for consistency: always use 'cuda:0'
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            
                            # First, ensure entire model is on correct device (defensive check)
                            # This is critical for PEFT models where base model params might not be properly moved
                            self.model.to(target_device)
                            
                            # CRITICAL: Recursively ensure ALL parameters and buffers are on GPU
                            # This is essential for PEFT models with nested structures
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # Ensure model is on target device first
                            self.model.to(target_device)
                            
                            # Verify ALL parameters are on GPU before proceeding
                            for name, param in self.model.named_parameters():
                                if not self._device_matches(param.device, target_device):
                                    print(f"    [Attacker {self.client_id}] CRITICAL in fallback: Parameter {name} on {param.device}, forcing to {target_device}")
                                    with torch.no_grad():
                                        param.data = param.data.to(target_device, non_blocking=True)
                            
                            # Verify ALL buffers are on GPU
                            for name, buffer in self.model.named_buffers():
                                if not self._device_matches(buffer.device, target_device):
                                    print(f"    [Attacker {self.client_id}] CRITICAL in fallback: Buffer {name} on {buffer.device}, forcing to {target_device}")
                                    buffer.data = buffer.data.to(target_device, non_blocking=True)
                            
                            # One more recursive check
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # Save original parameters and set new values
                            # CRITICAL: Use no_grad() for parameter setting to avoid tracking gradients
                            # We only need gradients for the forward pass, not for data copying
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in param_dict:
                                        # Save original parameter value (on current device)
                                        original_params[name] = param.data.clone()
                                        # Get new parameter value from param_dict
                                        new_value = param_dict[name]
                                        # Ensure new_value and param are both on target_device
                                        # Use normalized device matching to handle 'cuda' vs 'cuda:0'
                                        if not self._device_matches(new_value.device, target_device):
                                            new_value = new_value.to(target_device, non_blocking=True)
                                        # Ensure param is also on target_device
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        # Ensure data type matches
                                        if new_value.dtype != param.dtype:
                                            new_value = new_value.to(dtype=param.dtype)
                                        # Copy the value
                                        param.data.copy_(new_value)
                            
                            # Final verification: ensure all parameters and buffers are on correct device
                            # This is especially important for PEFT models with nested structures
                            # Double-check with recursive function (use normalized device)
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            self._ensure_model_on_device(self.model, target_device)
                            
                            # One more explicit check before forward pass
                            # Use normalized device: always 'cuda:0' for consistency
                            target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
                            for name, param in self.model.named_parameters():
                                if not self._device_matches(param.device, target_device):
                                    print(f"    [Attacker {self.client_id}] FINAL CHECK FAILED: Parameter {name} on {param.device}, should be on {target_device}")
                                    # Try one more time to fix it
                                    with torch.no_grad():
                                        param.data = param.data.to(target_device, non_blocking=True)
                            
                            # Final check for buffers
                            for name, buffer in self.model.named_buffers():
                                if not self._device_matches(buffer.device, target_device):
                                    print(f"    [Attacker {self.client_id}] FINAL CHECK FAILED: Buffer {name} on {buffer.device}, should be on {target_device}")
                                    buffer.data = buffer.data.to(target_device, non_blocking=True)
                            
                            # Run forward pass
                            # NewsClassifierModel.forward() returns logits directly
                            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                            
                            # Restore original parameters
                            # CRITICAL: Use no_grad() for parameter restoration to avoid tracking gradients
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in original_params:
                                        # Ensure restored value is on target_device
                                        restored_value = original_params[name]
                                        if not self._device_matches(restored_value.device, target_device):
                                            restored_value = restored_value.to(target_device, non_blocking=True)
                                        # Ensure param is also on target_device
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        param.data.copy_(restored_value)
                            
                            # Final check after restoration: ensure all parameters still on correct device
                            self._ensure_model_on_device(self.model, target_device)
                        except Exception as fallback_error:
                            print(f"    [Attacker {self.client_id}] Fallback method also failed: {fallback_error}")
                            # Restore parameters even if forward failed
                            # CRITICAL: Use no_grad() for parameter restoration
                            with torch.no_grad():
                                for name, param in self.model.named_parameters():
                                    if name in original_params:
                                        restored_value = original_params[name]
                                        if not self._device_matches(restored_value.device, target_device):
                                            restored_value = restored_value.to(target_device, non_blocking=True)
                                        if not self._device_matches(param.device, target_device):
                                            param.data = param.data.to(target_device, non_blocking=True)
                                        param.data.copy_(restored_value)
                            # Return zero loss as last resort
                            return torch.tensor(0.0, device=target_device)

                ce_loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + ce_loss
                batches += 1
                if batches >= max_batches:
                    break

            if batches == 0:
                result = torch.tensor(0.0, device=self.device)
            else:
                result = total_loss / batches
        except Exception as e:
            # ============================================================
            # CRITICAL: LoRA mode failures must raise, not return 0 loss
            # ============================================================
            # If LoRA mode, functional_call failure indicates FATAL error
            # Must raise to prevent silent optimization failure
            use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
            if use_lora:
                error_msg = (
                    f"[Attacker {self.client_id}] FATAL: LoRA functional_call failed in _proxy_global_loss: {e}\n"
                    f"LoRA mode requires functional_call to work - this is a configuration error.\n"
                    f"Cannot continue optimization with broken gradient link."
                )
                raise RuntimeError(error_msg) from e
            
            # Non-LoRA mode: Allow fallback (for backward compatibility)
            print(f"    [Attacker {self.client_id}] Error in proxy loss computation: {e}")
            result = torch.tensor(0.0, device=self.device)
        finally:
            # CRITICAL: Only move model back to CPU if keep_model_on_gpu=False
            # If keep_model_on_gpu=True, the loss will be used for backward pass,
            # and moving the model would break the computation graph!
            if not keep_model_on_gpu and model_was_on_cpu:
                # Before moving back, ensure all gradients are computed if needed
                # Actually, if keep_model_on_gpu is False, we assume backward is not needed
                # So it's safe to move back
                self.model.cpu()
                self._model_on_gpu = False

        return result

    def _construct_graph(self, reduced_features: torch.Tensor) -> torch.Tensor:
        """
        Construct graph according to the paper (Section III).
        
        Paper formulation:
        - Feature matrix F(t) = [w_1(t), ..., w_i(t)]^T ∈ R^{I×M}
        - Adjacency matrix A(t) ∈ R^{M×M} (NOT I×I!)
        - δ_{m,m'} = cosine similarity between w_m(t) and w_{m'}(t)
        - w_m(t) ∈ R^{I×1} is the m-th COLUMN of F(t)
        
        So we need to compute similarity between COLUMNS (parameter dimensions),
        not ROWS (clients).
        
        Args:
            reduced_features: Feature matrix F(t) ∈ R^{I×M} where I=num_clients, M=feature_dim
            
        Returns:
            Adjacency matrix A(t) ∈ R^{M×M} with binary edges based on cosine similarity threshold
        """
        # reduced_features shape: (I, M) where I=num_clients, M=feature_dim
        # We need to compute similarity between columns (parameter dimensions)
        # Transpose to get (M, I), then compute similarity
        
        # F^T shape: (M, I) - each row is a parameter dimension across all clients
        features_transposed = reduced_features.t()  # (M, I)
        
        # Normalize for cosine similarity (along dim=1, i.e., across clients)
        norm_features = F.normalize(features_transposed, p=2, dim=1)  # (M, I)
        
        # Compute adjacency matrix A ∈ R^{M×M}
        # A[m, m'] = cosine_sim(w_m, w_m') where w_m is m-th column of F
        similarity_matrix = torch.mm(norm_features, norm_features.t())  # (M, M)
        
        # Remove self-loops
        adj_matrix = similarity_matrix.clone()
        adj_matrix.fill_diagonal_(0)
        
        # Threshold for binarization (paper doesn't specify, but common practice)
        # Ensure graph_threshold is a Python float (not tensor)
        threshold = float(self.graph_threshold) if isinstance(self.graph_threshold, (int, float)) else 0.5
        adj_matrix = (adj_matrix > threshold).float()
        
        return adj_matrix

    def _train_vgae(self, adj_matrix: torch.Tensor, feature_matrix: torch.Tensor, epochs=None) -> torch.Tensor:
        """
        Train the VGAE model according to the paper.
        
        Paper formulation:
        - Input: A ∈ R^{M×M} (adjacency), F ∈ R^{I×M} (features)
        - For VGAE, we use F^T ∈ R^{M×I} as node features
        - Each node represents a parameter dimension
        - VGAE learns to reconstruct A
        
        Args:
            adj_matrix: Adjacency matrix A ∈ R^{M×M}
            feature_matrix: Feature matrix F ∈ R^{I×M}
            epochs: Number of training epochs (default: self.vgae_epochs)
            
        Returns:
            Reconstructed adjacency matrix Â ∈ R^{M×M} (detached)
        """
        if epochs is None:
            epochs = self.vgae_epochs
        
        # adj_matrix shape: (M, M) - from _construct_graph
        # feature_matrix shape: (I, M) - original features
        # For VGAE input, we use F^T as node features: (M, I)
        node_features = feature_matrix.t()  # (M, I)
        
        input_dim = int(node_features.shape[1])  # I (number of clients) - Convert to Python int
        num_nodes = int(node_features.shape[0])  # M (feature dimension) - Convert to Python int
        
        # Initialize VGAE if needed
        # Paper: input_dim = I (number of clients/benign models)
        vgae_input_dim = int(self.vgae.gc1.weight.shape[0]) if self.vgae is not None else None
        if self.vgae is None or vgae_input_dim != input_dim:
            # Use client_id-based seed for VGAE initialization to ensure diversity among attackers
            # This ensures different attackers have different VGAE initial weights, leading to different attack patterns
            import hashlib
            seed_str = f"vgae_{self.client_id}_{input_dim}_{self.vgae_hidden_dim}_{self.vgae_latent_dim}"
            vgae_seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
            
            # Save current random state
            rng_state_before = torch.get_rng_state()
            np_rng_state_before = np.random.get_state()
            
            # Set seed for VGAE initialization
            torch.manual_seed(vgae_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(vgae_seed)
            
            # Use configured VGAE architecture parameters (per paper: hidden1_dim=32, hidden2_dim=16)
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=self.vgae_hidden_dim, 
                            latent_dim=self.vgae_latent_dim, dropout=self.vgae_dropout,
                            kl_weight=self.vgae_kl_weight).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=self.vgae_lr)
            
            # Restore random state to avoid affecting other random number generation
            torch.set_rng_state(rng_state_before)
            np.random.set_state(np_rng_state_before)

        self.vgae.train()
        
        for _ in range(epochs):
            self.vgae_optimizer.zero_grad()
            
            # Forward pass: VGAE takes (node_features, adj_matrix)
            # node_features: (M, I), adj_matrix: (M, M)
            adj_recon, mu, logvar = self.vgae(node_features, adj_matrix)
            
            # Loss calculation
            loss = self.vgae.loss_function(adj_recon, adj_matrix, mu, logvar)
            
            loss.backward()
            self.vgae_optimizer.step()
        
        return adj_recon.detach()  # Return reconstructed adjacency for GSP

    def set_global_model_params(self, global_params: torch.Tensor):
        """
        Set global model parameters for constraint (4b) calculation.
        
        CRITICAL: In LoRA mode, converts full-model flat to LoRA-only flat.
        This ensures consistency with proxy_param and malicious_update (both are LoRA-only).
        
        Args:
            global_params: Global model parameters (full-model flat in non-LoRA mode,
                          may be full-model or LoRA-only flat in LoRA mode)
        """
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        
        if use_lora:
            # LoRA mode: Convert to LoRA-only flat
            # Strategy: If input is full-model flat, extract LoRA params using named_parameters()
            # If input is already LoRA-only flat and matches _flat_numel, use as-is
            
            input_numel = int(global_params.numel())
            
            if input_numel == self._flat_numel:
                # Already LoRA-only flat, use directly
                self.global_model_params = global_params.clone().detach().to(self.device)
            else:
                # Full-model flat: Extract LoRA parameters in same order as get_flat_params()
                # CRITICAL: Must match get_flat_params() order exactly
                trainables = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
                
                # Reconstruct full model params from flat (temporarily)
                # This is needed to extract LoRA params correctly
                # Note: This is a workaround - ideally server should send LoRA-only flat
                full_model_params = {}
                offset = 0
                for name, param in self.model.named_parameters():
                    numel = int(param.numel())
                    full_model_params[name] = global_params[offset:offset + numel].view_as(param.data)
                    offset += numel
                
                # CRITICAL: Verify offset matches input length (catches silent misalignment)
                if offset != input_numel:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] Full-model global_params length mismatch: "
                        f"consumed {offset}, provided {input_numel}. Check flatten order."
                    )
                
                # Extract LoRA parameters in order
                lora_params_flat = []
                for name, param in trainables:
                    if name in full_model_params:
                        lora_params_flat.append(full_model_params[name].view(-1))
                    else:
                        raise RuntimeError(
                            f"[Attacker {self.client_id}] LoRA parameter {name} not found in full_model_params"
                        )
                
                # Concatenate LoRA params to form LoRA-only flat
                self.global_model_params = torch.cat(lora_params_flat).clone().detach().to(self.device)
                
                # Verify dimension matches
                assert self.global_model_params.numel() == self._flat_numel, \
                    f"[Attacker {self.client_id}] LoRA extraction failed: " \
                    f"expected {self._flat_numel}, got {self.global_model_params.numel()}"
        else:
            # Non-LoRA mode: Use as-is
            self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, dist_bound: float = None,
                              sim_bound_low: float = None, sim_bound_up: float = None,
                              total_data_size: float = None, benign_data_sizes: dict = None):
        """
        Set constraint parameters for Formula 4.
        
        Args:
            dist_bound: Distance threshold for constraint (4b): d(w'_j(t), w'_g(t)) ≤ dist_bound
            sim_bound_low: Manual lower bound for cosine similarity (None = use benign min)
            sim_bound_up: Manual upper bound for cosine similarity (None = use benign mean)
            total_data_size: D(t) - Total data size for aggregation weight calculation (Paper Formula (2))
            benign_data_sizes: Dict {client_id: D_i(t)} - Data sizes for each benign client (Paper Formula (2))
        """
        self.dist_bound = dist_bound  # Constraint (4b): d(w'_j(t), w'_g(t)) ≤ dist_bound
        self.sim_bound_low = sim_bound_low  # Manual lower bound (None = benign min)
        self.sim_bound_up = sim_bound_up    # Manual upper bound (None = benign mean)
        self.total_data_size = total_data_size  # D(t) for weight calculation
        if benign_data_sizes is not None:
            self.benign_data_sizes = benign_data_sizes  # {client_id: D_i(t)}
    
    def set_lagrangian_params(self, use_lagrangian_dual: bool = False,
                              lambda_dist_init: float = 0.1,
                              lambda_dist_lr: float = 0.01,
                              use_cosine_similarity_constraint: bool = False,
                              use_pairwise_similarity_in_constraint: bool = False,
                              lambda_sim_low_init: float = 0.1,
                              lambda_sim_up_init: float = 0.1,
                              lambda_sim_low_lr: float = 0.01,
                              lambda_sim_up_lr: float = 0.01,
                              # ========== Augmented Lagrangian (ALM) parameters ==========
                              use_augmented_lagrangian: bool = False,
                              lambda_update_mode: str = "classic",
                              rho_dist_init: float = 1.0,
                              rho_sim_low_init: float = 1.0,
                              rho_sim_up_init: float = 1.0,
                              rho_adaptive: bool = True,
                              rho_theta: float = 0.5,
                              rho_increase_factor: float = 2.0,
                              rho_min: float = 1e-3,
                              rho_max: float = 1e3):
        """
        Set Lagrangian Dual parameters (initialized according to paper Algorithm 1)
        
        Paper reference: Section 3, Algorithm 1
        - Lagrangian function: eq:lagrangian
        - Optimization subproblem: eq:wprime_sub
        - Initialization: λ(1)≥0
        
        Args:
            use_lagrangian_dual: Whether to use Lagrangian Dual mechanism
            lambda_dist_init: Initial λ_dist(1) value (≥0, per paper Algorithm 1)
            lambda_dist_lr: Learning rate for λ_dist(t) update (subgradient step size)
            use_cosine_similarity_constraint: Whether to use cosine similarity constraints
            lambda_sim_low_init: Initial λ_sim_low value (≥0) for lower bound constraint
            lambda_sim_up_init: Initial λ_sim_up value (≥0) for upper bound constraint
            lambda_sim_low_lr: Learning rate for λ_sim_low(t) update
            lambda_sim_up_lr: Learning rate for λ_sim_up(t) update
        
        Modification 2: Save initial values for reset in prepare_for_round
        """
        self.use_lagrangian_dual = use_lagrangian_dual
        # Note: dist_bound is set by server, used only as fallback when distance prediction is disabled or no history
        if use_lagrangian_dual:
            # Paper: λ(1)≥0
            # Modification 2: Save initial values for reset each round
            self.lambda_dist_init = max(0.0, lambda_dist_init)
            self.lambda_dist = torch.tensor(self.lambda_dist_init, requires_grad=False)
            self.lambda_dist_lr = lambda_dist_lr
            
            # Cosine similarity constraint parameters (TWO-SIDED with TWO multipliers)
            self.use_cosine_similarity_constraint = use_cosine_similarity_constraint
            self.use_pairwise_similarity_in_constraint = bool(use_pairwise_similarity_in_constraint) and use_cosine_similarity_constraint
            if use_cosine_similarity_constraint:
                # Lower bound constraint: g_sim_low = sim_bound_low - sim_att <= 0
                self.lambda_sim_low_init = max(0.0, lambda_sim_low_init)
                self.lambda_sim_low = torch.tensor(self.lambda_sim_low_init, requires_grad=False)
                self.lambda_sim_low_lr = lambda_sim_low_lr
                
                # Upper bound constraint: g_sim_up = sim_att - sim_bound_up <= 0
                self.lambda_sim_up_init = max(0.0, lambda_sim_up_init)
                self.lambda_sim_up = torch.tensor(self.lambda_sim_up_init, requires_grad=False)
                self.lambda_sim_up_lr = lambda_sim_up_lr
            else:
                self.lambda_sim_low = None
                self.lambda_sim_up = None
                self.lambda_sim_low_init = None
                self.lambda_sim_up_init = None

            # ===================== ALM parameters =====================
            # ALM is meaningful only when using Lagrangian framework (use_lagrangian_dual=True).
            self.use_augmented_lagrangian = bool(use_augmented_lagrangian)
            self.lambda_update_mode = str(lambda_update_mode or "classic").lower()
            if self.lambda_update_mode not in ("classic", "alm"):
                raise ValueError(f"Invalid lambda_update_mode={lambda_update_mode!r}, expected 'classic' or 'alm'")

            # Penalty parameters ρ (must be positive)
            self.rho_adaptive = bool(rho_adaptive)
            self.rho_theta = float(rho_theta)
            self.rho_increase_factor = float(rho_increase_factor)
            self.rho_min = float(rho_min)
            self.rho_max = float(rho_max)

            if self.use_augmented_lagrangian:
                self.rho_dist_init = max(self.rho_min, float(rho_dist_init))
                self.rho_dist = torch.tensor(self.rho_dist_init, requires_grad=False)
                if self.use_cosine_similarity_constraint:
                    self.rho_sim_low_init = max(self.rho_min, float(rho_sim_low_init))
                    self.rho_sim_low = torch.tensor(self.rho_sim_low_init, requires_grad=False)
                    self.rho_sim_up_init = max(self.rho_min, float(rho_sim_up_init))
                    self.rho_sim_up = torch.tensor(self.rho_sim_up_init, requires_grad=False)
                else:
                    self.rho_sim_low = None
                    self.rho_sim_up = None
                    self.rho_sim_low_init = None
                    self.rho_sim_up_init = None
            else:
                # Standard Lagrangian only (no quadratic penalty)
                self.rho_dist = None
                self.rho_sim_low = None
                self.rho_sim_up = None
                self.rho_dist_init = None
                self.rho_sim_low_init = None
                self.rho_sim_up_init = None

            # Reset per-round violation history for ρ update
            self._prev_violation_dist = None
            self._prev_violation_sim_low = None
            self._prev_violation_sim_up = None
        else:
            # Hard constraint mode (Lagrangian disabled)
            self.lambda_dist = None
            self.lambda_dist_init = None
            self.use_cosine_similarity_constraint = False
            self.use_pairwise_similarity_in_constraint = False
            # Disable ALM as well
            self.use_augmented_lagrangian = False
            self.rho_dist = None
            self.rho_sim_low = None
            self.rho_sim_up = None
            self.rho_dist_init = None
            self.rho_sim_low_init = None
            self.rho_sim_up_init = None
            self._prev_violation_dist = None
            self._prev_violation_sim_low = None
            self._prev_violation_sim_up = None

    def _aggregate_update_no_beta(self, malicious_update: torch.Tensor, 
                                   benign_updates: List[torch.Tensor] = None,
                                   benign_updates_gpu: List[torch.Tensor] = None,
                                   other_attacker_updates_list: List[torch.Tensor] = None,
                                   other_attacker_updates_gpu: List[torch.Tensor] = None,
                                   other_attacker_updates: List[torch.Tensor] = None,
                                   include_current_attacker: bool = True) -> Tuple[torch.Tensor, float, List[float]]:
        # #region agent log
        import json
        try:
            with open('/Users/hanlincai/Desktop/Github/IoA-Attack-GRMP/.cursor/debug.log', 'a') as f:
                log_entry = {
                    "id": f"log_{int(__import__('time').time())}_{id(self)}",
                    "timestamp": int(__import__('time').time() * 1000),
                    "location": "client.py:1552",
                    "message": "_aggregate_update_no_beta called",
                    "data": {
                        "client_id": getattr(self, 'client_id', None),
                        "has_other_attacker_updates": other_attacker_updates is not None,
                        "has_other_attacker_updates_list": other_attacker_updates_list is not None,
                        "has_other_attacker_updates_gpu": other_attacker_updates_gpu is not None
                    },
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }
                f.write(json.dumps(log_entry) + '\n')
        except: pass
        # #endregion
        # Handle legacy parameter name: other_attacker_updates -> other_attacker_updates_list
        if other_attacker_updates is not None and other_attacker_updates_list is None:
            other_attacker_updates_list = other_attacker_updates
        """
        FedAvg-style aggregated update (NO beta selection).
        
        Aggregated update:
            If include_current_attacker=True:
                Δ_g = Σ_i (D_i / D_eff) * Δ_i + (D_att / D_eff) * Δ_att + Σ_j (D_j / D_eff) * Δ_j
            If include_current_attacker=False:
                Δ_g = Σ_i (D_i / D_eff) * Δ_i + Σ_j (D_j / D_eff) * Δ_j  (excludes current attacker)
        
        where D_eff = Σ D_i + D_att + Σ D_j (all participants) if include_current_attacker=True,
              or D_eff = Σ D_i + Σ D_j (excluding current attacker) if include_current_attacker=False.
        
        Args:
            malicious_update: Δ_att (attacker's update, should be on target device)
            benign_updates: List of benign updates (CPU, fallback)
            benign_updates_gpu: List of benign updates (GPU, preferred)
            other_attacker_updates_list: List of other attacker updates (CPU, fallback)
            other_attacker_updates_gpu: List of other attacker updates (GPU, preferred)
            include_current_attacker: Whether to include current attacker's update in aggregation (default: True)
        
        Returns:
            aggregated_update: Δ_g (aggregated update)
            w_att: attacker weight (D_att / D_eff) if include_current_attacker=True, else 0.0
            w_ben: list of benign weights [D_i / D_eff]
        """
        # ===== CRITICAL: Use GPU versions if available to avoid device transfers =====
        # Prefer GPU versions to maintain computation graph integrity
        if benign_updates_gpu is not None and len(benign_updates_gpu) > 0:
            benign_updates_to_use = benign_updates_gpu
        elif benign_updates is not None and len(benign_updates) > 0:
            benign_updates_to_use = benign_updates
        else:
            benign_updates_to_use = self.benign_updates if hasattr(self, 'benign_updates') else []
        
        device = malicious_update.device
        D_att = float(self.claimed_data_size) if include_current_attacker else 0.0
        
        # Collect benign data sizes
        D_sum = D_att if include_current_attacker else 0.0
        D_list = []
        for idx in range(len(benign_updates_to_use)):
            # Get client_id from benign_update_client_ids if available
            if hasattr(self, 'benign_update_client_ids') and idx < len(self.benign_update_client_ids):
                client_id = self.benign_update_client_ids[idx]
            else:
                client_id = idx  # Fallback to index
            
            # Get data size for this client
            if hasattr(self, 'benign_data_sizes') and client_id in self.benign_data_sizes:
                D_i = float(self.benign_data_sizes[client_id])
            else:
                D_i = 1.0  # Fallback
            
            D_list.append(D_i)
            D_sum += D_i
        
        # ===== CRITICAL: Use GPU versions if available to avoid device transfers =====
        # Prefer GPU versions to maintain computation graph integrity
        if benign_updates_gpu is not None and len(benign_updates_gpu) > 0:
            benign_updates_to_use = benign_updates_gpu
        elif benign_updates is not None:
            benign_updates_to_use = benign_updates
        else:
            benign_updates_to_use = self.benign_updates
        
        # Determine device from malicious_update (should already be on target device)
        device = malicious_update.device
        
        # ===== NEW: Include other attackers' updates for coordinated optimization =====
        # CRITICAL: Do NOT shadow the input parameter other_attacker_updates_list
        # Use a different name for the local accumulator
        other_attacker_weights = []
        other_updates_to_use = []  # Local accumulator, NOT shadowing input
        if other_attacker_updates_gpu is not None and len(other_attacker_updates_gpu) > 0:
            # Use GPU versions
            for idx, cid in enumerate(self.other_attacker_client_ids):
                if idx < len(other_attacker_updates_gpu):
                    if hasattr(self, 'other_attacker_data_sizes') and cid in self.other_attacker_data_sizes:
                        D_j = float(self.other_attacker_data_sizes[cid])
                    else:
                        D_j = float(self.claimed_data_size)
                    other_attacker_weights.append(D_j)
                    other_updates_to_use.append(other_attacker_updates_gpu[idx])
                    D_sum += D_j
        elif other_attacker_updates_list is not None and len(other_attacker_updates_list) > 0:
            # Use provided input parameter (CPU versions)
            for idx, cid in enumerate(self.other_attacker_client_ids):
                if idx < len(other_attacker_updates_list):
                    if hasattr(self, 'other_attacker_data_sizes') and cid in self.other_attacker_data_sizes:
                        D_j = float(self.other_attacker_data_sizes[cid])
                    else:
                        D_j = float(self.claimed_data_size)
                    other_attacker_weights.append(D_j)
                    other_updates_to_use.append(other_attacker_updates_list[idx])  # Use INPUT parameter
                    D_sum += D_j
        elif hasattr(self, 'other_attacker_updates') and self.other_attacker_updates:
            # Fallback to stored CPU versions
            for idx, cid in enumerate(self.other_attacker_client_ids):
                if idx < len(self.other_attacker_updates):
                    if hasattr(self, 'other_attacker_data_sizes') and cid in self.other_attacker_data_sizes:
                        D_j = float(self.other_attacker_data_sizes[cid])
                    else:
                        D_j = float(self.claimed_data_size)
                    other_attacker_weights.append(D_j)
                    other_updates_to_use.append(self.other_attacker_updates[idx])
                    D_sum += D_j
        # ==============================================================================
        
        # Compute weights
        denom = D_sum + 1e-12
        w_att = D_att / denom if include_current_attacker else 0.0
        w_ben = [D_i / denom for D_i in D_list]
        w_other_att = [D_j / denom for D_j in other_attacker_weights]
        
        # Aggregate updates: Δ_g = Σ w_i * Δ_i + Σ w_j * Δ_j [+ w_att * Δ_att if include_current_attacker]
        # CRITICAL: Initialize agg from malicious_update to preserve gradient connection
        # Even when include_current_attacker=False, we need to maintain gradient flow
        # Solution: Start with zero tensor but ensure it's connected to malicious_update's computation graph
        agg = torch.zeros_like(malicious_update, device=device).view(-1)
        # CRITICAL: Ensure agg requires grad if malicious_update requires grad (for gradient flow)
        if malicious_update.requires_grad:
            agg = agg + 0.0 * malicious_update.view(-1)  # Connect to computation graph
        
        # Add benign updates (GPU versions should already be on correct device)
        # CRITICAL: In optimization loop, GPU versions are pre-transferred to target_device
        # For final check (after GPU cleanup), allow device conversion to match malicious_update
        for w, benign_update in zip(w_ben, benign_updates_to_use):
            # In optimization loop: strict device check (GPU versions should match)
            # After optimization: allow conversion (final check uses CPU versions)
            if benign_update.device != device:
                # Allow conversion only if not in optimization loop (GPU caches cleaned up)
                # Check if we're in final check phase (GPU caches don't exist)
                if not (hasattr(self, 'benign_updates_gpu') and len(getattr(self, 'benign_updates_gpu', [])) > 0):
                    # Final check phase: allow conversion
                    benign_update = benign_update.to(device)
                else:
                    # Optimization loop: strict check (should not happen)
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] CRITICAL: Device mismatch in optimization loop! "
                        f"benign_update on {benign_update.device}, expected {device}. "
                        f"This should not happen if GPU versions are created correctly."
                    )
            agg = agg + w * benign_update.view(-1)
        
        # ===== NEW: Add other attackers' updates =====
        # CRITICAL: In optimization loop, GPU versions are pre-transferred to target_device
        # For final check (after GPU cleanup), allow device conversion to match malicious_update
        for w, other_attacker_update in zip(w_other_att, other_updates_to_use):
            # In optimization loop: strict device check (GPU versions should match)
            # After optimization: allow conversion (final check uses CPU versions)
            if other_attacker_update.device != device:
                # Allow conversion only if not in optimization loop (GPU caches cleaned up)
                # Check if we're in final check phase (GPU caches don't exist)
                if not (hasattr(self, 'other_attacker_updates_gpu') and len(getattr(self, 'other_attacker_updates_gpu', [])) > 0):
                    # Final check phase: allow conversion
                    other_attacker_update = other_attacker_update.to(device)
                else:
                    # Optimization loop: strict check (should not happen)
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] CRITICAL: Device mismatch in optimization loop! "
                        f"other_attacker_update on {other_attacker_update.device}, expected {device}. "
                        f"This should not happen if GPU versions are created correctly."
                    )
            agg = agg + w * other_attacker_update.view(-1)
        # ==============================================
        
        # Add current attacker's update (only if include_current_attacker=True)
        if include_current_attacker:
            agg = agg + w_att * malicious_update.view(-1)
        
        return agg.view(-1), w_att, w_ben
    
    def _aggregate_benign_only(self, benign_updates: List[torch.Tensor], device=None) -> torch.Tensor:
        """
        Aggregate ONLY benign updates (no attackers) for statistics computation.
        
        This ensures all attackers get the same dist_bound and sim_bound values when using
        automatic calculation based on benign statistics.
        
        Aggregated update:
            Δ_g_benign = Σ_i (D_i / D_benign_sum) * Δ_i
        
        where D_benign_sum = Σ D_i (only benign clients).
        
        Args:
            benign_updates: List of all benign updates Δ_i
            device: Target device for the aggregated tensor (None = use first update's device)
        
        Returns:
            aggregated_update: Δ_g_benign (aggregated update from benign clients only)
        """
        if len(benign_updates) == 0:
            # Return zero tensor if no benign updates
            # This should not happen in practice, but handle gracefully
            target_device = device if device is not None else self.device
            return torch.zeros(1, device=target_device)
        
        if device is None:
            device = benign_updates[0].device
        else:
            device = device
        D_list = []
        D_sum = 0.0
        
        # Collect benign data sizes
        for idx in range(len(benign_updates)):
            # Get client_id from benign_update_client_ids if available
            if hasattr(self, 'benign_update_client_ids') and idx < len(self.benign_update_client_ids):
                client_id = self.benign_update_client_ids[idx]
            else:
                client_id = idx  # Fallback to index
            
            # Get data size for this client
            if hasattr(self, 'benign_data_sizes') and client_id in self.benign_data_sizes:
                D_i = float(self.benign_data_sizes[client_id])
            else:
                D_i = 1.0  # Fallback
            
            D_list.append(D_i)
            D_sum += D_i
        
        # Compute weights (only benign)
        denom = D_sum + 1e-12
        w_ben = [D_i / denom for D_i in D_list]
        
        # Aggregate updates: Δ_g_benign = Σ w_i * Δ_i (only benign)
        # Get shape from first update, but place on target device
        first_update_shape = benign_updates[0].shape
        agg = torch.zeros(first_update_shape, device=device).view(-1)
        
        # Add benign updates only
        # OPTIMIZATION: Check device before converting (avoid unnecessary .to() calls)
        for w, benign_update in zip(w_ben, benign_updates):
            # Only convert if device mismatch (PyTorch optimizes .to() on same device, but explicit check is clearer)
            if benign_update.device != device:
                benign_update = benign_update.to(device)
            agg = agg + w * benign_update.view(-1)
        
        return agg.view(-1)
    
    def _aggregate_global_reference(self, benign_updates: List[torch.Tensor],
                                    other_attacker_updates: List[torch.Tensor] = None,
                                    other_attacker_updates_gpu: List[torch.Tensor] = None,
                                    current_attacker_update: torch.Tensor = None,
                                    device=None) -> torch.Tensor:
        """
        Aggregate global reference update (benign + other attackers + current attacker).
        
        This is the reference point for constraint calculations that represents the actual
        final server aggregation (includes ALL participants), making constraints directly
        aligned with the server's final aggregation result.
        
        Aggregated update:
            Δ_g_ref = Σ_i (D_i / D_total) * Δ_i + Σ_j (D_j / D_total) * Δ_j + (D_att / D_total) * Δ_att
        
        where D_total = Σ D_i + Σ D_j + D_att (ALL participants including current attacker).
        
        This ensures the constraint dist(Δ_att, Δ_g_ref) directly measures the distance
        between the current attacker's update and the final aggregated update.
        
        Args:
            benign_updates: List of benign updates Δ_i
            other_attacker_updates: List of other attacker updates (CPU, fallback)
            other_attacker_updates_gpu: List of other attacker updates (GPU, preferred)
            current_attacker_update: Current attacker's update Δ_att (optional, if None, compute without it)
            device: Target device for the aggregated tensor (None = use first update's device)
        
        Returns:
            aggregated_update: Δ_g_ref (global reference update: ALL participants including current attacker)
        """
        # Determine device
        if device is None:
            if len(benign_updates) > 0:
                device = benign_updates[0].device
            elif other_attacker_updates_gpu is not None and len(other_attacker_updates_gpu) > 0:
                device = other_attacker_updates_gpu[0].device
            elif other_attacker_updates is not None and len(other_attacker_updates) > 0:
                device = other_attacker_updates[0].device
            else:
                device = torch.device('cpu')
        else:
            device = device
        
        if len(benign_updates) == 0 and (other_attacker_updates is None or len(other_attacker_updates) == 0) and (other_attacker_updates_gpu is None or len(other_attacker_updates_gpu) == 0):
            # Return zero tensor if no updates
            return torch.zeros(self._flat_numel, device=device)
        
        # Use GPU versions if available
        if other_attacker_updates_gpu is not None and len(other_attacker_updates_gpu) > 0:
            other_updates_to_use = other_attacker_updates_gpu
        elif other_attacker_updates is not None and len(other_attacker_updates) > 0:
            other_updates_to_use = other_attacker_updates
        else:
            other_updates_to_use = []
        
        D_sum = 0.0
        D_list = []
        
        # Collect benign data sizes
        for idx in range(len(benign_updates)):
            # Get client_id from benign_update_client_ids if available
            if hasattr(self, 'benign_update_client_ids') and idx < len(self.benign_update_client_ids):
                client_id = self.benign_update_client_ids[idx]
            else:
                client_id = idx  # Fallback to index
            
            # Get data size for this client
            if hasattr(self, 'benign_data_sizes') and client_id in self.benign_data_sizes:
                D_i = float(self.benign_data_sizes[client_id])
            else:
                D_i = 1.0  # Fallback
            
            D_list.append(D_i)
            D_sum += D_i
        
        # Collect other attacker data sizes
        other_attacker_weights = []
        for idx, cid in enumerate(self.other_attacker_client_ids if hasattr(self, 'other_attacker_client_ids') else []):
            if idx < len(other_updates_to_use):
                if hasattr(self, 'other_attacker_data_sizes') and cid in self.other_attacker_data_sizes:
                    D_j = float(self.other_attacker_data_sizes[cid])
                else:
                    D_j = float(self.claimed_data_size)
                other_attacker_weights.append(D_j)
                D_sum += D_j
        
        # Add current attacker data size if provided
        current_attacker_weight = 0.0
        if current_attacker_update is not None:
            current_attacker_weight = float(self.claimed_data_size)
            D_sum += current_attacker_weight
        
        # Compute weights
        denom = D_sum + 1e-12
        w_ben = [D_i / denom for D_i in D_list]
        w_other_att = [D_j / denom for D_j in other_attacker_weights]
        w_current_att = current_attacker_weight / denom if current_attacker_update is not None else 0.0
        
        # Aggregate updates: Δ_g_ref = Σ w_i * Δ_i + Σ w_j * Δ_j + w_att * Δ_att
        if len(benign_updates) > 0:
            first_update_shape = benign_updates[0].shape
        elif len(other_updates_to_use) > 0:
            first_update_shape = other_updates_to_use[0].shape
        else:
            first_update_shape = (self._flat_numel,)
        agg = torch.zeros(first_update_shape, device=device).view(-1)
        
        # Add benign updates
        for w, benign_update in zip(w_ben, benign_updates):
            agg = agg + w * benign_update.to(device).view(-1)
        
        # Add other attacker updates
        for w, other_attacker_update in zip(w_other_att, other_updates_to_use):
            agg = agg + w * other_attacker_update.to(device).view(-1)
        
        # Add current attacker update if provided
        if current_attacker_update is not None and w_current_att > 0:
            agg = agg + w_current_att * current_attacker_update.to(device).view(-1)
        
        return agg.view(-1)
    
    def _compute_distance_update_space(self, malicious_update: torch.Tensor,
                                        benign_updates: List[torch.Tensor] = None,
                                        benign_updates_gpu: List[torch.Tensor] = None,
                                        other_attacker_updates_gpu: List[torch.Tensor] = None,
                                        include_current_attacker: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distance in UPDATE space: d(Δ_att, Δ_g) = ||Δ_att - Δ_g||.
        
        This is the correct constraint interpretation:
        - w'_j = w_g + Δ_att
        - w'_g = w_g + Δ_g
        => d(w'_j, w'_g) = ||Δ_att - Δ_g||
        
        Args:
            malicious_update: Δ_att (attacker's update, should be on target device)
            benign_updates: List of benign updates (CPU, fallback)
            benign_updates_gpu: List of benign updates (GPU, preferred)
            other_attacker_updates_gpu: List of other attacker updates (GPU, preferred)
            include_current_attacker: Whether to include current attacker in aggregation (default: False for optimization)
        
        Returns:
            distance: ||Δ_att - Δ_g||
            aggregated_update: Δ_g (for reuse)
        """
        aggregated_update, _, _ = self._aggregate_update_no_beta(
            malicious_update,
            benign_updates=benign_updates,
            benign_updates_gpu=benign_updates_gpu,
            other_attacker_updates_gpu=other_attacker_updates_gpu,
            include_current_attacker=include_current_attacker
        )
        diff = malicious_update.view(-1) - aggregated_update.view(-1)
        distance = torch.norm(diff)
        return distance, aggregated_update
    
    def _compute_cosine_similarity_to_aggregated(self, malicious_update: torch.Tensor,
                                                 benign_updates: List[torch.Tensor] = None,
                                                 benign_updates_gpu: List[torch.Tensor] = None,
                                                 other_attacker_updates_gpu: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cosine similarity between attacker update and aggregated update.
        
        Similar to distance calculation, uses standard aggregation (including attacker).
        sim_att = cosine_similarity(Δ_att, Δ_g)
        where Δ_g = Σ_i (D_i / D_eff) * Δ_i + (D_att / D_eff) * Δ_att (includes attacker)
        
        Args:
            malicious_update: Δ_att (attacker's update, should be on target device)
            benign_updates: List of benign updates (CPU, fallback)
            benign_updates_gpu: List of benign updates (GPU, preferred)
            other_attacker_updates_gpu: List of other attacker updates (GPU, preferred)
        
        Returns:
            cosine_similarity: cos(Δ_att, Δ_g) ∈ [-1, 1]
        """
        # Compute aggregated update (same as distance calculation)
        aggregated_update, _, _ = self._aggregate_update_no_beta(
            malicious_update,
            benign_updates=benign_updates,
            benign_updates_gpu=benign_updates_gpu,
            other_attacker_updates_gpu=other_attacker_updates_gpu
        )
        
        # Flatten tensors
        malicious_flat = malicious_update.view(-1)
        aggregated_flat = aggregated_update.view(-1)
        
        # Compute cosine similarity
        sim = torch.cosine_similarity(
            malicious_flat.unsqueeze(0),
            aggregated_flat.unsqueeze(0),
            dim=1
        )
        
        return sim.squeeze(0)  # Return scalar tensor
    
    def _compute_benign_cosine_similarity_statistics(self, benign_updates: List[torch.Tensor],
                                                     aggregated_ref: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarity statistics for updates relative to a reference aggregation.
        
        For each update Δ_i, compute cosine_similarity(Δ_i, Δ_ref),
        where Δ_ref is the provided aggregated reference update.
        
        CRITICAL: 
        1. For stable optimization, the reference should NOT include the current attacker.
        2. Only benign clients are used for statistics calculation (other attackers are excluded).
           This ensures clean baseline for threshold calculation.
        
        Args:
            benign_updates: List of all benign updates Δ_i
            aggregated_ref: Reference aggregated update (if None, compute from benign_updates only)
                           Should NOT include current attacker for stable optimization
        
        Returns:
            Dictionary with statistics: mean, std, min, max, median (computed from benign clients only)
        """
        # Use provided reference or compute from benign updates only
        if aggregated_ref is None:
            aggregated_ref = self._aggregate_benign_only(benign_updates, device=None)
        
        aggregated_flat = aggregated_ref.view(-1)
        device = aggregated_flat.device
        
        # Collect updates to compute statistics (ONLY benign clients, excluding other attackers)
        # CRITICAL: Only use benign clients for threshold calculation to ensure clean baseline
        all_updates = benign_updates.copy()
        # NOTE: other_attacker_updates are explicitly excluded from statistics calculation
        
        if len(all_updates) == 0:
            # Return dummy statistics if no updates
            return {
                'mean': torch.tensor(1.0, device=device),
                'std': torch.tensor(0.0, device=device),
                'min': torch.tensor(1.0, device=device),
                'max': torch.tensor(1.0, device=device),
                'median': torch.tensor(1.0, device=device)
            }
        
        # Compute similarity for each update relative to the reference aggregation
        all_similarities = []
        for update in all_updates:
            update_flat = update.view(-1).to(device)
            sim = torch.cosine_similarity(
                update_flat.unsqueeze(0),
                aggregated_flat.unsqueeze(0),
                dim=1
            )
            all_similarities.append(sim.squeeze(0))
        
        if len(all_similarities) == 0:
            # Return dummy statistics if no similarities
            return {
                'mean': torch.tensor(1.0, device=device),
                'std': torch.tensor(0.0, device=device),
                'min': torch.tensor(1.0, device=device),
                'max': torch.tensor(1.0, device=device),
                'median': torch.tensor(1.0, device=device)
            }
        
        all_similarities_tensor = torch.stack(all_similarities)
        
        return {
            'mean': all_similarities_tensor.mean(),
            'std': all_similarities_tensor.std(),
            'min': all_similarities_tensor.min(),
            'max': all_similarities_tensor.max(),
            'median': all_similarities_tensor.median()
        }
    
    def _compute_benign_pairwise_similarity_statistics(self, benign_updates: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise cosine similarity statistics among benign updates only (aligns with server pairwise mode).
        For each benign i: s_i = (1/(B-1)) sum_{j!=i} cos(Δ_i, Δ_j). Then return mean, std, min, max, median of s_i.
        Used when use_pairwise_similarity_in_constraint=True so constraint bounds match server's pairwise metric.
        """
        if not benign_updates:
            device = next(self.model.parameters()).device
            return {
                'mean': torch.tensor(1.0, device=device),
                'std': torch.tensor(0.0, device=device),
                'min': torch.tensor(1.0, device=device),
                'max': torch.tensor(1.0, device=device),
                'median': torch.tensor(1.0, device=device)
            }
        stacked = torch.stack([u.view(-1).float() for u in benign_updates])
        device = stacked.device
        normalized = F.normalize(stacked, p=2, dim=1)
        S = normalized @ normalized.T
        B = S.shape[0]
        if B == 1:
            derived = torch.tensor([1.0], device=device)
        else:
            derived = torch.zeros(B, device=device)
            for i in range(B):
                others = torch.cat([S[i, :i], S[i, i+1:]])
                derived[i] = others.mean()
        return {
            'mean': derived.mean(),
            'std': derived.std() if B > 1 else torch.tensor(0.0, device=device),
            'min': derived.min(),
            'max': derived.max(),
            'median': derived.median()
        }
    
    def _compute_benign_distance_statistics(self, benign_updates: List[torch.Tensor],
                                           benign_ref_update: torch.Tensor = None,
                                           current_attacker_update: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute statistics of distances from updates to a reference aggregation.
        
        For each update Δ_i, compute distance ||Δ_i - Δ_ref||,
        where Δ_ref is the provided reference aggregated update.
        
        CRITICAL: 
        1. For stable optimization, the reference should NOT include the current attacker.
        2. Only benign clients are used for statistics calculation (other attackers are excluded).
           This ensures clean baseline for threshold calculation.
        
        Args:
            benign_updates: List of all benign updates Δ_i
            benign_ref_update: Reference aggregated update (if None, compute from benign_updates only)
                             Should NOT include current attacker for stable optimization
            current_attacker_update: Optional current attacker's update (if provided and benign_ref_update is None,
                                    will be included in aggregation; but typically should be None for stable stats)
        
        Returns:
            Dictionary with statistics: mean, std, min, max, median (computed from benign clients only)
        """
        # Compute or use provided reference update
        # CRITICAL: For stable optimization, should NOT include current attacker
        if benign_ref_update is None:
            if current_attacker_update is not None:
                # If current_attacker_update is provided, include it (but this is not recommended for optimization)
                benign_ref_update = self._aggregate_global_reference(
                    benign_updates=benign_updates,
                    other_attacker_updates=getattr(self, 'other_attacker_updates', None),
                    current_attacker_update=current_attacker_update,
                    device=None
                )
            else:
                # Use aggregation without current attacker (recommended for stable optimization)
                # CRITICAL: Only use benign updates for reference aggregation (exclude other attackers)
                # This ensures clean baseline for threshold calculation
                benign_ref_update = self._aggregate_benign_only(benign_updates, device=None)
                # NOTE: other_attacker_updates are explicitly excluded from reference aggregation
        
        aggregated_flat = benign_ref_update.view(-1)
        device = aggregated_flat.device
        
        # Collect updates for statistics (ONLY benign clients, excluding other attackers and current attacker)
        # CRITICAL: 
        # 1. Exclude current attacker to avoid circular dependency in statistics
        # 2. Exclude other attackers to ensure clean baseline for threshold calculation
        all_updates = benign_updates.copy()
        # NOTE: other_attacker_updates are explicitly excluded from statistics calculation
        # NOTE: current_attacker_update is NOT included in statistics to avoid circular dependency
        
        # Compute distance for each update (benign clients only, excluding all attackers)
        all_distances = []
        for update in all_updates:
            update_flat = update.view(-1).to(device)
            diff = update_flat - aggregated_flat
            dist = torch.norm(diff)
            all_distances.append(dist)
        
        if len(all_distances) == 0:
            # Return dummy statistics if no updates
            return {
                'mean': torch.tensor(0.0, device=device),
                'std': torch.tensor(0.0, device=device),
                'min': torch.tensor(0.0, device=device),
                'max': torch.tensor(0.0, device=device),
                'median': torch.tensor(0.0, device=device)
            }
        
        all_distances_tensor = torch.stack(all_distances)
        
        return {
            'mean': all_distances_tensor.mean(),
            'std': all_distances_tensor.std(),
            'min': all_distances_tensor.min(),
            'max': all_distances_tensor.max(),
            'median': all_distances_tensor.median()
        }
    
    def _compute_global_loss(self, malicious_update: torch.Tensor, 
                            benign_updates: List[torch.Tensor] = None,
                            benign_updates_gpu: List[torch.Tensor] = None,
                            other_attacker_updates_gpu: List[torch.Tensor] = None,
                            include_current_attacker: bool = False) -> torch.Tensor:
        """
        Compute global loss F(w'_g) where w'_g = w_g + Δ_g (aggregated update).
        
        This is the CORRECT interpretation: optimize loss of the aggregated global model,
        not individual client models.
        
        Objective: maximize F(w_g + Δ_g) where Δ_g is the FedAvg aggregated update.
        
        Args:
            malicious_update: Δ_att (attacker's update, should be on target device)
            benign_updates: List of benign updates (CPU, fallback)
            benign_updates_gpu: List of benign updates (GPU, preferred)
            other_attacker_updates_gpu: List of other attacker updates (GPU, preferred)
            include_current_attacker: Whether to include current attacker in aggregation (default: False for optimization)
        
        Returns:
            Proxy for F(w'_g) to be maximized
        """
        # Compute aggregated update Δ_g
        aggregated_update, _, _ = self._aggregate_update_no_beta(
            malicious_update,
            benign_updates=benign_updates,
            benign_updates_gpu=benign_updates_gpu,
            other_attacker_updates_gpu=other_attacker_updates_gpu,
            include_current_attacker=include_current_attacker
        )
        
        # Compute loss on aggregated model: F(w_g + Δ_g)
        return self._proxy_global_loss(
            aggregated_update,
            max_batches=self.proxy_max_batches_opt,
            skip_dim_check=True,
            keep_model_on_gpu=True
        )
    
    def _compute_real_distance_to_global(self, malicious_update: torch.Tensor,
                                         benign_updates: List[torch.Tensor],
                                         legacy_param: Any = None) -> torch.Tensor:
        """
        Compute distance in UPDATE space (NEW implementation).
        
        Legacy signature preserved for backward compatibility, but now uses
        _compute_distance_update_space() internally.
        
        Returns:
            distance: ||Δ_att - Δ_g|| in UPDATE space
        """
        # Use new UPDATE space distance computation
        distance, _ = self._compute_distance_update_space(malicious_update, benign_updates)
        return distance
    
    # [DEPRECATED] Old model-space distance and logging functions - not used in optimization
    def _compute_real_distance_to_global_OLD_MODEL_SPACE(self, malicious_update, selected_benign, beta_selection):
        """[DEPRECATED] Old model-space distance. See _compute_distance_update_space() for current implementation."""
        # This function is kept for backward compatibility but should not be used
        # Use _compute_distance_update_space() instead
        dist, _ = self._compute_distance_update_space(malicious_update, self.benign_updates)
        return dist

    def _gsp_generate_malicious(self, feature_matrix: torch.Tensor, 
                                  adj_orig: torch.Tensor, adj_recon: torch.Tensor,
                                  poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        Graph Signal Processing (GSP) module according to the paper (Section III).
        
        Paper formulation:
        1. L = diag(A·1) - A                 (Laplacian of original graph)
        2. L = B Λ B^T                       (SVD decomposition)
        3. S = F · B                         (GFT coefficient matrix)
        4. L̂ = diag(Â·1) - Â                 (Laplacian of reconstructed graph)
        5. L̂ = B̂ Λ̂ B̂^T                       (SVD decomposition)
        6. F̂ = S · B̂^T                       (Reconstructed feature matrix)
        7. w'_j(t) selected from F̂           (Malicious model)
        
        Args:
            feature_matrix: F ∈ R^{I×M} - benign model features (reduced dimension)
            adj_orig: A ∈ R^{M×M} - original adjacency matrix
            adj_recon: Â ∈ R^{M×M} - reconstructed adjacency matrix from VGAE
            poisoned_update: Zero update (attackers don't train, unused parameter for compatibility)
            
        Returns:
            Malicious update generated using GSP (reduced dimension M, or None if failed)
        """
        # Ensure feature_matrix is valid and has correct shape
        if feature_matrix is None or not isinstance(feature_matrix, torch.Tensor):
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix is None or invalid")
        if len(feature_matrix.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix must be 2D, got shape {feature_matrix.shape}")
        # Get shape dimension - ensure it's a valid integer
        shape_dim = feature_matrix.shape[1]
        if shape_dim is None:
            raise ValueError(f"[Attacker {self.client_id}] feature_matrix.shape[1] is None")
        try:
            M = int(shape_dim)  # Reduced dimension - Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert shape[1]={shape_dim} to int: {e}")
        if M <= 0:
            raise ValueError(f"[Attacker {self.client_id}] Invalid M dimension: {M}")
        
        # Step 1: Compute Laplacian of original graph
        # L = diag(A·1) - A
        degree_orig = adj_orig.sum(dim=1)
        L_orig = torch.diag(degree_orig) - adj_orig  # (M, M)
        
        # Step 2: SVD of original Laplacian
        # L = B Λ B^T
        try:
            U_orig, S_orig, Vh_orig = torch.linalg.svd(L_orig, full_matrices=True)
            B_orig = U_orig  # GFT basis (M, M)
        except Exception as e:
            # Fallback if SVD fails: return zeros in reduced dimension
            print(f"    [Attacker {self.client_id}] SVD failed: {e}, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Step 3: Compute GFT coefficient matrix
        # S = F · B where F ∈ R^{I×M}, B ∈ R^{M×M}
        S = torch.mm(feature_matrix, B_orig)  # (I, M)
        
        # Step 4: Compute Laplacian of reconstructed graph
        # L̂ = diag(Â·1) - Â
        # Note: adj_recon from VGAE is logits, convert to probabilities for Laplacian
        adj_recon_probs = torch.sigmoid(adj_recon)  # Convert logits to probabilities
        degree_recon = adj_recon_probs.sum(dim=1)
        L_recon = torch.diag(degree_recon) - adj_recon_probs  # (M, M)
        
        # Step 5: SVD of reconstructed Laplacian
        try:
            U_recon, S_recon, Vh_recon = torch.linalg.svd(L_recon, full_matrices=True)
            B_recon = U_recon  # New GFT basis (M, M)
        except Exception as e:
            # Fallback if SVD fails: return zeros in reduced dimension
            print(f"    [Attacker {self.client_id}] SVD of recon failed: {e}, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Step 6: Generate reconstructed feature matrix
        # F̂ = S · B̂^T where S ∈ R^{I×M}, B̂ ∈ R^{M×M}
        F_recon = torch.mm(S, B_recon.t())  # (I, M)
        
        # Step 7: Generate malicious update
        # Paper: "vectors w'_j(t) in F̂ are selected as malicious local models"
        # According to paper, we select a vector from F̂ as w'_j(t)
        # F̂ shape: (I, M) where I is number of benign updates, M is feature dimension
        
        # Check if F_recon is valid
        F_recon_rows = int(F_recon.shape[0])  # Convert to Python int
        print(f"    [Attacker {self.client_id}] F_recon.shape = ({F_recon_rows}, {F_recon.shape[1]})")
        if F_recon_rows == 0:
            # Empty feature matrix: return zeros
            print(f"    [Attacker {self.client_id}] F_recon is empty, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Select a vector from F̂ as the malicious update
        # Paper: "vectors w'_j(t) in F̂ are selected as malicious local models"
        # 
        # BALANCED STRATEGY: Filter out vectors with very low similarity (avoid orthogonal),
        # then select from valid candidates based on client_id for diversity.
        # This balances between avoiding optimization instability and maintaining attack effectiveness.
        # 
        # Compute benign aggregate (mean of benign feature matrix) as reference
        benign_aggregate = feature_matrix.mean(dim=0)  # (M,) - mean of all benign updates in reduced space
        benign_aggregate_flat = benign_aggregate.view(-1)  # Ensure 1D
        
        # Compute cosine similarity for each row in F_recon with benign aggregate
        F_recon_flat = F_recon.view(F_recon_rows, -1)  # (I, M) - ensure 2D
        benign_aggregate_expanded = benign_aggregate_flat.unsqueeze(0).expand(F_recon_rows, -1)  # (I, M)
        
        # Batch cosine similarity computation
        similarities_tensor = torch.cosine_similarity(
            F_recon_flat,
            benign_aggregate_expanded,
            dim=1
        )  # (I,) - similarity for each row
        
        # Filter strategy: exclude vectors with very low similarity (e.g., < 0.0) to avoid orthogonal vectors
        # But allow selection from a range of similarities for better attack effectiveness
        similarity_threshold = -0.5  # Allow negative similarities but filter out very orthogonal ones
        valid_mask = similarities_tensor >= similarity_threshold
        valid_indices = torch.where(valid_mask)[0].tolist()
        
        if len(valid_indices) > 0:
            # Select from valid candidates based on client_id for diversity
            # This ensures different attackers may choose different vectors while avoiding orthogonal ones
            select_idx = valid_indices[int(self.client_id) % len(valid_indices)]
            selected_sim = similarities_tensor[select_idx].item()
            selection_strategy = "filtered_by_similarity"
        else:
            # Fallback: if all vectors are too orthogonal, select the one with highest similarity
            select_idx = int(torch.argmax(similarities_tensor).item())
            selected_sim = similarities_tensor[select_idx].item()
            selection_strategy = "fallback_max_similarity"
        
        # Log selection details
        min_sim = similarities_tensor.min().item()
        mean_sim = similarities_tensor.mean().item()
        max_sim = similarities_tensor.max().item()
        valid_count = len(valid_indices) if len(valid_indices) > 0 else F_recon_rows
        print(f"    [Attacker {self.client_id}] Selected F_recon[{select_idx}] with similarity={selected_sim:.4f} (strategy: {selection_strategy})")
        print(f"    [Attacker {self.client_id}] Similarity stats: min={min_sim:.4f}, mean={mean_sim:.4f}, max={max_sim:.4f}, valid={valid_count}/{F_recon_rows} (threshold={similarity_threshold:.2f})")
        
        gsp_attack = F_recon[select_idx].clone()  # Select one row from F̂ as w'_j(t), clone to avoid view issues
        # Note: Selected from valid candidates (similarity >= threshold) based on client_id for diversity
        
        # Ensure gsp_attack is 1D tensor (not scalar)
        gsp_dim_count = int(gsp_attack.dim())  # Convert to Python int
        if gsp_dim_count == 0:
            # Scalar tensor: expand to 1D
            gsp_attack = gsp_attack.unsqueeze(0)
        elif gsp_dim_count > 1:
            # Multi-dimensional: flatten
            gsp_attack = gsp_attack.flatten()
        
        # Final check: ensure it's a 1D tensor with correct size
        if gsp_attack.numel() != M:
            # Size mismatch: create zeros with correct size
            print(f"    [Attacker {self.client_id}] GSP attack size mismatch: got {gsp_attack.numel()}, expected {M}, using zeros")
            gsp_attack = torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        return gsp_attack

    def camouflage_update(self, poisoned_update: torch.Tensor) -> torch.Tensor:
        """
        AugMP camouflage update using VGAE + GSP (data-agnostic; no local private data).
        
        AugMP clients are not assigned local training data and do not perform local training.
        The submitted update is constructed from benign clients' updates via VGAE+GSP.
        
        Paper Algorithm 1:
        1. Calculate A according to cosine similarity (eq. 8)
        2. Train VGAE to maximize L_loss (eq. 12), obtain optimal Â
        3. Use GSP module to obtain F̂, determine w'_j(t) based on F̂
        
        Args:
            poisoned_update: Zero update (AugMP clients don't train locally, so this is always zero)
        
        Returns:
            Camouflaged model update generated using VGAE+GSP
        """
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Reset feature indices for this session
        self.feature_indices = None
        
        # ============================================================
        # STEP 1: Prepare feature matrix F ∈ R^{I×M}
        # ============================================================
        # NO BETA SELECTION: Use ALL benign updates
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates available, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Move updates to GPU for processing
        benign_gpu = [u.to(self.device) for u in self.benign_updates]
        benign_stack = torch.stack([u.detach() for u in benign_gpu])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(benign_gpu, fix_indices=False)  # (I, M)
        # Clean up benign_stack and benign_gpu after feature reduction
        del benign_stack, benign_gpu
        torch.cuda.empty_cache()
        
        # Ensure reduced_benign has valid shape
        if reduced_benign is None or not isinstance(reduced_benign, torch.Tensor):
            raise ValueError(f"[Attacker {self.client_id}] reduced_benign is None or invalid")
        if len(reduced_benign.shape) < 2:
            raise ValueError(f"[Attacker {self.client_id}] reduced_benign must be 2D, got shape={reduced_benign.shape}")
        try:
            M = int(reduced_benign.shape[1])  # Convert to Python int
            I = int(reduced_benign.shape[0])  # Convert to Python int
        except (TypeError, ValueError) as e:
            raise ValueError(f"[Attacker {self.client_id}] Cannot convert reduced_benign shape to int: {e}, shape={reduced_benign.shape}")
        
        # ============================================================
        # STEP 2: Construct adjacency matrix A ∈ R^{M×M}
        # According to paper eq. (8): δ_{m,m'} = cosine_sim(w_m, w_m')
        # ============================================================
        adj_matrix = self._construct_graph(reduced_benign)  # (M, M)
        
        # ============================================================
        # STEP 3: Train VGAE to learn graph structure
        # Paper: "Train VGAE to maximize L_loss"
        # ============================================================
        adj_recon = self._train_vgae(adj_matrix, reduced_benign)  # Returns Â
        
        # ============================================================
        # STEP 4: GSP module to generate malicious update
        # Paper: "Use GSP module to obtain F̂, determine w'_j(t)"
        # ============================================================
        gsp_attack_reduced = self._gsp_generate_malicious(
            reduced_benign, adj_matrix, adj_recon, poisoned_update
        )
        
        # Clean up VGAE-related GPU tensors
        del reduced_benign, adj_matrix, adj_recon
        torch.cuda.empty_cache()
        
        # ============================================================
        # STEP 5: Expand GSP attack back to full dimension
        # Expand GSP attack from reduced dimension M back to full dimension.
        # Non-selected dimensions remain zero.
        # ============================================================
        # Create malicious_update on CPU to save GPU memory
        # poisoned_update is likely on CPU, but ensure we create on CPU
        if poisoned_update.device.type == 'cuda':
            malicious_update = torch.zeros_like(poisoned_update)
        else:
            malicious_update = torch.zeros_like(poisoned_update, device='cpu')
        total_dim = int(malicious_update.shape[0])  # Convert to Python int
        
        # _gsp_generate_malicious always returns a tensor (never None)
        # But check if it's valid
        if gsp_attack_reduced is not None and isinstance(gsp_attack_reduced, torch.Tensor):
            # Ensure gsp_attack_reduced is 1D tensor
            gsp_dim_count = int(gsp_attack_reduced.dim())  # Convert to Python int
            if gsp_dim_count == 0:
                # Scalar tensor: expand to 1D
                gsp_attack_reduced = gsp_attack_reduced.unsqueeze(0)
            elif gsp_dim_count > 1:
                # Multi-dimensional: flatten
                gsp_attack_reduced = gsp_attack_reduced.flatten()
            
            # Get dimension as Python int (not tensor)
            gsp_dim = int(gsp_attack_reduced.shape[0])
            
            if self.feature_indices is not None:
                # Dimension reduction was applied
                expected_dim = len(self.feature_indices)
                if gsp_dim == expected_dim:
                    # Correct dimension: expand back to full dimension
                    # Ensure gsp_attack_reduced is on CPU to match malicious_update
                    if gsp_attack_reduced.device.type == 'cuda':
                        gsp_attack_reduced = gsp_attack_reduced.cpu()
                    # Ensure feature_indices is on CPU for indexing
                    feature_indices_cpu = self.feature_indices.cpu() if self.feature_indices.device.type == 'cuda' else self.feature_indices
                    malicious_update[feature_indices_cpu] = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {expected_dim}, using zeros")
            else:
                # No dimension reduction: GSP attack should be full dimension
                if gsp_dim == total_dim:
                    # Correct dimension: use directly
                    # Ensure gsp_attack_reduced is on CPU to match malicious_update
                    if gsp_attack_reduced.device.type == 'cuda':
                        malicious_update = gsp_attack_reduced.cpu()
                    else:
                        malicious_update = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {total_dim}, using zeros")
        else:
            # GSP attack is None: malicious_update remains zeros
            print(f"    [Attacker {self.client_id}] GSP attack is None, using zeros")
        
        # ============================================================
        # STEP 5: (Removed - attackers don't perform local training)
        # Attackers are data-agnostic and don't have local data for training.
        # The attack is generated purely from VGAE+GSP using benign updates.
        # ============================================================
        
        # ============================================================
        # STEP 6: Optimize attack objective (NO BETA SELECTION)
        # ============================================================
        # Paper objective: maximize F(w'_g) subject to d(w'_j, w'_g) ≤ d_T
        # 
        # Correct interpretation:
        # - w'_g = w_g + Δ_g where Δ_g is the aggregated update (including attacker)
        # - Optimize loss of the aggregated global model: F(w_g + Δ_g)
        # - Constraint in UPDATE space: ||Δ_att - Δ_g|| ≤ d_T
        # 
        # NO BETA SELECTION: Server aggregates ALL benign clients; attacker does not control participant set.
        # All benign updates are used for aggregation and distance computation.
        # ============================================================
        
        # ============================================================
        # STEP 7: Optimize w'_j(t) to maximize F(w'_g(t))
        # According to paper Equation 12, we maximize F(w'_g(t)) subject to constraints
        # ============================================================
        # CRITICAL: Hard preconditions check before optimization
        # global_model_params is always required; proxy_loader is optional (when use_proxy_data=False, no data access)
        if self.global_model_params is None:
            error_msg = (
                f"[Attacker {self.client_id}] Missing global_model_params before optimization. Cannot proceed."
            )
            print(f"    {error_msg}")
            raise RuntimeError(error_msg)
        if self.proxy_loader is None:
            print(f"    [Attacker {self.client_id}] No proxy data (attacker_use_proxy_data=False); optimizing constraints only (no F(w'_g) term).")
        
        # Verify global_model_params dimension matches _flat_numel (LoRA mode requirement)
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            global_numel = self.global_model_params.numel()
            if global_numel != self._flat_numel:
                error_msg = (
                    f"[Attacker {self.client_id}] LoRA mode: global_model_params dimension mismatch:\n"
                    f"  - global_model_params.numel(): {global_numel}\n"
                    f"  - _flat_numel: {self._flat_numel}\n"
                    f"  - model.use_lora: {use_lora}\n"
                    f"  - model.get_flat_params().numel(): {self.model.get_flat_params().numel()}\n"
                    f"Check set_global_model_params() conversion."
                )
                print(f"    {error_msg}")
                raise RuntimeError(error_msg)
            else:
                print(f"    [Attacker {self.client_id}] LoRA dimension check passed: "
                      f"global_params={global_numel}, _flat_numel={self._flat_numel}")
        proxy_lr = self.proxy_step
        # Initialize optimization variable from GSP-generated malicious update
        # Paper: optimize w'_j(t) to maximize F(w'_g(t)) subject to constraints
        # Optimization starts from the exact w'_j(t) generated by GSP (no perturbation)
        proxy_param = malicious_update.clone().detach().to(self.device)
        proxy_param.requires_grad_(True)
        proxy_opt = optim.Adam([proxy_param], lr=proxy_lr)
        
        # Check dimension once before loop (performance optimization)
        proxy_param_flat = proxy_param.view(-1)
        dim_valid = int(proxy_param_flat.numel()) == self._flat_numel
        
        # CRITICAL: Ensure model is on GPU before optimization loop
        # Model must stay on GPU during the entire optimization loop to maintain computation graph
        target_device = torch.device('cuda:0') if self.device.type == 'cuda' else self.device
        if not self._model_on_gpu:
            self.model.to(target_device)
            self._ensure_model_on_device(self.model, target_device)
            self._model_on_gpu = True
        
        # ===== CRITICAL: Initialize functional cache for LoRA mode =====
        # This must be done before optimization loop starts
        use_lora = hasattr(self.model, 'use_lora') and self.model.use_lora
        if use_lora:
            self._init_functional_param_cache(target_device)
            # Ensure base_params and base_buffers are on GPU (cache them on GPU)
            # This avoids repeated device transfers during optimization loop
            if not all(p.device.type == target_device.type for p in self.base_params.values()):
                for name in self.base_params:
                    self.base_params[name] = self.base_params[name].to(target_device)
            if not all(b.device.type == target_device.type for b in self.base_buffers.values()):
                for name in self.base_buffers:
                    self.base_buffers[name] = self.base_buffers[name].to(target_device)
        # Store use_lora for use in optimization loop
        self._use_lora_in_optimization = use_lora
        # ===================================================================
        
        # ===== CRITICAL: Pre-transfer all updates to GPU to avoid frequent CPU-GPU transfers =====
        # This prevents computation graph breaks and improves performance
        # All updates will stay on GPU during the entire optimization loop
        # CRITICAL: All GPU versions MUST be on the same device (target_device) to avoid any device transfers
        if len(self.benign_updates) > 0:
            self.benign_updates_gpu = [u.to(target_device) for u in self.benign_updates]
            # Verify device consistency to ensure no device transfers in optimization loop
            assert all(u.device == target_device for u in self.benign_updates_gpu), \
                f"[Attacker {self.client_id}] CRITICAL: All benign_updates_gpu must be on {target_device}"
        else:
            self.benign_updates_gpu = []
        if len(self.other_attacker_updates) > 0:
            self.other_attacker_updates_gpu = [u.to(target_device) for u in self.other_attacker_updates]
            # Verify device consistency to ensure no device transfers in optimization loop
            assert all(u.device == target_device for u in self.other_attacker_updates_gpu), \
                f"[Attacker {self.client_id}] CRITICAL: All other_attacker_updates_gpu must be on {target_device}"
        else:
            self.other_attacker_updates_gpu = []
        # ============================================================================================
        
        # ===== CRITICAL: Compute GLOBAL REFERENCE update for constraint judgement =====
        # This is the stable reference point for ALL constraint calculations (distance and similarity)
        # Key properties:
        # 1. Includes benign + other attackers (NOT current attacker) - matches server aggregation context
        # 2. Does NOT depend on proxy_param (independent of optimization variable)
        # 3. Computed ONCE before loop (constant reference throughout optimization)
        # 4. Same reference used for: constraints, statistics, initial/final checks
        # 
        # Why global reference (instead of benign-only)?
        # - Makes constraints more realistic and aligned with server's final aggregation
        # - Other attackers' updates affect the final aggregation, so should be considered in constraints
        # - Ensures optimization objective (global_loss) and constraints are consistent
        # =====================================================================================
        # Compute global reference on CPU (for statistics and final check)
        global_ref_cpu = self._aggregate_global_reference(
            self.benign_updates,
            other_attacker_updates=self.other_attacker_updates if hasattr(self, 'other_attacker_updates') else None,
            other_attacker_updates_gpu=None,  # Use CPU versions for initial reference
            device=torch.device('cpu')
        )
        # Compute global reference on GPU (for optimization loop constraints)
        global_ref_gpu = global_ref_cpu.to(target_device).detach()  # Detached constant
        global_ref_gpu.requires_grad_(False)  # Ensure no gradient tracking
        # ============================================================================================
        
        
        # OPTIMIZATION 5: Cache Lagrangian multipliers on GPU before loop
        # Ensure multipliers are on correct device to avoid repeated conversions
        if self.use_lagrangian_dual and self.lambda_dist is not None:
            if isinstance(self.lambda_dist, torch.Tensor):
                if not self._device_matches(self.lambda_dist.device, target_device):
                    self.lambda_dist = self.lambda_dist.to(target_device)
            else:
                self.lambda_dist = torch.tensor(self.lambda_dist, device=target_device)
        
        # Move similarity multipliers to target device
        if self.use_cosine_similarity_constraint:
            if self.lambda_sim_low is not None:
                if isinstance(self.lambda_sim_low, torch.Tensor):
                    if not self._device_matches(self.lambda_sim_low.device, target_device):
                        self.lambda_sim_low = self.lambda_sim_low.to(target_device)
                else:
                    self.lambda_sim_low = torch.tensor(self.lambda_sim_low, device=target_device)
            
            if self.lambda_sim_up is not None:
                if isinstance(self.lambda_sim_up, torch.Tensor):
                    if not self._device_matches(self.lambda_sim_up.device, target_device):
                        self.lambda_sim_up = self.lambda_sim_up.to(target_device)
                else:
                    self.lambda_sim_up = torch.tensor(self.lambda_sim_up, device=target_device)
        
        # ============================================================
        # Print initial optimization state
        # ============================================================
        print(f"    [Attacker {self.client_id}] Preparing optimization: "
              f"proxy_param.shape={proxy_param.shape}, proxy_param.numel()={proxy_param.numel()}, "
              f"_flat_numel={self._flat_numel}, use_lora={use_lora}")
        
        # ===== CRITICAL: Compute GLOBAL REFERENCE update for constraint judgement =====
        # This is the stable reference point for ALL constraint calculations (distance and similarity)
        # Key properties:
        # 1. Includes benign + other attackers (NOT current attacker) - matches server aggregation context
        # 2. Does NOT depend on proxy_param (independent of optimization variable)
        # 3. Computed ONCE before loop (constant reference throughout optimization)
        # 4. Same reference used for: constraints, statistics, initial/final checks
        # 
        # Why global reference (instead of benign-only)?
        # - Makes constraints more realistic and aligned with server's final aggregation
        # - Other attackers' updates affect the final aggregation, so should be considered in constraints
        # - Ensures optimization objective (global_loss) and constraints are consistent
        # =====================================================================================
        # Compute global reference on CPU (for statistics and final check)
        global_ref_cpu = self._aggregate_global_reference(
            self.benign_updates,
            other_attacker_updates=self.other_attacker_updates if hasattr(self, 'other_attacker_updates') else None,
            other_attacker_updates_gpu=None,  # Use CPU versions for initial reference
            device=torch.device('cpu')
        )
        # Compute global reference on GPU (for optimization loop constraints)
        global_ref_gpu = global_ref_cpu.to(target_device).detach()  # Detached constant
        global_ref_gpu.requires_grad_(False)  # Ensure no gradient tracking
        # ============================================================================================
        
        # Cache distance statistics before loop (for automatic d_T calculation)
        # CRITICAL: Use detach() to avoid creating computation graph (statistics are for threshold calculation only)
        # NOTE: Statistics use ONLY benign clients for threshold calculation (other attackers excluded)
        # This ensures clean baseline for threshold setting.
        # Using global_ref_cpu which excludes current attacker (consistent with optimization loop)
        if len(self.benign_updates) > 0:
            cached_benign_dist_stats = self._compute_benign_distance_statistics(
                self.benign_updates,
                benign_ref_update=global_ref_cpu,  # Benign-only aggregation (excludes all attackers)
                current_attacker_update=None  # Exclude current attacker for stable statistics
            )
        else:
            cached_benign_dist_stats = None
        
        # Pre-convert dist_bound to scalar once before loop (avoid repeated conversion)
        # If dist_bound is None, use benign statistics: max (largest distance among benign clients)
        # Reason: Using max ensures all benign clients are within the threshold, providing maximum flexibility
        if self.dist_bound is not None:
            dist_bound_val = float(self.dist_bound) if isinstance(self.dist_bound, torch.Tensor) else self.dist_bound
            dist_bound_source = "manual"
        else:
            # Use benign statistics when dist_bound is None
            if cached_benign_dist_stats is not None:
                # Use max (largest distance among benign clients) - ensures all benign clients are within threshold
                dist_bound_val = float(cached_benign_dist_stats['max'].item())
                dist_bound_source = "benign_max"
            else:
                dist_bound_val = None
                dist_bound_source = "none"
        
        # Store effective dist_bound for use in final constraint check
        self._effective_dist_bound = dist_bound_val
        
        # Cache cosine similarity statistics before loop (avoid recomputation each step)
        # CRITICAL: Use detach() to avoid creating computation graph (statistics are for threshold calculation only)
        # Compute this BEFORE initial state printing so we can include similarity info
        # NOTE: Use ONLY benign clients for statistics (excludes current attacker and other attackers)
        # This ensures clean baseline for threshold calculation
        if self.use_cosine_similarity_constraint and len(self.benign_updates) > 0:
            if getattr(self, 'use_pairwise_similarity_in_constraint', False):
                # Pairwise: align with server pairwise mode; bounds from benign-vs-benign mean similarities
                cached_benign_sim_stats = self._compute_benign_pairwise_similarity_statistics(self.benign_updates)
            else:
                # Aggregation-based: statistics relative to benign-only aggregation
                cached_benign_sim_stats = self._compute_benign_cosine_similarity_statistics(
                    self.benign_updates,
                    aggregated_ref=global_ref_cpu  # Benign-only aggregation (excludes all attackers)
                )
        else:
            cached_benign_sim_stats = None
        
        # ===== OPTIMIZATION: Pre-compute aggregation reference for similarity constraint =====
        # CRITICAL: Use ONLY benign clients aggregation (excludes current attacker and other attackers)
        # This is a constant (doesn't depend on proxy_param), so compute once before loop
        # Similar to global_ref_gpu, this avoids re-aggregating in each iteration
        # CRITICAL: Compute this ONCE before loop, then reuse in optimization loop and initial state printing
        if self.use_cosine_similarity_constraint and len(self.benign_updates) > 0:
            if hasattr(self, 'benign_updates_gpu') and self.benign_updates_gpu is not None and len(self.benign_updates_gpu) > 0:
                # Use pre-transferred GPU versions (already on target_device)
                aggregation_ref_gpu_sim = global_ref_gpu  # Benign-only aggregation (excludes all attackers)
            else:
                # Fallback: transfer CPU versions to GPU (shouldn't happen if preprocessing is correct)
                aggregation_ref_gpu_sim = global_ref_cpu.to(target_device).detach()
            aggregation_ref_gpu_sim.requires_grad_(False)  # Ensure no gradient tracking
        else:
            aggregation_ref_gpu_sim = None
        # ============================================================================================
        
        # Print initial state with dist_bound value (whether manual or auto-computed)
        if dist_bound_val is not None:
            try:
                # CRITICAL: Use detach() for initial calculations to avoid interfering with optimization loop's computation graph
                # Use GPU versions if available (will be created right after this)
                with torch.no_grad():
                    # CRITICAL: Compute distance to aggregation EXCLUDING current attacker to avoid circular dependency
                    initial_global_ref_gpu, _, _ = self._aggregate_update_no_beta(
                        proxy_param.detach(),
                        benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                        other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                        include_current_attacker=False  # CRITICAL: Exclude current attacker to avoid circular dependency
                    )
                    initial_dist_att_to_global = torch.norm(proxy_param.detach().view(-1) - initial_global_ref_gpu.view(-1))
                    initial_dist = initial_dist_att_to_global.item()
                    initial_g_dist = initial_dist - dist_bound_val
                    initial_lambda_dist = self.lambda_dist.item() if isinstance(self.lambda_dist, torch.Tensor) else self.lambda_dist if self.lambda_dist is not None else 0.0
                    # MODIFIED: Use include_current_attacker=True for initial_loss display
                    # This shows the actual global_loss that will be optimized (consistent with optimization loop)
                    initial_loss = self._compute_global_loss(
                        proxy_param.detach(),
                        benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                        other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                        include_current_attacker=True  # MODIFIED: Include current attacker (for display, matches optimization)
                    ).item()
                    
                    # Compute initial similarity metrics (always compute, not just when constraint is enabled)
                    initial_sim_info = ""
                    # CRITICAL: Always compute similarity for logging, even if constraint is not enabled
                    if len(self.benign_updates) > 0:
                        # CRITICAL: Compute initial similarity to aggregation EXCLUDING current attacker
                        # Use aggregation without current attacker to avoid circular dependency
                        # This matches the optimization loop's reference point
                        initial_aggregation_without_current, _, _ = self._aggregate_update_no_beta(
                            proxy_param.detach(),
                            benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                            other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                            include_current_attacker=False  # CRITICAL: Exclude current attacker to avoid circular dependency
                        )
                        proxy_param_flat = proxy_param.detach().view(-1)
                        initial_aggregation_flat = initial_aggregation_without_current.view(-1)
                        initial_sim_att_to_benign = torch.cosine_similarity(
                            proxy_param_flat.unsqueeze(0),
                            initial_aggregation_flat.unsqueeze(0),
                            dim=1
                        ).squeeze(0).item()
                        
                        # Compute similarity bounds if statistics are available
                        if cached_benign_sim_stats is not None:
                            benign_sim_mean = cached_benign_sim_stats['mean']
                            benign_sim_std = cached_benign_sim_stats['std']
                            benign_sim_min = cached_benign_sim_stats['min']
                            benign_sim_max = cached_benign_sim_stats['max']
                            
                            # Bounds: manual sim_bound_low/up if set; else lower=benign min, upper=benign mean
                            if self.sim_bound_low is not None:
                                actual_bound_low = max(-1.0, min(1.0, float(self.sim_bound_low)))
                            else:
                                actual_bound_low = torch.clamp(benign_sim_min, min=-1.0, max=1.0).item()
                            if self.sim_bound_up is not None:
                                actual_bound_up = max(-1.0, min(1.0, float(self.sim_bound_up)))
                            else:
                                actual_bound_up = torch.clamp(benign_sim_mean, min=-1.0, max=1.0).item()
                            bounds_src = "manual" if (self.sim_bound_low is not None or self.sim_bound_up is not None) else "benign_min_mean"
                            print(f"    [Attacker {self.client_id}] Similarity stats (relative to aggregation without current attacker): "
                                  f"mean={benign_sim_mean.item():.4f}, std={benign_sim_std.item():.4f}, "
                                  f"min={benign_sim_min.item():.4f}, max={benign_sim_max.item():.4f}, "
                                  f"bounds=[{bounds_src}]=[{actual_bound_low:.4f}, {actual_bound_up:.4f}] (upper=mean)")
                            
                            if self.use_cosine_similarity_constraint:
                                initial_sim_bound_low = actual_bound_low
                                initial_sim_bound_up = actual_bound_up
                                
                                # Compute constraint violations
                                initial_g_sim_low = initial_sim_bound_low - initial_sim_att_to_benign
                                initial_g_sim_up = initial_sim_att_to_benign - initial_sim_bound_up
                                
                                # Get initial lambda values
                                initial_lambda_sim_low = self.lambda_sim_low.item() if isinstance(self.lambda_sim_low, torch.Tensor) else self.lambda_sim_low if self.lambda_sim_low is not None else 0.0
                                initial_lambda_sim_up = self.lambda_sim_up.item() if isinstance(self.lambda_sim_up, torch.Tensor) else self.lambda_sim_up if self.lambda_sim_up is not None else 0.0
                                
                                initial_sim_info = f", initial_sim={initial_sim_att_to_benign:.4f}∈[{initial_sim_bound_low:.4f},{initial_sim_bound_up:.4f}], " \
                                                  f"g_sim_low={initial_g_sim_low:.4f}, g_sim_up={initial_g_sim_up:.4f}, " \
                                                  f"lambda_sim_low={initial_lambda_sim_low:.4f}, lambda_sim_up={initial_lambda_sim_up:.4f}"
                            else:
                                # Similarity constraint not enabled, but still show similarity value and bounds
                                if cached_benign_sim_stats is not None:
                                    if self.sim_bound_low is not None:
                                        initial_sim_bound_low = max(-1.0, min(1.0, float(self.sim_bound_low)))
                                    else:
                                        initial_sim_bound_low = torch.clamp(cached_benign_sim_stats['min'], min=-1.0, max=1.0).item()
                                    if self.sim_bound_up is not None:
                                        initial_sim_bound_up = max(-1.0, min(1.0, float(self.sim_bound_up)))
                                    else:
                                        initial_sim_bound_up = torch.clamp(cached_benign_sim_stats['mean'], min=-1.0, max=1.0).item()
                                    initial_sim_info = f", initial_sim={initial_sim_att_to_benign:.4f}∈[{initial_sim_bound_low:.4f},{initial_sim_bound_up:.4f}] (constraint disabled)"
                                else:
                                    initial_sim_info = f", initial_sim={initial_sim_att_to_benign:.4f}"
                        else:
                            # No similarity statistics available, just show the value
                            initial_sim_info = f", initial_sim={initial_sim_att_to_benign:.4f}"
                    
                dist_bound_info = f"dist_bound={dist_bound_val:.4f} ({dist_bound_source})" if dist_bound_source != "none" else "dist_bound=None"
                print(f"    [Attacker {self.client_id}] Starting optimization (UPDATE space): "
                      f"initial_dist={initial_dist:.4f}, {dist_bound_info}, g_dist={initial_g_dist:.4f}, "
                      f"lambda_dist={initial_lambda_dist:.4f}, loss={initial_loss:.4f}{initial_sim_info}, steps={self.proxy_steps}")
            except Exception as e:
                print(f"    [Attacker {self.client_id}] ERROR computing initial state: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Early stopping variables: track constraint satisfaction stability
        constraint_satisfied_steps = 0
        constraint_stability_steps = self.early_stop_constraint_stability_steps  # Stop after N consecutive steps satisfying constraint
        prev_dist_val = None
        
        for step in range(self.proxy_steps):
            # ============================================================
            # CRITICAL: Zero gradients using set_to_none=True for efficiency
            # ============================================================
            proxy_opt.zero_grad(set_to_none=True)
            
            # OPTIMIZATION 4: Reduce device check frequency (every 5 steps instead of every step)
            # Model should remain on GPU during optimization loop, so full check is rarely needed
            if step % 5 == 0:  # Check every 5 steps
                # Quick check using a sample parameter
                sample_param = next(iter(self.model.parameters()), None)
                if sample_param is not None and not self._device_matches(sample_param.device, target_device):
                    # If device mismatch detected, perform full check
                    self._ensure_model_on_device(self.model, target_device)
            # Note: Full device check removed from here for performance (checked before loop and every 5 steps)
            
            # ============================================================
            # Compute base objective function F(w'_g(t)) according to paper Formula (3)
            # ============================================================
            # ============================================================
            # Compute loss of AGGREGATED global model: F(w_g + Δ_g)
            # 
            # MODIFICATION: Use include_current_attacker=True for global_loss calculation
            # This allows global_loss to depend on proxy_param, enabling gradient-based optimization.
            # 
            # Strategy: Separate reference points for objective and constraints
            # - Objective (global_loss): include_current_attacker=True
            #   → F(w'_g) where w'_g = aggregate(proxy_param, benign, others)
            #   → Gradient: ∇_proxy_param F(w'_g) ≠ 0 (can optimize)
            # 
            # - Constraints: include_current_attacker=False (computed separately below)
            #   → Distance: ||proxy_param - aggregate(benign, others)||
            #   → Stable reference point, avoids circular dependency
            # 
            # Why this works:
            # - The difference between the two reference points is proxy_param's weighted contribution
            # - If attacker weight is small (w_att << 1), the difference is negligible
            # - Optimizing F(w'_g) and constraining ||proxy_param - aggregate(benign, others)|| are consistent
            # ============================================================
            # CRITICAL: Use GPU versions to avoid device transfers and maintain computation graph
            global_loss = self._compute_global_loss(
                proxy_param,
                benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                include_current_attacker=True  # MODIFIED: Include current attacker to enable gradient-based optimization
            )
            
            # ============================================================
            # Build optimization objective: choose mechanism based on whether using Lagrangian Dual
            # ============================================================
            
            if self.use_lagrangian_dual and self.lambda_dist is not None:
                # ========== Use Lagrangian Dual mechanism (paper eq:lagrangian and eq:wprime_sub) ==========
                
                # OPTIMIZATION 5: Use cached multipliers directly (already on correct device)
                lambda_dist_tensor = self.lambda_dist  # Direct use, no conversion needed
                
                # ============================================================
                # Distance Constraint: g_dist(x) = dist(Δ_att, Δ_g) - dist_bound <= 0
                # ============================================================
                # CRITICAL: Compute global reference EXCLUDING current attacker to avoid circular dependency
                # When include_current_attacker=True, global aggregation includes proxy_param, creating:
                #   global_agg = f(proxy_param, ...)
                #   distance = ||proxy_param - global_agg||
                # This creates a circular dependency where updating proxy_param changes the distance definition.
                # Using False ensures stable optimization: distance measures to aggregation without current attacker.
                global_ref_gpu, _, _ = self._aggregate_update_no_beta(
                    proxy_param,
                    benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                    other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                    include_current_attacker=False  # CRITICAL: Exclude current attacker to avoid circular dependency
                )
                dist_att_to_global = torch.norm(proxy_param.view(-1) - global_ref_gpu.view(-1))
                
                # ============================================================
                # Standard Lagrangian Dual formulation (NO ReLU in Lagrangian term)
                # ============================================================
                # Distance constraint: g_dist(x) = dist(Δ_att, Δ_g) - dist_bound ≤ 0
                # 
                # Standard Lagrangian: dist_lagr_term = λ_dist * g_dist
                # - When dist < dist_bound (satisfied): g_dist < 0, term < 0 (negative, "rewards" satisfaction)
                # - When dist > dist_bound (violated):  g_dist > 0, term > 0 (positive, penalizes violation)
                #
                # Why Standard Lagrangian (NOT Augmented):
                # - Provides directional guidance even when constraint is satisfied (g_dist < 0)
                # - Negative penalty (reward) when satisfied helps prevent deviation from constraint boundary
                # - Prevents optimization from moving too far from constraint when constraint is satisfied
                # - NO quadratic penalty term (this is standard Lagrangian, NOT Augmented Lagrangian)
                # ============================================================
                # Use pre-converted dist_bound_val (converted before loop to avoid repeated conversion)
                g_dist = dist_att_to_global - dist_bound_val if dist_bound_val is not None else torch.tensor(0.0, device=target_device)
                
                # Standard Lagrangian distance term (NO ReLU here)
                # dist_lagr_term = λ_dist * g_dist
                dist_lagr_term = lambda_dist_tensor * g_dist
                
                # ============================================================
                # Cosine Similarity Constraints (TWO-SIDED with TWO multipliers)
                # ============================================================
                # Lower bound constraint: g_sim_low = sim_bound_low - sim_att <= 0
                # Upper bound constraint: g_sim_up = sim_att - sim_bound_up <= 0
                # ============================================================
                # Initialize similarity constraint variables (will be zero if constraint disabled)
                sim_lagr_term = torch.tensor(0.0, device=target_device)
                g_sim_low = torch.tensor(0.0, device=target_device)
                g_sim_up = torch.tensor(0.0, device=target_device)
                lambda_sim_low_tensor = torch.tensor(0.0, device=target_device)
                lambda_sim_up_tensor = torch.tensor(0.0, device=target_device)
                
                if self.use_cosine_similarity_constraint and cached_benign_sim_stats is not None:
                    proxy_param_flat = proxy_param.view(-1)
                    if getattr(self, 'use_pairwise_similarity_in_constraint', False):
                        # Pairwise: sim_att = mean over all other clients of cos(Δ_att, Δ_j) (aligns with server pairwise)
                        if hasattr(self, 'benign_updates_gpu') and self.benign_updates_gpu is not None and len(self.benign_updates_gpu) > 0:
                            benign_updates_for_ref = self.benign_updates_gpu
                        else:
                            benign_updates_for_ref = [u.to(target_device) for u in self.benign_updates] if len(self.benign_updates) > 0 else []
                        other_att = getattr(self, 'other_attacker_updates_gpu', None) or []
                        all_others = list(benign_updates_for_ref) + list(other_att)
                        if len(all_others) == 0:
                            sim_att_to_benign = torch.tensor(1.0, device=target_device, dtype=proxy_param_flat.dtype)
                        else:
                            sims_list = []
                            for u in all_others:
                                u_flat = u.view(-1).to(target_device) if u.device != target_device else u.view(-1)
                                sims_list.append(torch.cosine_similarity(
                                    proxy_param_flat.unsqueeze(0), u_flat.unsqueeze(0), dim=1
                                ).squeeze(0))
                            sim_att_to_benign = torch.stack(sims_list).mean()
                    else:
                        # Aggregation-based: sim_att = cos(Δ_att, Δ_agg) with Δ_agg excluding current attacker
                        if hasattr(self, 'benign_updates_gpu') and self.benign_updates_gpu is not None and len(self.benign_updates_gpu) > 0:
                            benign_updates_for_ref = self.benign_updates_gpu
                        else:
                            benign_updates_for_ref = [u.to(target_device) for u in self.benign_updates] if len(self.benign_updates) > 0 else []
                        aggregation_without_current, _, _ = self._aggregate_update_no_beta(
                            proxy_param,
                            benign_updates_gpu=benign_updates_for_ref,
                            other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                            include_current_attacker=False
                        )
                        aggregation_without_current_flat = aggregation_without_current.view(-1)
                        sim_att_to_benign = torch.cosine_similarity(
                            proxy_param_flat.unsqueeze(0),
                            aggregation_without_current_flat.unsqueeze(0),
                            dim=1
                        ).squeeze(0)
                    
                    # Bounds = benign similarity min/max (attackers must stay within benign range)
                    # Bounds: use manual sim_bound_low/up if set, else benign min/max
                    if self.sim_bound_low is not None:
                        sim_bound_low = torch.clamp(torch.tensor(float(self.sim_bound_low), device=target_device, dtype=sim_att_to_benign.dtype), min=-1.0, max=1.0)
                    else:
                        sim_bound_low = torch.clamp(cached_benign_sim_stats['min'].to(target_device), min=-1.0, max=1.0)
                    if self.sim_bound_up is not None:
                        sim_bound_up = torch.clamp(torch.tensor(float(self.sim_bound_up), device=target_device, dtype=sim_att_to_benign.dtype), min=-1.0, max=1.0)
                    else:
                        sim_bound_up = torch.clamp(cached_benign_sim_stats['mean'].to(target_device), min=-1.0, max=1.0)
                    
                    # Two-sided constraints (NO ReLU in Lagrangian terms)
                    # g_sim_low = sim_bound_low - sim_att_to_benign <= 0
                    # g_sim_up = sim_att_to_benign - sim_bound_up <= 0
                    g_sim_low = sim_bound_low - sim_att_to_benign
                    g_sim_up = sim_att_to_benign - sim_bound_up
                    
                    # Two independent Lagrangian multipliers
                    lambda_sim_low_tensor = self.lambda_sim_low
                    lambda_sim_up_tensor = self.lambda_sim_up
                    
                # Similarity Lagrangian terms (NO ReLU here, standard Lagrangian)
                # sim_lagr_term = λ_sim_low * g_sim_low + λ_sim_up * g_sim_up
                # If use_cosine_similarity_constraint=False, all terms are zero, so sim_lagr_term = 0
                sim_lagr_term = lambda_sim_low_tensor * g_sim_low + lambda_sim_up_tensor * g_sim_up
                
                
                # ============================================================
                # Build Lagrangian objective function (paper formula eq:wprime_sub)
                # ============================================================
                # Paper: maximize F(w'_g(t)) subject to constraints
                # Standard Lagrangian: L = F(w'_g) - λ_dist * g_dist - λ_sim_low * g_sim_low - λ_sim_up * g_sim_up
                # Converting to minimization: minimize -L = -F(w'_g) + λ_dist * g_dist + λ_sim_low * g_sim_low + λ_sim_up * g_sim_up
                # 
                # Three constraints:
                # 1. Distance: g_dist = dist_att_to_agg - dist_bound <= 0
                # 2. Sim lower: g_sim_low = sim_bound_low - sim_att_to_agg <= 0
                # 3. Sim upper: g_sim_up = sim_att_to_agg - sim_bound_up <= 0
                # 
                # When constraint satisfied (g <= 0): Lagrangian term < 0 (rewards satisfaction)
                # When constraint violated (g > 0): Lagrangian term > 0 (penalizes violation)
                # ============================================================
                # Build (Augmented) Lagrangian objective
                #
                # Standard Lagrangian:
                #   minimize  -F(w'_g) + Σ_i λ_i g_i
                #
                # Standard Augmented Lagrangian (ALM):
                #   minimize  -F(w'_g) + Σ_i [ λ_i g_i + (ρ_i/2) g_i^2 ]
                #
                # NOTE:
                # - We keep the same reference-point separation:
                #   objective global_loss uses include_current_attacker=True,
                #   constraints g_i use include_current_attacker=False (stable reference).
                # ============================================================
                if self.use_augmented_lagrangian:
                    # Keep penalty parameters on the same device
                    rho_dist = self.rho_dist.to(target_device) if isinstance(self.rho_dist, torch.Tensor) else torch.tensor(float(self.rho_dist or 0.0), device=target_device)
                    aug_term = (rho_dist / 2.0) * (g_dist ** 2)

                    # Similarity quadratic penalties (two-sided, two ρ's)
                    if self.use_cosine_similarity_constraint:
                        rho_sim_low = self.rho_sim_low.to(target_device) if isinstance(self.rho_sim_low, torch.Tensor) else torch.tensor(float(self.rho_sim_low or 0.0), device=target_device)
                        rho_sim_up = self.rho_sim_up.to(target_device) if isinstance(self.rho_sim_up, torch.Tensor) else torch.tensor(float(self.rho_sim_up or 0.0), device=target_device)
                        aug_term = aug_term + (rho_sim_low / 2.0) * (g_sim_low ** 2) + (rho_sim_up / 2.0) * (g_sim_up ** 2)

                    lagrangian_objective = -global_loss + dist_lagr_term + sim_lagr_term + aug_term
                else:
                    lagrangian_objective = -global_loss + dist_lagr_term + sim_lagr_term
                
                # ============================================================
                # Compute constraint violations (for dual ascent updates, using ReLU)
                # ============================================================
                # Distance constraint violation (for updating λ_dist)
                # CRITICAL: Check dist_bound_val (effective bound) not self.dist_bound
                has_dist_bound = (dist_bound_val is not None)
                constraint_dist_violation = F.relu(g_dist) if has_dist_bound else torch.tensor(0.0, device=target_device)
                
            else:
                # ========== Use hard constraint mechanism (Lagrangian disabled) ==========
                # Objective: maximize global_loss => minimize -global_loss
                lagrangian_objective = -global_loss
                
                # Compute constraint violations (for logging only)
                # Use real distance calculation according to paper Constraint (4b)
                # ============================================================
                # Distance Constraint: d(Δ_att, Δ_g) ≤ dist_bound in UPDATE space
                # ============================================================
                dist_bound_val_hc = float(self.dist_bound) if isinstance(self.dist_bound, torch.Tensor) else self.dist_bound if self.dist_bound is not None else None
                if dist_bound_val_hc is not None:
                    # CRITICAL: Use GPU versions even in hard constraint mode to avoid device transfers
                    # CRITICAL: Exclude current attacker to avoid circular dependency in optimization
                    dist_att_to_agg, _ = self._compute_distance_update_space(
                        proxy_param,
                        benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                        other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                        include_current_attacker=False  # CRITICAL: Exclude current attacker to avoid circular dependency
                    )
                    constraint_dist_violation = F.relu(dist_att_to_agg - dist_bound_val_hc)
                else:
                    dist_att_to_agg = torch.tensor(0.0, device=target_device)
                    constraint_dist_violation = torch.tensor(0.0, device=target_device)
                
            
            # ============================================================
            # CRITICAL: Compute gradients using torch.autograd.grad (NO backward())
            # ============================================================
            # PROHIBITED: Cannot use backward() twice on the same graph
            # Solution: Use autograd.grad() once, manually set proxy_param.grad
            # If we need to verify gradient link (every 5 steps in LoRA mode), retain graph for second check
            need_check = bool(self._use_lora_in_optimization and (step % 5 == 0))
            
            try:
                grad = torch.autograd.grad(
                    lagrangian_objective,
                    proxy_param,
                    retain_graph=need_check,   # <-- keep graph only when we will do an extra check
                    allow_unused=False,
                    create_graph=False
                )[0]
                proxy_param.grad = grad
            except RuntimeError as e:
                if "second time" in str(e) or "already been freed" in str(e):
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL: Graph already used - double backward detected at step {step}. "
                        f"Check for multiple backward/grad passes on the same graph."
                    ) from e
                raise
            
            # ============================================================
            # Hard acceptance criteria for LoRA mode
            # ============================================================
            if self._use_lora_in_optimization:
                if proxy_param.grad is None:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL at step {step}: proxy_param.grad is None in LoRA mode."
                    )
                grad_norm = proxy_param.grad.norm().item()
                if grad_norm < 1e-8:
                    raise RuntimeError(
                        f"[Attacker {self.client_id}] FATAL at step {step}: Gradient norm too small ({grad_norm:.2e})."
                    )

                # Additional gradient-link verification (only when need_check=True)
                # MODIFIED: With include_current_attacker=True, global_loss DOES depend on proxy_param,
                # so its gradient w.r.t. proxy_param should be non-zero (enabling optimization).
                # lagrangian_objective SHOULD also have gradient from both global_loss and constraint terms.
                # We verify lagrangian_objective's gradient to ensure the full computation graph is intact.
                if need_check:
                    try:
                        # Verify lagrangian_objective has gradient (this should always be true)
                        g_lagr = torch.autograd.grad(
                            lagrangian_objective,
                            proxy_param,
                            retain_graph=False,   # <-- free graph after verification
                            allow_unused=False,   # Should not be unused - lagrangian_objective depends on proxy_param
                            create_graph=False
                        )[0]
                        if g_lagr is None:
                            raise RuntimeError(
                                f"[Attacker {self.client_id}] Gradient verification failed at step {step}: "
                                f"lagrangian_objective gradient is None (computation graph broken)."
                            )
                        grad_norm_lagr = g_lagr.norm().item()
                        if grad_norm_lagr < 1e-10:
                            raise RuntimeError(
                                f"[Attacker {self.client_id}] Gradient verification failed at step {step}: "
                                f"lagrangian_objective gradient norm too small ({grad_norm_lagr:.2e})."
                            )
                        
                        # Optional: Also check global_loss gradient (for information, not error)
                        # MODIFIED: With include_current_attacker=True, global_loss gradient should be non-zero
                        # This verifies that the optimization objective is properly connected to proxy_param
                        try:
                            g_global = torch.autograd.grad(
                                global_loss,
                                proxy_param,
                                retain_graph=False,
                                allow_unused=True,  # Allow unused for safety, but should not be unused
                                create_graph=False
                            )[0]
                            if g_global is not None:
                                grad_norm_global = g_global.norm().item()
                                if grad_norm_global < 1e-10:
                                    print(f"    [Attacker {self.client_id}] Warning at step {step}: "
                                          f"global_loss gradient is very small ({grad_norm_global:.2e}), "
                                          f"may indicate optimization issue.")
                                else:
                                    print(f"    [Attacker {self.client_id}] Info at step {step}: "
                                          f"global_loss gradient norm: {grad_norm_global:.4f} (non-zero, good)")
                        except:
                            # Ignore errors in optional check
                            pass
                            
                    except RuntimeError as e:
                        # Re-raise RuntimeError (e.g., graph already freed)
                        raise
                    except Exception as e:
                        # Other errors indicate a problem
                        raise RuntimeError(
                            f"[Attacker {self.client_id}] Gradient verification failed at step {step}: {e}"
                        ) from e
            # ===========================================================
            
            # Gradient clipping for proxy parameter update (separate from benign client training)
            torch.nn.utils.clip_grad_norm_([proxy_param], max_norm=self.proxy_grad_clip_norm)
            
            proxy_opt.step()
            
            # ============================================================
            # CRITICAL: Recompute distance and similarity AFTER parameter update
            # This ensures Early stopping check uses the actual current state
            # ============================================================
            # After proxy_opt.step(), proxy_param has been updated, so we need to recompute
            # distance and similarity using the updated parameter values for accurate constraint checking
            if dist_bound_val is not None:
                # Recompute global reference using updated proxy_param
                # Note: global_ref_gpu may be slightly different if other_attackers are also optimizing,
                # but in practice they are fixed during this attacker's optimization
                global_ref_gpu_after_step, _, _ = self._aggregate_update_no_beta(
                    proxy_param,  # Updated value after proxy_opt.step()
                    benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                    other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                    include_current_attacker=False  # CRITICAL: Exclude current attacker (consistent with optimization)
                )
                dist_att_to_global_after_step = torch.norm(proxy_param.view(-1) - global_ref_gpu_after_step.view(-1))
                dist_att_val_after_step = dist_att_to_global_after_step.item()
            else:
                dist_att_val_after_step = 0.0
                global_ref_gpu_after_step = None
            
            # Recompute similarity after step if similarity constraint is enabled
            sim_att_val_after_step = None
            if self.use_cosine_similarity_constraint and cached_benign_sim_stats is not None:
                proxy_param_flat_after_step = proxy_param.view(-1)
                if getattr(self, 'use_pairwise_similarity_in_constraint', False):
                    benign_updates_for_ref = self.benign_updates_gpu if (hasattr(self, 'benign_updates_gpu') and self.benign_updates_gpu) else [u.to(target_device) for u in self.benign_updates] if self.benign_updates else []
                    other_att = getattr(self, 'other_attacker_updates_gpu', None) or []
                    all_others = list(benign_updates_for_ref) + list(other_att)
                    if len(all_others) == 0:
                        sim_att_to_benign_after_step = torch.tensor(1.0, device=target_device, dtype=proxy_param_flat_after_step.dtype)
                    else:
                        sims_after = [torch.cosine_similarity(proxy_param_flat_after_step.unsqueeze(0), u.view(-1).to(target_device).unsqueeze(0), dim=1).squeeze(0) for u in all_others]
                        sim_att_to_benign_after_step = torch.stack(sims_after).mean()
                else:
                    benign_updates_for_ref = self.benign_updates_gpu if (hasattr(self, 'benign_updates_gpu') and self.benign_updates_gpu) else [u.to(target_device) for u in self.benign_updates] if self.benign_updates else []
                    aggregation_without_current_after_step, _, _ = self._aggregate_update_no_beta(
                        proxy_param, benign_updates_gpu=benign_updates_for_ref,
                        other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                        include_current_attacker=False
                    )
                    sim_att_to_benign_after_step = torch.cosine_similarity(
                        proxy_param_flat_after_step.unsqueeze(0),
                        aggregation_without_current_after_step.view(-1).unsqueeze(0), dim=1
                    ).squeeze(0)
                sim_att_val_after_step = sim_att_to_benign_after_step.item()
            
            # ============================================================
            # Update Lagrangian multipliers (if using Lagrangian mechanism)
            # Use dual ascent method according to paper dual problem
            # Paper Algorithm 1 Step 7: Update λ(t) according to eq:dual
            # ============================================================
            if self.use_lagrangian_dual and self.lambda_dist is not None:
                # Dual ascent method: λ(t+1) = max(0, λ(t) + α_λ * g)
                # g = constraint value (can be negative when satisfied)
                # When constraint is violated (g > 0), λ increases to penalize violation
                # When constraint is satisfied (g < 0), λ decreases (but clamped to ≥ 0)
                
                # CRITICAL: Check dist_bound_val (effective dist_bound) instead of self.dist_bound
                # dist_bound_val may be computed from benign_max even when self.dist_bound is None
                if dist_bound_val is not None:
                    # Use distance computed BEFORE step for lambda update (consistent with optimization objective)
                    # Lambda update should be based on the state that was used to compute the gradient
                    dist_att_val = dist_att_to_global.item()
                    lambda_dist_val = self.lambda_dist.item() if isinstance(self.lambda_dist, torch.Tensor) else self.lambda_dist
                    
                    # Standard dual ascent update: λ_dist(t+1) = max(0, λ_dist(t) + α_λ * g_dist)
                    # where g_dist = dist_att_to_global - dist_bound
                    # - If constraint is violated (g_dist > 0): λ_dist increases
                    # - If constraint is satisfied (g_dist < 0): λ_dist decreases (clamped to ≥ 0)
                    g_dist_val = dist_att_val - dist_bound_val
                    # Two λ update modes:
                    # - "classic": λ += lr * g   (current implementation)
                    # - "alm":     λ += ρ * g    (standard ALM-style multiplier update)
                    if self.use_augmented_lagrangian and self.lambda_update_mode == "alm":
                        rho_dist_val = float(self.rho_dist.item() if isinstance(self.rho_dist, torch.Tensor) else (self.rho_dist or 0.0))
                        new_lambda_dist = lambda_dist_val + rho_dist_val * g_dist_val
                    else:
                        new_lambda_dist = lambda_dist_val + self.lambda_dist_lr * g_dist_val
                    new_lambda_dist = max(0.0, new_lambda_dist)  # Ensure non-negative
                    # OPTIMIZATION 5: Keep multiplier on same device when updating
                    self.lambda_dist = torch.tensor(new_lambda_dist, device=target_device, requires_grad=False)
                    
                    # Update TWO cosine similarity constraint multipliers independently
                    if self.use_cosine_similarity_constraint:
                        if self.lambda_sim_low is not None and self.lambda_sim_up is not None:
                            # Get current values
                            lambda_sim_low_val = self.lambda_sim_low.item() if isinstance(self.lambda_sim_low, torch.Tensor) else self.lambda_sim_low
                            lambda_sim_up_val = self.lambda_sim_up.item() if isinstance(self.lambda_sim_up, torch.Tensor) else self.lambda_sim_up
                            
                            # g_sim_low = sim_bound_low - sim_att_to_agg <= 0
                            # g_sim_up = sim_att_to_agg - sim_bound_up <= 0
                            # Use similarity computed BEFORE step for lambda update
                            g_sim_low_val = g_sim_low.item()
                            g_sim_up_val = g_sim_up.item()
                            
                            # Standard dual ascent update (independently for each constraint)
                            # λ_sim_low(t+1) = max(0, λ_sim_low(t) + α * g_sim_low)
                            # λ_sim_up(t+1) = max(0, λ_sim_up(t) + α * g_sim_up)
                            if self.use_augmented_lagrangian and self.lambda_update_mode == "alm":
                                rho_sim_low_val = float(self.rho_sim_low.item() if isinstance(self.rho_sim_low, torch.Tensor) else (self.rho_sim_low or 0.0))
                                new_lambda_sim_low = lambda_sim_low_val + rho_sim_low_val * g_sim_low_val
                            else:
                                new_lambda_sim_low = lambda_sim_low_val + self.lambda_sim_low_lr * g_sim_low_val
                            new_lambda_sim_low = max(0.0, new_lambda_sim_low)
                            
                            if self.use_augmented_lagrangian and self.lambda_update_mode == "alm":
                                rho_sim_up_val = float(self.rho_sim_up.item() if isinstance(self.rho_sim_up, torch.Tensor) else (self.rho_sim_up or 0.0))
                                new_lambda_sim_up = lambda_sim_up_val + rho_sim_up_val * g_sim_up_val
                            else:
                                new_lambda_sim_up = lambda_sim_up_val + self.lambda_sim_up_lr * g_sim_up_val
                            new_lambda_sim_up = max(0.0, new_lambda_sim_up)
                            
                            self.lambda_sim_low = torch.tensor(new_lambda_sim_low, device=target_device, requires_grad=False)
                            self.lambda_sim_up = torch.tensor(new_lambda_sim_up, device=target_device, requires_grad=False)

                    # ============================================================
                    # Augmented Lagrangian penalty update (ρ) - monotone strategy
                    #
                    # For each constraint i:
                    #   σ_i = max(0, g_i)
                    #   if prev exists and σ_i > rho_theta * prev: increase ρ_i
                    #   else keep ρ_i unchanged
                    #
                    # We update ρ based on g values computed BEFORE the step
                    # (consistent with the gradient/objective used for this iteration).
                    # ============================================================
                    if self.use_augmented_lagrangian and self.rho_adaptive:
                        theta = float(self.rho_theta)
                        inc = float(self.rho_increase_factor)
                        rho_min = float(self.rho_min)
                        rho_max = float(self.rho_max)

                        # Distance constraint
                        sigma_dist = max(0.0, float(g_dist_val))
                        if self.rho_dist is not None:
                            prev = self._prev_violation_dist
                            rho_val = float(self.rho_dist.item() if isinstance(self.rho_dist, torch.Tensor) else self.rho_dist)
                            if prev is not None and sigma_dist > theta * prev:
                                rho_val = min(rho_max, max(rho_min, rho_val * inc))
                                self.rho_dist = torch.tensor(rho_val, device=target_device, requires_grad=False)
                            self._prev_violation_dist = sigma_dist

                        # Similarity constraints (two-sided)
                        if self.use_cosine_similarity_constraint:
                            sigma_low = max(0.0, float(g_sim_low_val))
                            sigma_up = max(0.0, float(g_sim_up_val))

                            if self.rho_sim_low is not None:
                                prev = self._prev_violation_sim_low
                                rho_val = float(self.rho_sim_low.item() if isinstance(self.rho_sim_low, torch.Tensor) else self.rho_sim_low)
                                if prev is not None and sigma_low > theta * prev:
                                    rho_val = min(rho_max, max(rho_min, rho_val * inc))
                                    self.rho_sim_low = torch.tensor(rho_val, device=target_device, requires_grad=False)
                                self._prev_violation_sim_low = sigma_low

                            if self.rho_sim_up is not None:
                                prev = self._prev_violation_sim_up
                                rho_val = float(self.rho_sim_up.item() if isinstance(self.rho_sim_up, torch.Tensor) else self.rho_sim_up)
                                if prev is not None and sigma_up > theta * prev:
                                    rho_val = min(rho_max, max(rho_min, rho_val * inc))
                                    self.rho_sim_up = torch.tensor(rho_val, device=target_device, requires_grad=False)
                                self._prev_violation_sim_up = sigma_up
                    
                    # Logging: Print every step with multiplier values (before and after update)
                    # Track all multipliers and constraint values
                    # Show both before-step and after-step values for clarity
                    grad_norm = proxy_param.grad.norm().item() if proxy_param.grad is not None else 0.0
                    log_msg = f"      [Attacker {self.client_id}] Step {step}/{self.proxy_steps-1}: " \
                              f"dist_att(before={dist_att_val:.4f}, after={dist_att_val_after_step:.4f}), " \
                              f"g_dist={g_dist_val:.4f}, " \
                              f"λ_dist({lambda_dist_val:.4f}→{new_lambda_dist:.4f}), " \
                              f"loss={global_loss.item():.4f}, grad_norm={grad_norm:.4f}"

                    # Add ρ info when using ALM
                    if self.use_augmented_lagrangian:
                        rho_dist_val_log = float(self.rho_dist.item() if isinstance(self.rho_dist, torch.Tensor) else (self.rho_dist or 0.0))
                        log_msg += f", ρ_dist={rho_dist_val_log:.4f}"
                    
                    # Add similarity constraint info if enabled
                    if self.use_cosine_similarity_constraint and cached_benign_sim_stats is not None:
                        sim_att_log = sim_att_to_benign.item()
                        sim_bound_low_log = max(-1.0, min(1.0, float(self.sim_bound_low))) if self.sim_bound_low is not None else max(-1.0, min(1.0, cached_benign_sim_stats['min'].item()))
                        if self.sim_bound_up is not None:
                            sim_bound_up_log = max(-1.0, min(1.0, float(self.sim_bound_up)))
                        else:
                            sim_bound_up_log = max(-1.0, min(1.0, cached_benign_sim_stats['mean'].item()))
                        
                        log_msg += f", sim_att(before={sim_att_log:.4f}, after={sim_att_val_after_step:.4f})∈[{sim_bound_low_log:.4f},{sim_bound_up_log:.4f}], " \
                                   f"λ_sim_low({lambda_sim_low_val:.4f}→{new_lambda_sim_low:.4f}), " \
                                   f"λ_sim_up({lambda_sim_up_val:.4f}→{new_lambda_sim_up:.4f}), " \
                                   f"g_low={g_sim_low_val:.4f}, g_up={g_sim_up_val:.4f}"
                        if self.use_augmented_lagrangian:
                            rho_sim_low_val_log = float(self.rho_sim_low.item() if isinstance(self.rho_sim_low, torch.Tensor) else (self.rho_sim_low or 0.0))
                            rho_sim_up_val_log = float(self.rho_sim_up.item() if isinstance(self.rho_sim_up, torch.Tensor) else (self.rho_sim_up or 0.0))
                            log_msg += f", ρ_sim_low={rho_sim_low_val_log:.4f}, ρ_sim_up={rho_sim_up_val_log:.4f}"
                    print(log_msg)
                    
                    # ============================================================
                    # Early stopping: Stop when ALL constraints are satisfied and stable
                    # ============================================================
                    # Strategy: Stop after N consecutive steps satisfying ALL constraints
                    # CRITICAL: Use AFTER-STEP values for accurate constraint checking
                    # This ensures Early stopping check matches the final state after parameter update
                    # ============================================================
                    
                    # Check distance constraint using AFTER-STEP value
                    dist_satisfied = dist_att_val_after_step <= dist_bound_val if dist_bound_val is not None else True
                    
                    # Check similarity constraints (TWO-SIDED: both lower and upper bounds) using AFTER-STEP value
                    if self.use_cosine_similarity_constraint and cached_benign_sim_stats is not None:
                        sim_att_val = sim_att_val_after_step
                        sim_bound_low_val = max(-1.0, min(1.0, float(self.sim_bound_low))) if self.sim_bound_low is not None else max(-1.0, min(1.0, cached_benign_sim_stats['min'].item()))
                        if self.sim_bound_up is not None:
                            sim_bound_up_val = max(-1.0, min(1.0, float(self.sim_bound_up)))
                        else:
                            sim_bound_up_val = max(-1.0, min(1.0, cached_benign_sim_stats['mean'].item()))
                        sim_satisfied = (sim_att_val >= sim_bound_low_val) and (sim_att_val <= sim_bound_up_val)
                    else:
                        sim_satisfied = True  # If similarity constraint not enabled, consider it satisfied
                        sim_att_val = 0.0  # Dummy value for logging
                        benign_sim_mean_val = 0.0  # Dummy value for logging
                    
                    # Both constraints satisfied
                    all_constraints_satisfied = dist_satisfied and sim_satisfied
                    
                    if all_constraints_satisfied:
                        constraint_satisfied_steps += 1
                        if constraint_satisfied_steps >= constraint_stability_steps:
                            log_msg = f"    [Attacker {self.client_id}] Early stopping: "
                            log_msg += f"dist_att={dist_att_val_after_step:.4f} <= dist_bound={dist_bound_val:.4f} "
                            if self.use_cosine_similarity_constraint and cached_benign_sim_stats is not None:
                                sim_bound_low_log = max(-1.0, min(1.0, float(self.sim_bound_low))) if self.sim_bound_low is not None else max(-1.0, min(1.0, cached_benign_sim_stats['min'].item()))
                                if self.sim_bound_up is not None:
                                    sim_bound_up_log = max(-1.0, min(1.0, float(self.sim_bound_up)))
                                else:
                                    sim_bound_up_log = max(-1.0, min(1.0, cached_benign_sim_stats['mean'].item()))
                                log_msg += f", sim_att={sim_att_val_after_step:.4f}∈[{sim_bound_low_log:.4f},{sim_bound_up_log:.4f}] "
                            log_msg += f"for {constraint_satisfied_steps} consecutive steps (step {step}/{self.proxy_steps-1})"
                            print(log_msg)
                            break
                    else:
                        constraint_satisfied_steps = 0  # Reset counter when any constraint is violated
                    
                    prev_dist_val = dist_att_val_after_step
        
        # ============================================================
        # Print final optimization state
        # ============================================================
        # Use effective dist_bound (which may be auto-computed from benign max if self.dist_bound is None)
        effective_dist_bound_final = getattr(self, '_effective_dist_bound', self.dist_bound)
        if effective_dist_bound_final is not None:
            # CRITICAL: Compute distance to aggregation EXCLUDING current attacker (consistent with optimization loop)
            # Optimization does not include current attacker
            # CRITICAL: Compute distance to aggregation EXCLUDING current attacker (consistent with optimization loop)
            # This matches the constraint used during optimization
            final_global_ref_gpu, _, _ = self._aggregate_update_no_beta(
                proxy_param,
                benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                include_current_attacker=False  # CRITICAL: Exclude current attacker (consistent with optimization)
            )
            final_dist_att_to_global = torch.norm(proxy_param.view(-1) - final_global_ref_gpu.view(-1))
            final_dist_att = final_dist_att_to_global.item()
            dist_bound_final = float(effective_dist_bound_final) if isinstance(effective_dist_bound_final, torch.Tensor) else effective_dist_bound_final
            final_g_dist = final_dist_att - dist_bound_final
            final_lambda_dist = self.lambda_dist.item() if isinstance(self.lambda_dist, torch.Tensor) else self.lambda_dist if self.lambda_dist is not None else 0.0
            # MODIFIED: Use include_current_attacker=True for final_loss display
            # This shows the actual global_loss that was optimized (consistent with optimization loop)
            final_loss = self._compute_global_loss(
                proxy_param,
                benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                include_current_attacker=True  # MODIFIED: Include current attacker (for display, matches optimization)
            ).item()
            final_violation = max(0, final_g_dist)
            violation_pct = (final_violation / dist_bound_final * 100) if dist_bound_final > 0 else 0.0
            
            # Compute final similarity metrics (always compute, not just when constraint is enabled)
            final_sim_info = ""
            if len(self.benign_updates) > 0:
                proxy_param_flat = proxy_param.view(-1)
                target_device = proxy_param_flat.device
                if getattr(self, 'use_pairwise_similarity_in_constraint', False):
                    benign_list = getattr(self, 'benign_updates_gpu', None) or [u.to(target_device) for u in self.benign_updates]
                    other_list = getattr(self, 'other_attacker_updates_gpu', None) or []
                    all_others = list(benign_list) + list(other_list)
                    if len(all_others) == 0:
                        final_sim_att_to_benign = 1.0
                    else:
                        final_sims = [torch.cosine_similarity(proxy_param_flat.unsqueeze(0), u.view(-1).to(target_device).unsqueeze(0), dim=1).squeeze(0).item() for u in all_others]
                        final_sim_att_to_benign = sum(final_sims) / len(final_sims)
                else:
                    final_aggregation_without_current, _, _ = self._aggregate_update_no_beta(
                        proxy_param,
                        benign_updates_gpu=getattr(self, 'benign_updates_gpu', None),
                        other_attacker_updates_gpu=getattr(self, 'other_attacker_updates_gpu', None),
                        include_current_attacker=False
                    )
                    final_aggregation_flat = final_aggregation_without_current.view(-1)
                    final_sim_att_to_benign = torch.cosine_similarity(
                        proxy_param_flat.unsqueeze(0),
                        final_aggregation_flat.unsqueeze(0), dim=1
                    ).squeeze(0).item()
                
                # Compute similarity bounds (manual sim_bound_low/up if set, else benign min/max)
                if cached_benign_sim_stats is not None:
                    if self.sim_bound_low is not None:
                        final_sim_bound_low = max(-1.0, min(1.0, float(self.sim_bound_low)))
                    else:
                        final_sim_bound_low = torch.clamp(cached_benign_sim_stats['min'], min=-1.0, max=1.0).item()
                    if self.sim_bound_up is not None:
                        final_sim_bound_up = max(-1.0, min(1.0, float(self.sim_bound_up)))
                    else:
                        final_sim_bound_up = torch.clamp(cached_benign_sim_stats['mean'], min=-1.0, max=1.0).item()
                    
                    if self.use_cosine_similarity_constraint:
                        # Compute constraint violations
                        final_g_sim_low = final_sim_bound_low - final_sim_att_to_benign
                        final_g_sim_up = final_sim_att_to_benign - final_sim_bound_up
                        
                        # Get final lambda values
                        final_lambda_sim_low = self.lambda_sim_low.item() if isinstance(self.lambda_sim_low, torch.Tensor) else self.lambda_sim_low if self.lambda_sim_low is not None else 0.0
                        final_lambda_sim_up = self.lambda_sim_up.item() if isinstance(self.lambda_sim_up, torch.Tensor) else self.lambda_sim_up if self.lambda_sim_up is not None else 0.0
                        
                        final_sim_info = f", final_sim={final_sim_att_to_benign:.4f}∈[{final_sim_bound_low:.4f},{final_sim_bound_up:.4f}], " \
                                        f"g_sim_low={final_g_sim_low:.4f}, g_sim_up={final_g_sim_up:.4f}, " \
                                        f"λ_sim_low={final_lambda_sim_low:.4f}, λ_sim_up={final_lambda_sim_up:.4f}"
                    else:
                        # Similarity constraint not enabled, but still show similarity value and bounds
                        final_sim_info = f", final_sim={final_sim_att_to_benign:.4f}∈[{final_sim_bound_low:.4f},{final_sim_bound_up:.4f}] (constraint disabled)"
                else:
                    # No similarity statistics available, just show the value
                    final_sim_info = f", final_sim={final_sim_att_to_benign:.4f}"
            
            print(f"    [Attacker {self.client_id}] Optimization completed: "
                  f"final_dist_att={final_dist_att:.4f}, dist_bound={dist_bound_final:.4f}, g_dist={final_g_dist:.4f}, "
                  f"λ_dist={final_lambda_dist:.4f}, loss={final_loss:.4f}, "
                  f"violation={final_violation:.4f} ({violation_pct:.1f}%){final_sim_info}")
        
        malicious_update = proxy_param.detach()
        
        # CRITICAL: Now that optimization is complete, move model back to CPU to free GPU memory
        # The computation graph is no longer needed after optimization
        if self._model_on_gpu:
            self.model.cpu()
            self._ensure_model_on_device(self.model, torch.device('cpu'))
            self._model_on_gpu = False
        
        # Clean up optimizer and GPU caches to free GPU memory
        del proxy_opt
        # CRITICAL: Clean up GPU versions of updates to free memory
        if hasattr(self, 'benign_updates_gpu'):
            del self.benign_updates_gpu
        if hasattr(self, 'other_attacker_updates_gpu'):
            del self.other_attacker_updates_gpu
        torch.cuda.empty_cache()
        
        # ============================================================
        # STEP 8: Final constraint check (pure Lagrangian optimization, no projection)
        # ============================================================
        # Use effective dist_bound (which may be auto-computed from benign max if self.dist_bound is None)
        # CRITICAL: Ensure malicious_update is on CPU for final check (GPU caches have been cleaned up)
        malicious_update_cpu = malicious_update.cpu() if malicious_update.device.type == 'cuda' else malicious_update
        effective_dist_bound = getattr(self, '_effective_dist_bound', self.dist_bound)
        if effective_dist_bound is not None:
            # CRITICAL: Compute distance to aggregation EXCLUDING current attacker (consistent with optimization loop)
            # Use CPU versions since GPU caches have been cleaned up
            # This matches the constraint used during optimization
            final_global_ref_cpu, _, _ = self._aggregate_update_no_beta(
                malicious_update_cpu,
                benign_updates=self.benign_updates,
                other_attacker_updates_list=self.other_attacker_updates if hasattr(self, 'other_attacker_updates') else None,
                include_current_attacker=False  # CRITICAL: Exclude current attacker (consistent with optimization)
            )
            dist_to_global_final = torch.norm(malicious_update_cpu.view(-1) - final_global_ref_cpu.view(-1))
            dist_to_global = dist_to_global_final.item()
            dist_bound_final = float(effective_dist_bound) if isinstance(effective_dist_bound, torch.Tensor) else effective_dist_bound
            constraint_violation = max(0, dist_to_global - dist_bound_final)
            violation_ratio = constraint_violation / dist_bound_final if dist_bound_final > 0 else 0.0
            
            if constraint_violation > 0:
                if self.use_lagrangian_dual:
                    lambda_dist_final = self.lambda_dist.item() if isinstance(self.lambda_dist, torch.Tensor) else self.lambda_dist
                    print(f"    [Attacker {self.client_id}] Distance constraint violation: {constraint_violation:.6f} "
                          f"(λ_dist={lambda_dist_final:.4f}, violation={violation_ratio*100:.1f}%)")
                else:
                    print(f"    [Attacker {self.client_id}] Distance constraint violation: {constraint_violation:.6f} "
                          f"(violation={violation_ratio*100:.1f}%)")
        
        # Compute final attack objective value for logging
        # Use aggregated update for final loss (consistent with optimization objective)
        # CRITICAL: Use CPU version for final aggregation (GPU caches have been cleaned up)
        # MODIFIED: Include current attacker to match optimization objective (global_loss calculation)
        final_aggregated_update, _, _ = self._aggregate_update_no_beta(
            malicious_update_cpu, 
            self.benign_updates,
            other_attacker_updates_list=self.other_attacker_updates if hasattr(self, 'other_attacker_updates') else None,
            include_current_attacker=True  # MODIFIED: Include current attacker (matches optimization objective)
        )
        final_global_loss = self._proxy_global_loss(final_aggregated_update, max_batches=self.proxy_max_batches_eval, skip_dim_check=True)
        
        malicious_norm = torch.norm(malicious_update).item()
        log_msg = f"    [Attacker {self.client_id}] AugMP: " \
                  f"F(w'_g)={final_global_loss.item():.4f}, " \
                  f"||w'_j||={malicious_norm:.4f}"
        
        # If using the Lagrangian mechanism, display the multiplier values
        if self.use_lagrangian_dual and self.lambda_dist is not None:
            lambda_dist_final = self.lambda_dist.item() if isinstance(self.lambda_dist, torch.Tensor) else self.lambda_dist
            log_msg += f", λ_dist={lambda_dist_final:.4f}"
            
            # Add similarity multipliers if enabled
            if self.use_cosine_similarity_constraint:
                if self.lambda_sim_low is not None:
                    lambda_sim_low_final = self.lambda_sim_low.item() if isinstance(self.lambda_sim_low, torch.Tensor) else self.lambda_sim_low
                    log_msg += f", λ_sim_low={lambda_sim_low_final:.4f}"
                if self.lambda_sim_up is not None:
                    lambda_sim_up_final = self.lambda_sim_up.item() if isinstance(self.lambda_sim_up, torch.Tensor) else self.lambda_sim_up
                    log_msg += f", λ_sim_up={lambda_sim_up_final:.4f}"
        
        print(log_msg)
        
        # CRITICAL: malicious_update_cpu is already on CPU (created above for final check)
        # No need to convert again, but ensure it's detached
        malicious_update_final = malicious_update_cpu.detach()
        # Clean up GPU references if malicious_update was on GPU
        if malicious_update.device.type == 'cuda':
            del malicious_update
        del final_global_loss
        torch.cuda.empty_cache()
        
        return malicious_update_final