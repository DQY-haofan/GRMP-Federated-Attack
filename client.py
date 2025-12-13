# client.py
# Provides the Client class for federated learning clients, including benign and attacker clients.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from models import VGAE
from torch.nn.utils import stateless

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
            alpha: Proximal regularization coefficient α ∈ [0,1] from paper formula (1) (must be provided, no default)
        
        Note: All parameters must be explicitly provided. Default values are removed to prevent
        inconsistencies with config settings. See main.py for proper usage.
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.alpha = alpha  # Regularization coefficient α ∈ [0,1] from paper formula (1)
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
        """
        Calculate the model update (Current - Initial).
        
        Args:
            initial_params: Initial model parameters (flattened)
            
        Returns:
            Model update tensor (flattened)
        """
        current_params = self.model.get_flat_params()
        return current_params - initial_params

    def local_train(self, epochs=None) -> torch.Tensor:
        """Base local training method (to be overridden)."""
        raise NotImplementedError


# BenignClient class for benign clients
class BenignClient(Client):

    def __init__(self, client_id: int, model: nn.Module, data_loader, lr, local_epochs, alpha,
                 data_indices=None):
        super().__init__(client_id, model, data_loader, lr, local_epochs, alpha)
        # Track assigned data indices for proper aggregation weighting
        self.data_indices = data_indices or []

    def prepare_for_round(self, round_num: int):
        """Benign clients do not require special preparation."""
        self.set_round(round_num)

    def local_train(self, epochs=None) -> torch.Tensor:
        """Perform local training - includes proximal regularization."""
        if epochs is None:
            epochs = self.local_epochs
            
        self.model.train()
        initial_params = self.model.get_flat_params().clone()
        
        # Proximal regularization coefficient (paper formula (1): α ∈ [0,1])
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
                 data_indices, lr, local_epochs, alpha,
                 dim_reduction_size=10000,
                 vgae_epochs=20, vgae_lr=0.01, graph_threshold=0.5,
                 proxy_step=0.1,
                 claimed_data_size=1.0):
        """
        Initialize an attacker client with VGAE-based camouflage capabilities.
        
        Args:
            client_id: Unique identifier for the client
            model: The neural network model (will be deep copied)
            data_manager: DataManager instance for managing attacker data
            data_indices: List of data indices assigned to this client
            lr: Learning rate for local training (must be provided, no default)
            local_epochs: Number of local training epochs per round (must be provided, no default)
            alpha: Proximal regularization coefficient α ∈ [0,1] (must be provided, no default)
            dim_reduction_size: Dimensionality for feature reduction (default: 10000)
            vgae_epochs: Number of epochs for VGAE training (default: 20)
            vgae_lr: Learning rate for VGAE optimizer (default: 0.01)
            graph_threshold: Threshold for graph adjacency matrix binarization (default: 0.5)
            proxy_step: Step size for gradient-free ascent toward global-loss proxy (default: 0.1)
            claimed_data_size: Reported data size D'_j(t) for weighted aggregation (default: 1.0)
        
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

        dummy_loader = data_manager.get_empty_loader()
        super().__init__(client_id, model, dummy_loader, lr, local_epochs, alpha)
        self.is_attacker = True

        # VGAE components
        self.vgae = None
        self.vgae_optimizer = None
        self.benign_updates = []
        self.feature_indices = None
        
        # Data-agnostic attack: no local data usage
        self.original_business_loader = None
        self.proxy_loader = data_manager.get_proxy_eval_loader()
        
        # Formula 4 constraints parameters
        self.d_T = None  # Distance threshold for constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = None  # Upper bound for constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
        self.global_model_params = None  # Store global model params for constraint (4b)
        self._flat_numel = int(self.model.get_flat_params().numel())  # Convert to Python int

    def prepare_for_round(self, round_num: int):
        """Prepare for a new training round."""
        self.set_round(round_num)
        # Data-agnostic attacker keeps an empty loader
        self.data_loader = self.data_manager.get_empty_loader()

    def receive_benign_updates(self, updates: List[torch.Tensor]):
        """Receive updates from benign clients."""
        # Store detached copies to avoid graph retention issues
        self.benign_updates = [u.detach().clone() for u in updates]

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
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from mean for all benign updates
        benign_stack = torch.stack([u.detach() for u in self.benign_updates])
        benign_mean = benign_stack.mean(dim=0)
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        
        # If gamma is not set, use all updates
        if self.gamma is None:
            return self.benign_updates
        
        # Convert gamma to capacity
        capacity = float(self.gamma)
        n = len(distances)
        
        # Handle edge cases
        if n == 0:
            return []
        if capacity <= 0:
            # If capacity is 0 or negative, select item with minimum distance
            min_idx = np.argmin(distances)
            return [self.benign_updates[min_idx]]
        
        # For 0-1 Knapsack with minimization objective:
        # We want to minimize sum of selected distances, subject to sum ≤ capacity
        # This is equivalent to: maximize number of items selected (or maximize sum of unselected)
        # while keeping sum of selected ≤ capacity
        
        # Greedy approach: sort by distance and select items until capacity is reached
        # This gives a good approximation and is efficient
        sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        
        selected_indices = []
        total_dist = 0.0
        
        for idx in sorted_indices:
            d = distances[idx]
            if total_dist + d <= capacity:
                selected_indices.append(idx)
                total_dist += d
            else:
                break
        
        # If no items selected (all distances > capacity), select the one with minimum distance
        if not selected_indices:
            min_idx = np.argmin(distances)
            selected_indices = [min_idx]
        
        # Return selected updates
        selected = [self.benign_updates[i] for i in sorted(selected_indices)]
        
        return selected
    
    def _get_selected_benign_indices(self) -> List[int]:
        """
        Get indices of selected benign updates (β selection).
        This is a helper method to avoid tensor comparison issues.
        """
        if not self.benign_updates:
            return []
        
        # Compute distances from mean for all benign updates
        benign_stack = torch.stack([u.detach() for u in self.benign_updates])
        benign_mean = benign_stack.mean(dim=0)
        distances = torch.norm(benign_stack - benign_mean, dim=1).cpu().numpy()
        
        # If gamma is not set, use all updates
        if self.gamma is None:
            return list(range(len(self.benign_updates)))
        
        # Convert gamma to capacity
        capacity = float(self.gamma)
        n = len(distances)
        
        # Handle edge cases
        if n == 0:
            return []
        if capacity <= 0:
            # If capacity is 0 or negative, select item with minimum distance
            min_idx = np.argmin(distances)
            return [int(min_idx)]
        
        # Greedy approach: sort by distance and select items until capacity is reached
        sorted_indices = np.argsort(distances)  # Sort by distance (ascending)
        
        selected_indices = []
        total_dist = 0.0
        
        for idx in sorted_indices:
            d = distances[idx]
            if total_dist + d <= capacity:
                selected_indices.append(int(idx))
                total_dist += d
            else:
                break
        
        # If no items selected (all distances > capacity), select the one with minimum distance
        if not selected_indices:
            min_idx = np.argmin(distances)
            selected_indices = [int(min_idx)]
        
        return sorted(selected_indices)

    def local_train(self, epochs=None) -> torch.Tensor:
        """
        Attacker does not perform local training (data-agnostic attack).
        
        Attackers are not assigned local data, so they return zero update.
        The actual attack is generated in camouflage_update using VGAE+GSP.
        """
        # Attackers don't have local data, return zero update
        initial_params = self.model.get_flat_params().clone()
        return torch.zeros_like(initial_params)

    def _get_reduced_features(self, updates: List[torch.Tensor], fix_indices=True) -> torch.Tensor:
        """
        Helper function to reduce dimensionality of updates.
        Randomly selects indices to slice the high-dimensional vector.
        
        Args:
            updates: List of update tensors to reduce
            fix_indices: If True, reuse existing feature_indices; if False, generate new ones
            
        Returns:
            Stacked reduced features tensor of shape (I, M) where I=num_updates, M=dim_reduction_size
        """
        stacked_updates = torch.stack(updates)
        total_dim = int(stacked_updates.shape[1])  # Convert to Python int
        
        # If update dimension is smaller than reduction target, skip reduction
        if total_dim <= self.dim_reduction_size:
            return stacked_updates
            
        # Fix feature indices at the start of each attack round to ensure training consistency within the round
        if self.feature_indices is None or not fix_indices:
            # Randomly select indices, but use client_id to ensure different attackers get different indices
            # This ensures diversity among multiple attackers
            import hashlib
            # Use client_id and total_dim to create a unique seed for each attacker
            seed_str = f"{self.client_id}_{total_dim}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
            torch.manual_seed(seed)
            self.feature_indices = torch.randperm(total_dim)[:self.dim_reduction_size].to(self.device)
            torch.manual_seed(None)  # Reset to default random state
            
        # Select features
        reduced_features = torch.index_select(stacked_updates, 1, self.feature_indices)
        return reduced_features

    def _flat_to_param_dict(self, flat_params: torch.Tensor, skip_dim_check: bool = False) -> Dict[str, torch.Tensor]:
        """
        Convert flat tensor to param dict for stateless.functional_call.
        
        Args:
            flat_params: Flattened parameter tensor
            skip_dim_check: If True, skip dimension check (for performance in loops)
        """
        param_dict = {}
        offset = 0
        flat_params = flat_params.view(-1)  # Ensure 1D (O(1), just view change)
        total_numel = int(flat_params.numel())  # Convert to Python int
        
        for name, param in self.model.named_parameters():
            numel = int(param.numel())  # Convert to Python int
            if not skip_dim_check and offset + numel > total_numel:
                # Dimension mismatch: return empty dict to avoid errors
                print(f"    [Attacker {self.client_id}] Param dict dimension mismatch at {name}: offset {offset} + numel {numel} > total {total_numel}")
                return {}
            param_dict[name] = flat_params[offset:offset + numel].view_as(param)
            offset += numel
        return param_dict

    def _proxy_global_loss(self, malicious_update: torch.Tensor, max_batches: int = 1, 
                           skip_dim_check: bool = False) -> torch.Tensor:
        """
        Differentiable proxy for F(w'_g): cross-entropy on a small clean subset,
        using stateless.functional_call with (w_g + malicious_update).
        
        Args:
            malicious_update: Update vector to evaluate
            max_batches: Maximum number of batches to process
            skip_dim_check: If True, skip dimension check (for performance in loops)
        """
        if self.global_model_params is None or self.proxy_loader is None:
            return torch.tensor(0.0, device=self.device)

        # Ensure shapes match: flatten to 1D and check dimension
        malicious_update = malicious_update.view(-1)  # Flatten to 1D (O(1), just view change)
        if not skip_dim_check and int(malicious_update.numel()) != self._flat_numel:
            # Dimension mismatch: return zero loss to avoid errors
            print(f"    [Attacker {self.client_id}] Proxy loss dimension mismatch: got {malicious_update.numel()}, expected {self._flat_numel}")
            return torch.tensor(0.0, device=self.device)

        candidate_params = self.global_model_params + malicious_update
        # Skip dimension check if already validated (performance optimization)
        param_dict = self._flat_to_param_dict(candidate_params, skip_dim_check=skip_dim_check)

        total_loss = 0.0
        batches = 0

        for batch in self.proxy_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = stateless.functional_call(
                self.model,
                param_dict,
                args=(),
                kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
            )

            ce_loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + ce_loss
            batches += 1
            if batches >= max_batches:
                break

        if batches == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / batches

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
            # Following paper: hidden1_dim=32, hidden2_dim=16
            hidden_dim = 32
            latent_dim = 16
            self.vgae = VGAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=0.0).to(self.device)
            self.vgae_optimizer = optim.Adam(self.vgae.parameters(), lr=self.vgae_lr)

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
        """Set global model parameters for constraint (4b) calculation."""
        self.global_model_params = global_params.clone().detach().to(self.device)
    
    def set_constraint_params(self, d_T: float = None, gamma: float = None):
        """Set constraint parameters for Formula 4."""
        self.d_T = d_T  # Constraint (4b): d(w'_j(t), w_g(t)) ≤ d_T
        self.gamma = gamma  # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ

    def _compute_global_loss(self, malicious_update: torch.Tensor, 
                            selected_benign: List[torch.Tensor],
                            beta_selection: List[int]) -> torch.Tensor:
        """
        Compute global loss F(w'_g(t)) according to paper Equation 3.
        
        Paper formulation:
        F(w'_g(t)) = Σ_{i=1}^I (D_i(t)/D(t)) β'_{i,j}(t) F(w_i(t)) 
                    + (D'(t)/D(t)) F'(w'_j(t))
        
        For optimization, we use proxy loss F'(w'_j(t)) which approximates
        the global loss when w'_g(t) = w_g(t) + w'_j(t).
        
        Args:
            malicious_update: w'_j(t) - malicious update
            selected_benign: List of selected benign updates (based on β selection)
            beta_selection: List of indices indicating which benign updates are selected
        
        Returns:
            Global loss F(w'_g(t)) to be maximized
        """
        if self.global_model_params is None or self.proxy_loader is None:
            # Fallback: return zero if proxy not available
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute proxy loss F'(w'_j(t)) using the proxy loader
        # This approximates the global loss when the malicious update is aggregated
        proxy_loss = self._proxy_global_loss(malicious_update, max_batches=3, skip_dim_check=False)
        
        # Note: The full formulation would require computing F(w_i(t)) for each benign client,
        # but for optimization purposes, we use the proxy loss as an approximation.
        # The proxy loss captures the effect of the malicious update on the global model.
        
        return proxy_loss

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
        M = int(feature_matrix.shape[1])  # Reduced dimension - Convert to Python int
        
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
        degree_recon = adj_recon.sum(dim=1)
        L_recon = torch.diag(degree_recon) - adj_recon  # (M, M)
        
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
        if F_recon_rows == 0:
            # Empty feature matrix: return zeros
            print(f"    [Attacker {self.client_id}] F_recon is empty, using zero fallback")
            return torch.zeros(M, device=feature_matrix.device, dtype=feature_matrix.dtype)
        
        # Select a vector from F̂ as the malicious update
        # Use client_id to select different vectors for different attackers (for diversity)
        # This ensures different attackers use different malicious models from F̂
        select_idx = int(self.client_id % F_recon_rows)  # Ensure Python int
        gsp_attack = F_recon[select_idx].clone()  # Select one row from F̂ as w'_j(t), clone to avoid view issues
        
        # If F_recon has only one row, add small client-specific perturbation to ensure diversity
        # This prevents multiple attackers from generating identical attacks
        if F_recon_rows == 1:
            perturbation_scale = 0.01 * (self.client_id + 1)  # Scale based on client_id
            perturbation = torch.randn_like(gsp_attack) * perturbation_scale
            gsp_attack = gsp_attack + perturbation
        
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
        GRMP Attack using VGAE + GSP (data-agnostic attack).
        
        Attackers are not assigned local data and do not perform local training.
        The attack is generated purely from benign updates using VGAE+GSP.
        
        Paper Algorithm 1:
        1. Calculate A according to cosine similarity (eq. 8)
        2. Train VGAE to maximize L_loss (eq. 12), obtain optimal Â
        3. Use GSP module to obtain F̂, determine w'_j(t) based on F̂
        
        Args:
            poisoned_update: Zero update (attackers don't train, so this is always zero)
        
        Returns:
            Malicious update generated using VGAE+GSP
        """
        if not self.benign_updates:
            print(f"    [Attacker {self.client_id}] No benign updates, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        # Reset feature indices for this session
        self.feature_indices = None
        
        # ============================================================
        # STEP 1: Prepare feature matrix F ∈ R^{I×M}
        # ============================================================
        selected_benign = self._select_benign_subset()
        if not selected_benign:
            print(f"    [Attacker {self.client_id}] No benign subset selected, return zero update")
            return poisoned_update  # poisoned_update is always zero (attackers don't train)

        benign_stack = torch.stack([u.detach() for u in selected_benign])  # (I, full_dim)
        
        # Reduce dimensionality for computational efficiency
        reduced_benign = self._get_reduced_features(selected_benign, fix_indices=False)  # (I, M)
        M = int(reduced_benign.shape[1])  # Convert to Python int
        I = int(reduced_benign.shape[0])  # Convert to Python int
        
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
        
        # ============================================================
        # STEP 5: Expand GSP attack back to full dimension
        # Expand GSP attack from reduced dimension M back to full dimension.
        # Non-selected dimensions remain zero.
        # ============================================================
        malicious_update = torch.zeros_like(poisoned_update)
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
                    malicious_update[self.feature_indices] = gsp_attack_reduced
                else:
                    # Dimension mismatch: log warning and use zeros
                    print(f"    [Attacker {self.client_id}] GSP dimension mismatch: got {gsp_dim}, expected {expected_dim}, using zeros")
            else:
                # No dimension reduction: GSP attack should be full dimension
                if gsp_dim == total_dim:
                    # Correct dimension: use directly
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
        # STEP 6: Optimize attack objective according to paper Formula 4
        # Paper: max_{w'_j(t), β'_{i,j}(t)} F(w'_g(t))
        # s.t. d(w'_j(t), w'_g(t)) ≤ d_T  (Constraint 4b)
        #      Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ  (Constraint 4c)
        # ============================================================
        
        # Get beta selection indices (which benign updates were selected)
        # Use helper method to avoid tensor comparison issues
        beta_selection = self._get_selected_benign_indices()
        
        # ============================================================
        # STEP 7: Optimize w'_j(t) to maximize F(w'_g(t))
        # According to paper Equation 12, we maximize F(w'_g(t)) subject to constraints
        # ============================================================
        proxy_steps = 20  # Increased for better convergence
        proxy_lr = self.proxy_step
        # Add small client-specific perturbation to initial malicious_update to ensure diversity
        # This helps different attackers converge to different local optima
        if self.is_attacker:
            perturbation_scale = 0.001 * (self.client_id + 1)  # Small scale, client-specific
            initial_perturbation = torch.randn_like(malicious_update) * perturbation_scale
            proxy_param = (malicious_update + initial_perturbation).clone().detach().to(self.device)
        else:
            proxy_param = malicious_update.clone().detach().to(self.device)
        proxy_param.requires_grad_(True)
        proxy_opt = optim.Adam([proxy_param], lr=proxy_lr)
        
        # Check dimension once before loop (performance optimization)
        proxy_param_flat = proxy_param.view(-1)
        dim_valid = int(proxy_param_flat.numel()) == self._flat_numel
        
        for step in range(proxy_steps):
            proxy_opt.zero_grad()
            
            # Attack objective: Maximize F(w'_g(t)) according to paper Formula 4a
            # We use proxy loss as approximation of F(w'_g(t))
            global_loss = self._proxy_global_loss(proxy_param, max_batches=2, skip_dim_check=dim_valid)
            
            # Objective: maximize global_loss => minimize -global_loss
            objective = -global_loss
            
            # Compute constraint violations for logging (not used in optimization)
            constraint_b_violation = torch.tensor(0.0, device=self.device)
            if self.d_T is not None:
                # Constraint (4b): d(w'_j(t), w'_g(t)) ≤ d_T
                # Note: w'_g(t) = weighted_avg(selected_benign) + (D'_j/D) * w'_j(t)
                # For simplicity, we approximate: d(w'_j, w'_g) ≈ ||w'_j - w_g||
                # This is exact when D'_j << D (attacker has small weight)
                # Ensure torch.norm returns a scalar by flattening first
                dist_to_global = torch.norm(proxy_param.view(-1))
                constraint_b_violation = F.relu(dist_to_global - self.d_T)
            
            constraint_c_violation = torch.tensor(0.0, device=self.device)
            if self.gamma is not None and len(selected_benign) > 0:
                # Constraint (4c): Σ β'_{i,j}(t) d(w_i(t), w̄_i(t)) ≤ Γ
                sel_stack = torch.stack(selected_benign)
                sel_mean = sel_stack.mean(dim=0)
                distances = torch.norm(sel_stack - sel_mean, dim=1)
                agg_dist = distances.sum()  # This returns a scalar tensor (0-dim)
                constraint_c_violation = F.relu(agg_dist - self.gamma)
            
            # Backpropagate to maximize global loss
            objective.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_([proxy_param], max_norm=1.0)
            
            proxy_opt.step()
            
            # Apply hard constraint (4b) projection after each step
            # This ensures d(w'_j, w'_g) ≤ d_T is always satisfied
            if self.d_T is not None:
                dist_to_global = torch.norm(proxy_param).item()
                if dist_to_global > self.d_T:
                    # Project to constraint set: scale down to satisfy d ≤ d_T
                    proxy_param.data = proxy_param.data * (self.d_T / dist_to_global)
        
        malicious_update = proxy_param.detach()
        
        # ============================================================
        # STEP 8: Final hard constraint enforcement (Constraint 4b)
        # Ensure d(w'_j(t), w'_g(t)) ≤ d_T is strictly satisfied
        # ============================================================
        if self.d_T is not None:
            # Compute distance from global model
            # Approximation: d(w'_j, w'_g) ≈ ||w'_j - w_g||
            # This is exact when attacker weight is small
            dist_to_global = torch.norm(malicious_update).item()
            
            if dist_to_global > self.d_T:
                # Hard constraint: project to satisfy d ≤ d_T
                scale_factor = self.d_T / dist_to_global
                malicious_update = malicious_update * scale_factor
                final_norm = torch.norm(malicious_update).item()
                print(f"    [Attacker {self.client_id}] Applied hard constraint projection: "
                      f"scaled from {dist_to_global:.4f} to {final_norm:.4f}")
        
        # Compute final attack objective value for logging
        final_global_loss = self._proxy_global_loss(malicious_update, max_batches=1, skip_dim_check=False)
        
        # Compute constraint (4c) for logging
        constraint_c_value = torch.tensor(0.0, device=self.device)
        if self.gamma is not None and len(selected_benign) > 0:
            sel_stack = torch.stack(selected_benign)
            sel_mean = sel_stack.mean(dim=0)
            distances = torch.norm(sel_stack - sel_mean, dim=1)
            constraint_c_value = distances.sum()  # This returns a scalar tensor (0-dim), safe for .item()
        
        malicious_norm = torch.norm(malicious_update).item()
        print(f"    [Attacker {self.client_id}] GRMP Attack: "
              f"F(w'_g)={final_global_loss.item():.4f}, "
              f"||w'_j||={malicious_norm:.4f}, "
              f"constraint_c={constraint_c_value.item():.4f}")
        
        return malicious_update.detach()
