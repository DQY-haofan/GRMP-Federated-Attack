# models.py
# This module defines the NewsClassifierModel for News classification
# and the VGAE model for GRMP attack.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from typing import Tuple, Optional

# --- Constants ---
MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 4

# Optional LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("  Warning: peft library not available. LoRA support disabled. Install with: pip install peft")


class NewsClassifierModel(nn.Module):
    """
    DistilBERT-based model for news classification.
    Supports both full fine-tuning and LoRA fine-tuning modes.
    Wraps the Hugging Face AutoModelForSequenceClassification.
    
    Args:
        model_name: Pre-trained model name or path
        num_labels: Number of classification labels
        use_lora: If True, use LoRA fine-tuning instead of full fine-tuning
        lora_r: LoRA rank (rank of the low-rank matrices)
        lora_alpha: LoRA alpha (scaling factor, typically 2*r)
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS,
                 use_lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, lora_target_modules: Optional[list] = None):
        super().__init__()
        
        self.use_lora = use_lora
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Verify that the correct model is loaded
        model_type = type(self.model).__name__
        
        # Setup LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA support requires peft library. Install with: pip install peft"
                )
            
            # Default target modules for DistilBERT
            if lora_target_modules is None:
                # DistilBERT uses these module names for attention layers
                lora_target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",  # Don't add bias parameters
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, peft_config)
            
            # Print LoRA statistics
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Loaded model: {model_type} (from {model_name}) with LoRA")
            print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")
        else:
            print(f"  Loaded model: {model_type} (from {model_name}) [Full Fine-tuning]")
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights to avoid initial bias."""
        with torch.no_grad():
            # In LoRA mode, model structure may be different (PEFT wrapper)
            # Try to access classifier from base_model if using PEFT
            if self.use_lora and hasattr(self.model, 'base_model'):
                # PEFT model: access through base_model
                base_model = self.model.base_model.model
                if hasattr(base_model, 'classifier'):
                    nn.init.xavier_uniform_(base_model.classifier.weight)
                    nn.init.zeros_(base_model.classifier.bias)
            elif hasattr(self.model, 'classifier'):
                # Standard model: direct access
                nn.init.xavier_uniform_(self.model.classifier.weight)
                nn.init.zeros_(self.model.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    def get_flat_params(self) -> torch.Tensor:
        """
        Get model parameters flattened into a single 1D tensor.
        - Full fine-tuning: Returns all parameters
        - LoRA: Returns only LoRA parameters (trainable parameters)
        
        Useful for Federated Learning aggregation.
        """
        if self.use_lora:
            return self._get_lora_params()
        else:
            return self._get_full_params()
    
    def _get_full_params(self) -> torch.Tensor:
        """Get all model parameters (full fine-tuning mode)."""
        # Use self.model.parameters() to access the actual model parameters
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _get_lora_params(self) -> torch.Tensor:
        """Get only LoRA parameters (LoRA fine-tuning mode)."""
        lora_params = []
        # Use self.model.parameters() to access the actual model parameters
        # In LoRA mode, only trainable parameters are LoRA params
        for param in self.model.parameters():
            if param.requires_grad:
                lora_params.append(param.data.view(-1))
        
        if not lora_params:
            # Fallback: if no trainable params found, return empty tensor
            # This shouldn't happen, but handle gracefully
            return torch.tensor([], dtype=torch.float32)
        
        return torch.cat(lora_params)

    def set_flat_params(self, flat_params: torch.Tensor):
        """
        Set model parameters from a single flattened 1D tensor.
        - Full fine-tuning: Sets all parameters
        - LoRA: Sets only LoRA parameters (trainable parameters)
        """
        if self.use_lora:
            self._set_lora_params(flat_params)
        else:
            self._set_full_params(flat_params)
    
    def _set_full_params(self, flat_params: torch.Tensor):
        """Set all model parameters (full fine-tuning mode)."""
        offset = 0
        # Use self.model.parameters() to access the actual model parameters
        for param in self.model.parameters():
            numel = param.numel()
            param.data.copy_(
                flat_params[offset:offset + numel].view(param.shape)
            )
            offset += numel
    
    def _set_lora_params(self, flat_params: torch.Tensor):
        """Set only LoRA parameters (LoRA fine-tuning mode)."""
        offset = 0
        # Use self.model.parameters() to maintain consistent order with _get_lora_params
        # Only update trainable parameters (LoRA params)
        for param in self.model.parameters():
            if param.requires_grad:
                numel = param.numel()
                if offset + numel > flat_params.numel():
                    raise ValueError(
                        f"Flat params size mismatch: trying to set {numel} params "
                        f"but only {flat_params.numel() - offset} remaining. "
                        f"Total needed: {offset + numel}, provided: {flat_params.numel()}"
                    )
                param.data.copy_(
                    flat_params[offset:offset + numel].view(param.shape)
                )
                offset += numel
        
        # Verify we used all parameters
        if offset != flat_params.numel():
            raise ValueError(
                f"Flat params size mismatch: used {offset} params "
                f"but {flat_params.numel()} provided. "
                f"Some LoRA parameters may not have been set."
            )


class GraphConvolutionLayer(nn.Module):
    """
    Simple Graph Convolution Layer (GCN).
    Formula: Output = A * X * W + b
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier Uniform."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Support = X * W
        support = torch.mm(x, self.weight)
        # Output = Adj * Support + b
        output = torch.mm(adj, support) + self.bias
        return output


class VGAE(nn.Module):
    """
    Variational Graph Autoencoder (VGAE) for GRMP attack.
    
    This model learns the relational structure among benign updates (as a graph)
    to generate adversarial gradients that mimic legitimate patterns.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        # --- Encoder Layers ---
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2_mu = GraphConvolutionLayer(hidden_dim, latent_dim)
        self.gc2_logvar = GraphConvolutionLayer(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes input features and adjacency matrix into latent distribution parameters."""
        
        # Normalize adjacency matrix (symmetric normalization)
        adj_norm = self._normalize_adj(adj)

        # Layer 1: GCN + ReLU + Dropout
        hidden = self.gc1(x, adj_norm)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        # Layer 2: Output Mean and Log Variance
        mu = self.gc2_mu(hidden, adj_norm)
        logvar = self.gc2_logvar(hidden, adj_norm)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        Allows backpropagation through stochastic nodes.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inner product decoder: reconstructs the adjacency matrix.
        A_pred = sigmoid(Z * Z^T)
        """
        adj_reconstructed = torch.sigmoid(torch.mm(z, z.t()))
        return adj_reconstructed

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed, mu, logvar

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetrically normalize adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2).
        Implementation handles self-loops by adding Identity matrix.
        """
        # Add self-loops
        adj_with_loop = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Calculate degree matrix D
        d_vec = adj_with_loop.sum(1)
        
        # Calculate D^(-1/2)
        d_inv_sqrt = torch.pow(d_vec, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # A_norm = D^(-1/2) * A * D^(-1/2)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_loop), d_mat_inv_sqrt)

    def loss_function(self, adj_reconstructed: torch.Tensor, adj_orig: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculates VGAE loss: Reconstruction Loss (Weighted BCE) + KL Divergence.
        """
        n_nodes = adj_orig.size(0)
        
        # Calculate weights for imbalanced classes (edges vs non-edges)
        # Typically graphs are sparse, so we weight positive edges more
        num_edges = adj_orig.sum().item()  # Convert to Python scalar
        num_non_edges = n_nodes * n_nodes - num_edges
        
        # Avoid division by zero
        if num_edges == 0:
            pos_weight = torch.tensor(1.0, device=adj_orig.device)
        else:
            pos_weight = torch.tensor(num_non_edges / num_edges, device=adj_orig.device)
            
        norm = (n_nodes * n_nodes) / (num_non_edges * 2) if num_non_edges > 0 else 1.0

        # 1. Reconstruction Loss (Weighted Binary Cross Entropy)
        bce_loss = norm * F.binary_cross_entropy_with_logits(
            adj_reconstructed, 
            adj_orig, 
            pos_weight=pos_weight
        )

        # 2. KL Divergence (Regularization term)
        # KL(N(mu, sigma) || N(0, 1))
        kl_loss = -0.5 / n_nodes * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        # Combine losses (KL term is often weighted less to prevent posterior collapse)
        return bce_loss + 0.1 * kl_loss