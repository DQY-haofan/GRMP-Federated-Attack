# models.py
# This module defines the NewsClassifierModel for AG News classification
# and the VGAE model for GRMP attack.


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from typing import Tuple

# --- Constants ---
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 4


class NewsClassifierModel(nn.Module):
    """
    DistilBERT-based model for 4-class news classification.
    Wraps the Hugging Face DistilBertForSequenceClassification model.
    """

    def __init__(self, model_name: str = DISTILBERT_MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize classifier weights to avoid initial bias."""
        with torch.no_grad():
            if hasattr(self.model, 'classifier'):
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
        Get all model parameters flattened into a single 1D tensor.
        Useful for Federated Learning aggregation.
        """
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params: torch.Tensor):
        """
        Set model parameters from a single flattened 1D tensor.
        """
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            # Copy data from flat_params to param.data
            param.data.copy_(
                flat_params[offset:offset + numel].view(param.shape)
            )
            offset += numel


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
        num_edges = adj_orig.sum()
        num_non_edges = n_nodes * n_nodes - num_edges
        
        # Avoid division by zero
        if num_edges == 0:
            pos_weight = torch.tensor(1.0, device=adj_orig.device)
        else:
            pos_weight = num_non_edges / num_edges
            
        norm = (n_nodes * n_nodes) / ((num_non_edges) * 2)

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