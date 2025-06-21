# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
import numpy as np


class NewsClassifierModel(nn.Module):
    """DistilBERT-based model for 4-class news classification"""

    def __init__(self):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=4  # AG News has 4 classes
        )

        # Initialize classifier weights to avoid initial bias
        with torch.no_grad():
            nn.init.xavier_uniform_(self.model.classifier.weight)
            nn.init.zeros_(self.model.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    def get_flat_params(self):
        """Get flattened model parameters"""
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        """Set model parameters from flattened tensor"""
        offset = 0
        for param in self.parameters():
            param_length = param.numel()
            param.data = flat_params[offset:offset + param_length].view(param.shape)
            offset += param_length


class GraphConvolutionLayer(nn.Module):
    """Graph Convolution Layer for VGAE"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support) + self.bias
        return output


class VGAE(nn.Module):
    """Variational Graph Autoencoder for GRMP attack

    This model learns the relational structure among benign updates
    to generate adversarial gradients that mimic legitimate patterns
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()

        # Encoder
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2_mu = GraphConvolutionLayer(hidden_dim, latent_dim)
        self.gc2_logvar = GraphConvolutionLayer(hidden_dim, latent_dim)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def encode(self, x, adj):
        """Encode to latent space"""
        # Normalize adjacency matrix
        adj_norm = self._normalize_adj(adj)

        # First layer with ReLU activation
        hidden = F.relu(self.gc1(x, adj_norm))
        hidden = self.dropout(hidden)

        # Get mean and log variance
        mu = self.gc2_mu(hidden, adj_norm)
        logvar = self.gc2_logvar(hidden, adj_norm)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        """Decode from latent space (inner product decoder)"""
        adj_reconstructed = torch.sigmoid(torch.mm(z, z.t()))
        return adj_reconstructed

    def forward(self, x, adj):
        """Full forward pass"""
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed, mu, logvar

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix"""
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        d = adj.sum(1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def loss_function(self, adj_reconstructed, adj_orig, mu, logvar):
        """VGAE loss = Reconstruction loss + KL divergence"""
        # Reconstruction loss (binary cross entropy)
        n_nodes = adj_orig.size(0)
        pos_weight = (n_nodes * n_nodes - adj_orig.sum()) / adj_orig.sum()
        norm = n_nodes * n_nodes / ((n_nodes * n_nodes - adj_orig.sum()) * 2)

        # Weighted BCE loss
        bce_loss = norm * F.binary_cross_entropy_with_logits(
            adj_reconstructed, adj_orig, pos_weight=pos_weight
        )

        # KL divergence
        kl_loss = -0.5 / n_nodes * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        return bce_loss + 0.1 * kl_loss  # Weight KL term less