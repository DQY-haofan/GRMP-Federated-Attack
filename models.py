# models.py
# This module defines the NewsClassifierModel for News classification
# and the VGAE model used by AugMP (graph-augmented model manipulation).
#
# Supported Model Architectures:
# - Encoder-only (BERT-style): distilbert-base-uncased, bert-base-uncased, roberta-base, deberta-v3-base
# - Decoder-only (GPT-style): EleutherAI/pythia-160m, EleutherAI/pythia-1b, facebook/opt-125m, gpt2, Qwen/Qwen2.5-0.5B

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from typing import Tuple, Optional

# --- Constants ---
MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 4

# --- Model Architecture Detection ---
def get_model_architecture(model_name: str) -> str:
    """
    Detect model architecture type based on model name.
    
    Returns:
        'encoder': BERT-style bidirectional models
        'decoder': GPT-style causal/autoregressive models
        'encoder-decoder': T5-style seq2seq models
    """
    model_name_lower = model_name.lower()
    
    # Decoder-only models (GPT-style)
    decoder_patterns = ['pythia', 'gpt', 'opt-', 'llama', 'bloom', 'falcon', 'mistral', 'phi-', 'qwen']
    for pattern in decoder_patterns:
        if pattern in model_name_lower:
            return 'decoder'
    
    # Encoder-decoder models (T5-style)
    enc_dec_patterns = ['t5', 'bart', 'pegasus', 'marian']
    for pattern in enc_dec_patterns:
        if pattern in model_name_lower:
            return 'encoder-decoder'
    
    # Default: Encoder-only (BERT-style)
    return 'encoder'

# Optional LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("  Warning: peft library not available. LoRA support disabled. Install with: pip install peft")


class NewsClassifierModel(nn.Module):
    """
    Transformer-based model for news classification.
    Supports both Encoder-only (BERT-style) and Decoder-only (GPT-style) architectures.
    Supports both full fine-tuning and LoRA fine-tuning modes.
    Wraps the Hugging Face AutoModelForSequenceClassification.
    
    Supported Models:
        - Encoder-only: distilbert-base-uncased, bert-base-uncased, roberta-base, deberta-v3-base
        - Decoder-only: EleutherAI/pythia-160m, EleutherAI/pythia-1b, facebook/opt-125m, gpt2, Qwen/Qwen2.5-0.5B
    
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
        self.architecture = get_model_architecture(model_name)
        
        # Load base model
        # For decoder-only models, we need to set pad_token_id to avoid warnings
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # For decoder-only models (GPT-style), set pad_token_id if not set.
        # GPTNeoXConfig (e.g. Pythia) in transformers>=4.35 may not have pad_token_id at all; use getattr/setattr.
        if self.architecture == 'decoder':
            pad_id = getattr(self.model.config, 'pad_token_id', None)
            if pad_id is None:
                eos_id = getattr(self.model.config, 'eos_token_id', None)
                if eos_id is not None:
                    setattr(self.model.config, 'pad_token_id', eos_id)
        
        # Verify that the correct model is loaded
        model_type = type(self.model).__name__
        
        # Setup LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "LoRA support requires peft library. Install with: pip install peft"
                )
            
            # Default target modules based on model family
            if lora_target_modules is None:
                model_name_lower = model_name.lower()
                
                # ========== Decoder-only Models (GPT-style) ==========
                # Pythia / GPT-NeoX uses fused QKV attention + MLP layers
                # Standard LoRA configuration includes:
                # - query_key_value: Attention QKV fusion projection
                # - dense_h_to_4h: MLP up-projection (hidden → 4×hidden)
                # - dense_4h_to_h: MLP down-projection (4×hidden → hidden)
                if "pythia" in model_name_lower or "gpt-neox" in model_name_lower:
                    lora_target_modules = [
                        "query_key_value",      # Attention layer: QKV fusion projection
                        "dense_h_to_4h",       # MLP layer: up-projection (hidden → 4×hidden)
                        "dense_4h_to_h"        # MLP layer: down-projection (4×hidden → hidden)
                    ]
                # OPT uses separate projections
                elif "opt-" in model_name_lower or "/opt" in model_name_lower:
                    lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
                # GPT-2 uses c_attn (fused) and c_proj
                elif "gpt2" in model_name_lower:
                    lora_target_modules = ["c_attn", "c_proj"]
                # LLaMA / Mistral / Qwen2 style (shared architecture: q_proj, k_proj, v_proj, o_proj)
                elif "llama" in model_name_lower or "mistral" in model_name_lower or "qwen" in model_name_lower:
                    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                # Bloom
                elif "bloom" in model_name_lower:
                    lora_target_modules = ["query_key_value"]
                # Falcon
                elif "falcon" in model_name_lower:
                    lora_target_modules = ["query_key_value"]
                
                # ========== Encoder-only Models (BERT-style) ==========
                # DistilBERT uses these module names for attention layers
                elif "distilbert" in model_name_lower:
                    lora_target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
                # DeBERTa v2/v3 uses projection module names in attention
                elif "deberta" in model_name_lower:
                    lora_target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
                # BERT/RoBERTa style attention module names
                elif "bert" in model_name_lower or "roberta" in model_name_lower:
                    lora_target_modules = ["query", "key", "value", "dense"]
                else:
                    # Fallback: keep None and let PEFT raise a clearer error if unsupported
                    lora_target_modules = None
            
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
        """
        Initialize classifier weights to avoid initial bias.
        
        Note: Different model architectures use different classifier head names:
        - BERT-style (Encoder): 'classifier'
        - GPT-style (Decoder): 'score' (e.g., GPT2ForSequenceClassification, GPTNeoXForSequenceClassification)
        
        Decoder-only (GPT-NeoX/Pythia) uses smaller init to avoid large initial logits and loss=nan.
        """
        with torch.no_grad():
            classifier_names = ['classifier', 'score']
            # Decoder (Pythia/GPT-NeoX) is more sensitive: small init avoids gradient explosion / nan
            use_small_init = self.architecture == 'decoder'
            
            def _init_head(clf):
                if hasattr(clf, 'weight'):
                    if use_small_init:
                        nn.init.normal_(clf.weight, mean=0.0, std=0.02)
                    else:
                        nn.init.xavier_uniform_(clf.weight)
                if hasattr(clf, 'bias') and clf.bias is not None:
                    nn.init.zeros_(clf.bias)
            
            if self.use_lora and hasattr(self.model, 'base_model'):
                base_model = self.model.base_model.model
                for cls_name in classifier_names:
                    if hasattr(base_model, cls_name):
                        _init_head(getattr(base_model, cls_name))
                        break
            else:
                for cls_name in classifier_names:
                    if hasattr(self.model, cls_name):
                        _init_head(getattr(self.model, cls_name))
                        break

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    def get_flat_params(self, requires_grad: bool = False) -> torch.Tensor:
        """
        Get model parameters flattened into a single 1D tensor.
        - Full fine-tuning: Returns all parameters
        - LoRA: Returns only LoRA parameters (trainable parameters)
        
        Args:
            requires_grad: If True, preserve gradients (for training). If False, detach (for aggregation).
        
        Useful for Federated Learning aggregation.
        """
        if self.use_lora:
            return self._get_lora_params(requires_grad=requires_grad)
        else:
            return self._get_full_params(requires_grad=requires_grad)
    
    def _get_full_params(self, requires_grad: bool = False) -> torch.Tensor:
        """Get all model parameters (full fine-tuning mode)."""
        # Use self.model.parameters() to access the actual model parameters
        if requires_grad:
            # Preserve gradients for training (e.g., proximal regularization)
            return torch.cat([p.view(-1) for p in self.model.parameters()])
        else:
            # Detach for aggregation/updates
            return torch.cat([p.data.view(-1) for p in self.model.parameters()])
    
    def _get_lora_params(self, requires_grad: bool = False) -> torch.Tensor:
        """Get only LoRA parameters (LoRA fine-tuning mode)."""
        lora_params = []
        # Use self.model.parameters() to access the actual model parameters
        # In LoRA mode, only trainable parameters are LoRA params
        for param in self.model.parameters():
            if param.requires_grad:
                if requires_grad:
                    # Preserve gradients for training (e.g., proximal regularization)
                    lora_params.append(param.view(-1))
                else:
                    # Detach for aggregation/updates
                    lora_params.append(param.data.view(-1))
        
        if not lora_params:
            # No trainable LoRA parameters found - this indicates a configuration error
            raise RuntimeError(
                "No trainable LoRA parameters found. "
                "Please check LoRA configuration (target_modules, r, etc.)."
            )
        
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
                # Get the parameter slice
                param_slice = flat_params[offset:offset + numel].view(param.shape)
                # CRITICAL: Ensure param_slice is on the same device as param
                # This prevents device mismatch errors, especially when flat_params is on CPU
                # but param is on GPU (or vice versa)
                if param_slice.device != param.device:
                    param_slice = param_slice.to(param.device)
                # Ensure dtype matches
                if param_slice.dtype != param.dtype:
                    param_slice = param_slice.to(dtype=param.dtype)
                param.data.copy_(param_slice)
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
    Variational Graph Autoencoder (VGAE) for AugMP.
    
    This model learns the relational structure among benign updates (as a graph)
    to produce graph-conditioned directions aligned with benign update patterns.
    
    Standard VGAE architecture:
    - Encoder: Two-layer GCN that outputs mean (μ) and log variance (log σ²)
    - Reparameterization: z = μ + σ * ε (where ε ~ N(0,1))
    - Decoder: Inner product decoder for adjacency matrix reconstruction
    - Loss: L = L_recon + β * KL(q(z|X,A) || p(z))
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32, 
                 dropout: float = 0.2, kl_weight: float = 0.1):
        """
        Initialize VGAE model.
        
        Args:
            input_dim: Input feature dimension (number of clients/benign models)
            hidden_dim: Hidden layer dimension (default: 64)
            latent_dim: Latent space dimension (default: 32)
            dropout: Dropout rate (default: 0.2)
            kl_weight: Weight for KL divergence term in loss function (default: 0.1)
                       Lower values prevent posterior collapse, higher values enforce
                       stronger regularization toward standard normal distribution.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.kl_weight = kl_weight
        
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
        Returns logits (before sigmoid) for use with binary_cross_entropy_with_logits.
        A_pred = Z * Z^T (logits)
        
        Note: Apply sigmoid if probabilities are needed (e.g., for GSP module).
        """
        adj_reconstructed = torch.mm(z, z.t())  # Return logits, not probabilities
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
        
        Standard VGAE loss formulation:
            L = L_recon + β * KL(q(z|X,A) || p(z))
        
        where:
            - L_recon: Weighted binary cross-entropy for adjacency matrix reconstruction
            - KL: KL divergence from approximate posterior q(z|X,A) to prior p(z) = N(0,1)
            - β: Weighting factor (self.kl_weight) to balance reconstruction and regularization
        
        Args:
            adj_reconstructed: Reconstructed adjacency matrix from decoder
            adj_orig: Original adjacency matrix
            mu: Mean of latent distribution (from encoder), shape: (n_nodes, latent_dim)
            logvar: Log variance of latent distribution (from encoder), shape: (n_nodes, latent_dim)
        
        Returns:
            Total VGAE loss (scalar tensor)
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
        # Formula: -[y*log(σ(x)) + (1-y)*log(1-σ(x))] with pos_weight for class imbalance
        bce_loss = norm * F.binary_cross_entropy_with_logits(
            adj_reconstructed, 
            adj_orig, 
            pos_weight=pos_weight
        )

        # 2. KL Divergence (Regularization term)
        # Standard formula: KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ[1 + log(σ²) - μ² - σ²]
        # where logvar = log(σ²), so σ² = exp(logvar)
        # Per node: sum over latent dimensions, then average over all nodes
        kl_per_node = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_per_node)  # Average over all nodes (standard VGAE implementation)

        # Combine losses: L = L_recon + β * KL
        # β (kl_weight) balances reconstruction quality vs. regularization strength
        return bce_loss + self.kl_weight * kl_loss