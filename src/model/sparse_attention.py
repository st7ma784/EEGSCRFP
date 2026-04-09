"""Transformer model with sparse attention control."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, Tuple, List
import torch.nn.functional as F


class SparseAttentionWrapper(nn.Module):
    """Wraps HuggingFace transformer and injects sparsity into attention."""
    
    def __init__(self, model_name: str = "gpt2", sparsity_type: str = "topk"):
        """
        Initialize sparse attention wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            sparsity_type: "topk" or "sparsemax"
        """
        super().__init__()
        self.model_name = model_name
        self.sparsity_type = sparsity_type
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.current_sparsity_level = 1.0  # Default: no sparsity
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """Register hooks to intercept attention computation."""
        self._attention_data = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                # Store raw attention for later analysis
                if isinstance(output, tuple):
                    attn_weights = output[0]
                    self._attention_data[name] = attn_weights.detach()
            return hook
        
        # Register hooks on attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'dropout'):
                module.register_forward_hook(attention_hook(name))
    
    def set_sparsity_level(self, sparsity_level: float):
        """
        Set sparsity level (0.0 = full sparse, 1.0 = no sparsity).
        
        Args:
            sparsity_level: Float between 0.0 and 1.0
        """
        assert 0.0 <= sparsity_level <= 1.0, "Sparsity level must be in [0, 1]"
        self.current_sparsity_level = sparsity_level
    
    def _apply_topk_sparsity(self, attention_weights: torch.Tensor, k_percent: float) -> torch.Tensor:
        """
        Apply top-k masking to attention weights.
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            k_percent: Fraction of weights to keep (sparsity_level)
            
        Returns:
            Sparse attention weights, renormalized
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Calculate k for each head
        k = max(1, int(seq_len * k_percent))
        
        # Flatten for topk operation
        flat_attn = attention_weights.view(batch_size * num_heads, seq_len, seq_len)
        
        # Apply topk per sequence position
        topk_vals, topk_indices = torch.topk(flat_attn, k=k, dim=-1)
        
        # Create sparse mask
        sparse_attn = torch.zeros_like(flat_attn)
        sparse_attn.scatter_(-1, topk_indices, topk_vals)
        
        # Renormalize
        sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        return sparse_attn.view(batch_size, num_heads, seq_len, seq_len)
    
    def _apply_sparsemax(self, attention_weights: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Apply entmax/sparsemax-like sparsity.
        
        For simplicity, we use a threshold approach:
        - Scale down activation based on alpha
        - Apply threshold and renormalize
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            alpha: Sparsity level
            
        Returns:
            Sparse attention weights
        """
        # Reduce entropy by sharpening
        temperature = 1.0 / (alpha + 0.1)  # Lower alpha = higher temperature = more entropy
        sharpened = attention_weights ** temperature
        
        # Renormalize
        normalized = sharpened / (sharpened.sum(dim=-1, keepdim=True) + 1e-8)
        
        return normalized
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = True,
        return_attention_maps: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sparsity injection.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_hidden_states: Whether to return all hidden states
            return_attention_maps: Whether to return attention maps
            
        Returns:
            Dictionary with:
                - 'last_hidden_state': [batch_size, seq_len, hidden_size]
                - 'hidden_states': list of hidden states if return_hidden_states
                - 'attention_maps': dict of sparse attention if return_attention_maps
        """
        # Clear attention data
        self._attention_data = {}
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state,
        }
        
        if return_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        
        if return_attention_maps:
            # Apply sparsity to attention weights
            sparse_attentions = []
            for layer_attn in outputs.attentions:
                if self.sparsity_type == "topk":
                    sparse_attn = self._apply_topk_sparsity(
                        layer_attn,
                        k_percent=self.current_sparsity_level
                    )
                else:  # sparsemax
                    sparse_attn = self._apply_sparsemax(
                        layer_attn,
                        alpha=self.current_sparsity_level
                    )
                sparse_attentions.append(sparse_attn)
            
            result['attention_maps'] = sparse_attentions
        
        return result


def create_sparse_model(model_name: str = "gpt2", sparsity_type: str = "topk"):
    """
    Create and return sparse attention wrapper.
    
    Args:
        model_name: HuggingFace model identifier
        sparsity_type: "topk" or "sparsemax"
        
    Returns:
        SparseAttentionWrapper instance
    """
    return SparseAttentionWrapper(model_name, sparsity_type)
