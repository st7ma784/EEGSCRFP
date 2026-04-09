"""Compute pathway metrics from attention maps."""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import entr
from scipy.stats import entropy


class PathwayMetricsComputer:
    """Compute routing/pathway metrics from attention weights."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def routing_sparsity(attention_maps: List[torch.Tensor]) -> float:
        """
        Compute effective number of active paths (routing sparsity).
        
        Uses the Leinster-Cobbold diversity index on attention weights.
        High value = more distributed (less sparse), Low value = concentrated (sparse)
        
        Args:
            attention_maps: List of attention tensors, shape [batch, heads, seq_len, seq_len]
            
        Returns:
            Scalar measure of routing sparsity
        """
        all_attentions = []
        for attn in attention_maps:
            # Average over batch and sequence positions
            # Shape: [batch, heads, seq_len, seq_len] -> [heads, seq_len, seq_len]
            attn_mean = attn.mean(dim=0)  # Average over batch
            all_attentions.append(attn_mean)
        
        # Flatten and compute effective number of active paths
        all_flat = torch.cat([a.flatten() for a in all_attentions])
        
        # Remove near-zero values
        all_flat = all_flat[all_flat > 1e-5]
        
        if len(all_flat) == 0:
            return 0.0
        
        # Compute diversity (effective number of paths)
        # Using Rényi entropy of order 2
        probs = all_flat / all_flat.sum()
        renyi_entropy = -torch.log(torch.sum(probs ** 2) + 1e-10)
        effective_paths = torch.exp(renyi_entropy).item()
        
        return float(effective_paths)
    
    @staticmethod
    def path_competition_index(attention_maps: List[torch.Tensor]) -> float:
        """
        Compute path competition index: max(attention) / mean(attention).
        
        High = strong winner (sparse), Low = distributed (dense)
        
        Args:
            attention_maps: List of attention tensors
            
        Returns:
            Scalar competition index
        """
        max_vals = []
        mean_vals = []
        
        for attn in attention_maps:
            # Focus on non-zero attention values
            attn_mean = attn.mean(dim=0)  # [heads, seq_len, seq_len]
            attn_flat = attn_mean.flatten()
            attn_flat = attn_flat[attn_flat > 1e-5]
            
            if len(attn_flat) > 0:
                max_vals.append(attn_flat.max().item())
                mean_vals.append(attn_flat.mean().item())
        
        if not max_vals or not mean_vals:
            return 0.0
        
        avg_max = np.mean(max_vals)
        avg_mean = np.mean(mean_vals)
        
        if avg_mean < 1e-8:
            return 0.0
        
        competition = avg_max / avg_mean
        return float(competition)
    
    @staticmethod
    def path_efficiency(attention_maps: List[torch.Tensor], topk_percent: float = 0.2) -> float:
        """
        Compute path efficiency: energy concentration in top-k weights.
        
        Measure how much attention weight is concentrated in a small set of connections.
        Higher = more concentrated = more sparse
        
        Args:
            attention_maps: List of attention tensors
            topk_percent: Fraction of top weights to consider (default 20%)
            
        Returns:
            Scalar efficiency measure
        """
        efficiencies = []
        
        for attn in attention_maps:
            attn_mean = attn.mean(dim=0)  # [heads, seq_len, seq_len]
            attn_flat = attn_mean.flatten()
            
            if len(attn_flat) == 0:
                continue
            
            # Get top-k values
            k = max(1, int(len(attn_flat) * topk_percent))
            topk_vals = torch.topk(attn_flat, k=k)[0]
            
            # Efficiency = mass in top-k / total mass
            efficiency = topk_vals.sum().item() / attn_flat.sum().item()
            efficiencies.append(efficiency)
        
        return float(np.mean(efficiencies)) if efficiencies else 0.0
    
    @staticmethod
    def routing_entropy(attention_maps: List[torch.Tensor]) -> float:
        """
        Compute Shannon entropy of attention distributions.
        
        High = distributed (low sparsity), Low = concentrated (high sparsity)
        
        Args:
            attention_maps: List of attention tensors
            
        Returns:
            Scalar entropy value
        """
        entropies = []
        
        for attn in attention_maps:
            attn_mean = attn.mean(dim=0)  # [heads, seq_len, seq_len]
            
            # Compute per-position entropies
            for pos_attn in attn_mean:  # Iterate over sequences
                for head_attn in pos_attn:  # Iterate over sequence positions
                    if head_attn.sum() > 1e-8:
                        probs = head_attn / head_attn.sum()
                        probs = probs[probs > 1e-10]
                        if len(probs) > 0:
                            h = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                            entropies.append(h)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    @staticmethod
    def inter_head_divergence(attention_maps: List[torch.Tensor]) -> float:
        """
        Compute KL divergence between attention heads (averaged over layers).
        
        High = heads diverge (different routes), Low = heads agree (similar routes)
        
        Args:
            attention_maps: List of attention tensors [batch, heads, seq_len, seq_len]
            
        Returns:
            Scalar divergence measure
        """
        divergences = []
        
        for attn in attention_maps:  # Per layer
            attn_mean = attn.mean(dim=0)  # Average over batch: [heads, seq_len, seq_len]
            num_heads = attn_mean.shape[0]
            
            if num_heads < 2:
                continue
            
            # Compute pairwise KL divergences between heads
            head_pairs = []
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    h1 = attn_mean[i].flatten()
                    h2 = attn_mean[j].flatten()
                    
                    # Normalize
                    p = h1 / (h1.sum() + 1e-10)
                    q = h2 / (h2.sum() + 1e-10)
                    
                    # KL(p || q)
                    kl = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).item()
                    head_pairs.append(kl)
            
            if head_pairs:
                divergences.append(np.mean(head_pairs))
        
        return float(np.mean(divergences)) if divergences else 0.0
    
    @staticmethod
    def layer_stability(attention_maps: List[torch.Tensor]) -> float:
        """
        Compute cosine similarity between consecutive layer attention maps.
        
        High = smooth transitions between layers, Low = abrupt changes
        
        Args:
            attention_maps: List of attention tensors
            
        Returns:
            Scalar stability measure
        """
        similarities = []
        
        for i in range(len(attention_maps) - 1):
            attn_curr = attention_maps[i].mean(dim=0).flatten()  # [heads*seq*seq]
            attn_next = attention_maps[i + 1].mean(dim=0).flatten()
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(
                attn_curr.unsqueeze(0),
                attn_next.unsqueeze(0)
            ).item()
            similarities.append(cos_sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def compute_all_metrics(self, attention_maps: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute all pathway metrics.
        
        Args:
            attention_maps: List of attention tensors from all layers
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'routing_sparsity': self.routing_sparsity(attention_maps),
            'path_competition_index': self.path_competition_index(attention_maps),
            'path_efficiency': self.path_efficiency(attention_maps),
            'routing_entropy': self.routing_entropy(attention_maps),
            'inter_head_divergence': self.inter_head_divergence(attention_maps),
            'layer_stability': self.layer_stability(attention_maps),
        }
        return metrics


def compute_pathway_features(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pathway feature vector from attention maps.
    
    Args:
        attention_maps: List of attention tensors from all layers
        
    Returns:
        Feature vector of shape [6] containing all metrics
    """
    computer = PathwayMetricsComputer()
    metrics = computer.compute_all_metrics(attention_maps)
    
    # Create feature vector in consistent order
    feature_vector = torch.tensor([
        metrics['routing_sparsity'],
        metrics['path_competition_index'],
        metrics['path_efficiency'],
        metrics['routing_entropy'],
        metrics['inter_head_divergence'],
        metrics['layer_stability'],
    ], dtype=torch.float32)
    
    return feature_vector
