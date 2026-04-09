"""MLP predictor for sparsity levels."""
import torch
import torch.nn as nn
from typing import List, Optional


class MLPPredictor(nn.Module):
    """MLP to predict sparsity level from input features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize MLP predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (usually 1 for regression)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim]
        """
        return self.net(x)


def create_predictor(
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
) -> MLPPredictor:
    """
    Create and return MLP predictor.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        
    Returns:
        MLPPredictor instance
    """
    return MLPPredictor(input_dim, hidden_dims, output_dim=1, dropout=dropout)
