"""Project pathway features to EEG-like signals."""
import torch
import torch.nn as nn
from typing import Optional


class EEGProjector(nn.Module):
    """Project pathway features to EEG-like output."""
    
    def __init__(
        self,
        input_dim: int = 6,  # 6 pathway metrics
        output_channels: int = 105,  # Standard EEG channel count
        add_noise: bool = True,
        noise_std: float = 0.1,
        smoothing_window: Optional[int] = None,
    ):
        """
        Initialize EEG projector.
        
        Args:
            input_dim: Number of input pathway features
            output_channels: Number of EEG channels
            add_noise: Whether to add Gaussian noise
            noise_std: Standard deviation of noise
            smoothing_window: Optional smoothing window size
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.smoothing_window = smoothing_window
        
        # Linear projection: pathway_metrics -> EEG_channels
        self.projection = nn.Linear(input_dim, output_channels, bias=True)
        
        # Initialize with Xavier initialization
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, pathway_features: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Project pathway features to EEG signals.
        
        Args:
            pathway_features: [batch_size, input_dim]
            training: Whether in training mode (controls noise injection)
            
        Returns:
            EEG signals [batch_size, output_channels]
        """
        # Linear projection
        eeg_signal = self.projection(pathway_features)  # [batch, channels]
        
        # Add noise during training
        if self.add_noise and training:
            noise = torch.randn_like(eeg_signal) * self.noise_std
            eeg_signal = eeg_signal + noise
        
        # Optional smoothing (temporal smoothing across simulated time)
        if self.smoothing_window is not None and self.smoothing_window > 1:
            eeg_signal = self._apply_smoothing(eeg_signal)
        
        return eeg_signal
    
    def _apply_smoothing(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal smoothing to EEG signal.
        
        Args:
            signal: [batch_size, channels]
            
        Returns:
            Smoothed signal [batch_size, channels]
        """
        # Simple moving average smoothing
        # Treat as if we have a temporal dimension
        window = self.smoothing_window
        if window is None or window < 2:
            return signal
        
        # Add temporal dimension for convolution
        signal = signal.unsqueeze(1)  # [batch, 1, channels]
        signal = signal.transpose(1, 2)  # [batch, channels, 1]
        
        # Apply 1D convolution for smoothing
        kernel = torch.ones(self.output_channels, 1, window) / window
        kernel = kernel.to(signal.device)
        
        # Pad input
        signal_padded = torch.nn.functional.pad(signal, (window // 2, window // 2))
        
        # Apply convolution per channel
        smoothed = torch.nn.functional.conv1d(
            signal_padded,
            kernel,
            groups=self.output_channels
        )
        
        return smoothed.squeeze(-1)  # [batch, channels]


def project_to_eeg(
    pathway_features: torch.Tensor,
    projector: EEGProjector,
    training: bool = True,
) -> torch.Tensor:
    """
    Convenience function to project pathway features to EEG.
    
    Args:
        pathway_features: [batch_size, num_features]
        projector: EEGProjector instance
        training: Whether in training mode
        
    Returns:
        EEG signals [batch_size, num_channels]
    """
    return projector(pathway_features, training=training)
