"""Quick demo script to test all components."""
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import get_default_config
from src.model.sparse_attention import create_sparse_model
from src.data.dataset import create_default_prompts
from src.data.tokenizer import TextTokenizer
from src.metrics.pathway_metrics import compute_pathway_features
from src.projection.eeg_projector import EEGProjector
from src.predictor.mlp import MLPPredictor


def test_sparse_attention_model():
    """Test sparse attention wrapper."""
    print("\n" + "="*80)
    print("TEST 1: Sparse Attention Model")
    print("="*80)
    
    model = create_sparse_model(model_name="gpt2", sparsity_type="topk")
    tokenizer = TextTokenizer()
    
    prompt = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.tokenize([prompt])
    
    print(f"Prompt: {prompt}")
    print(f"Input shape: {tokens['input_ids'].shape}")
    
    for sparsity_level in [0.3, 0.6, 0.9]:
        model.set_sparsity_level(sparsity_level)
        output = model(
            tokens['input_ids'],
            return_attention_maps=True,
            return_hidden_states=False,
        )
        print(f"\n  Sparsity level: {sparsity_level}")
        print(f"  Output hidden state shape: {output['last_hidden_state'].shape}")
        print(f"  Number of attention layers: {len(output['attention_maps'])}")
        print(f"  First layer attention shape: {output['attention_maps'][0].shape}")
    
    print("\n✓ Sparse attention model works!")
    return model, tokenizer


def test_pathway_metrics(model, tokenizer):
    """Test pathway metrics computation."""
    print("\n" + "="*80)
    print("TEST 2: Pathway Metrics")
    print("="*80)
    
    prompt = "Machine learning is fascinating."
    tokens = tokenizer.tokenize([prompt])
    
    model.set_sparsity_level(0.5)
    output = model(tokens['input_ids'], return_attention_maps=True)
    
    features = compute_pathway_features(output['attention_maps'])
    
    metric_names = [
        'Routing Sparsity',
        'Path Competition Index',
        'Path Efficiency',
        'Routing Entropy',
        'Inter-head Divergence',
        'Layer-wise Stability',
    ]
    
    print(f"\nMetrics for prompt: '{prompt}'")
    print(f"Feature vector shape: {features.shape}")
    for i, (name, value) in enumerate(zip(metric_names, features.tolist())):
        print(f"  {i+1}. {name:.<30} {value:.4f}")
    
    print("\n✓ Pathway metrics extraction works!")
    return features


def test_eeg_projector(features):
    """Test EEG projection."""
    print("\n" + "="*80)
    print("TEST 3: EEG Projection")
    print("="*80)
    
    projector = EEGProjector(
        input_dim=6,
        output_channels=105,
        add_noise=False,
    )
    
    eeg_signal = projector(features.unsqueeze(0), training=False)
    
    print(f"\nPathway features shape: {features.shape}")
    print(f"EEG signal shape: {eeg_signal.shape}")
    print(f"EEG signal stats:")
    print(f"  Mean: {eeg_signal.mean().item():.4f}")
    print(f"  Std: {eeg_signal.std().item():.4f}")
    print(f"  Min: {eeg_signal.min().item():.4f}")
    print(f"  Max: {eeg_signal.max().item():.4f}")
    
    print("\n✓ EEG projection works!")
    return eeg_signal


def test_predictor(features, eeg_signal):
    """Test MLP predictor."""
    print("\n" + "="*80)
    print("TEST 4: MLP Predictor")
    print("="*80)
    
    pathway_predictor = MLPPredictor(input_dim=6, hidden_dims=[128, 64])
    eeg_predictor = MLPPredictor(input_dim=105, hidden_dims=[128, 64])
    
    pathway_pred = pathway_predictor(features.unsqueeze(0))
    eeg_pred = eeg_predictor(eeg_signal)
    
    print(f"\nPathway predictor:")
    print(f"  Input shape: {features.unsqueeze(0).shape}")
    print(f"  Output shape: {pathway_pred.shape}")
    print(f"  Prediction: {pathway_pred.item():.4f}")
    
    print(f"\nEEG predictor:")
    print(f"  Input shape: {eeg_signal.shape}")
    print(f"  Output shape: {eeg_pred.shape}")
    print(f"  Prediction: {eeg_pred.item():.4f}")
    
    print("\n✓ MLP predictors work!")


def test_sparsity_levels():
    """Test different sparsity levels."""
    print("\n" + "="*80)
    print("TEST 5: Sparsity Level Effects")
    print("="*80)
    
    model = create_sparse_model()
    tokenizer = TextTokenizer()
    
    prompt = "Testing sparsity effects."
    tokens = tokenizer.tokenize([prompt])
    
    print(f"\nPrompt: '{prompt}'")
    print(f"\nMetrics across sparsity levels:")
    print(f"{'Sparsity':<12} {'Routing Sparsity':<20} {'Path Competition':<20}")
    print("-" * 52)
    
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for sparsity_level in sparsity_levels:
        model.set_sparsity_level(sparsity_level)
        output = model(tokens['input_ids'], return_attention_maps=True)
        features = compute_pathway_features(output['attention_maps'])
        
        routing_sparsity = features[0].item()
        path_competition = features[1].item()
        
        print(f"{sparsity_level:<12.1f} {routing_sparsity:<20.4f} {path_competition:<20.4f}")
    
    print("\n✓ Sparsity level effects are observable!")


def test_config():
    """Test configuration."""
    print("\n" + "="*80)
    print("TEST 6: Configuration")
    print("="*80)
    
    config = get_default_config()
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Sparsity type: {config.sparsity.sparsity_type}")
    print(f"  EEG channels: {config.projection.output_channels}")
    print(f"  Sparsity levels: {config.sparsity.sparsity_levels}")
    print(f"  Training epochs: {config.training.max_epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    
    print("\n✓ Configuration loaded successfully!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EEG CAUSAL HYPOTHESIS PROTOTYPE - QUICK DEMO")
    print("="*80)
    
    try:
        # Test 6: Config
        test_config()
        
        # Test 1: Sparse attention model
        model, tokenizer = test_sparse_attention_model()
        
        # Test 2: Pathway metrics
        features = test_pathway_metrics(model, tokenizer)
        
        # Test 3: EEG projection
        eeg_signal = test_eeg_projector(features)
        
        # Test 4: Predictors
        test_predictor(features, eeg_signal)
        
        # Test 5: Sparsity effects
        test_sparsity_levels()
        
        # Summary
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the full system with:")
        print("  python main.py --epochs 5 --batch-size 8 --num-prompts 3")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
