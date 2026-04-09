"""Transformer model with sparse attention control."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, List


class SparseAttentionWrapper(nn.Module):
    """Wraps a HuggingFace transformer and injects sparsity into attention.

    GPU-friendly design notes
    -------------------------
    * The sparsity method (`topk` vs `sparsemax`) is resolved once at
      construction time and stored as ``self._sparsify`` — no branch inside
      ``forward()``.
    * All per-layer attention tensors are stacked into ``[L*B, H, S, S]``
      before sparsification so a single kernel dispatch handles every layer.
    """

    def __init__(self, model_name: str = "gpt2", sparsity_type: str = "topk"):
        super().__init__()
        self.model_name = model_name
        self.sparsity_type = sparsity_type

        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.current_sparsity_level = 1.0

        # Resolve method at construction time — no if-branch in forward()
        if sparsity_type == "topk":
            self._sparsify = self._apply_topk_sparsity
        else:
            self._sparsify = self._apply_sparsemax

        self._register_attention_hooks()

    # ------------------------------------------------------------------
    # Attention hooks
    # ------------------------------------------------------------------

    def _register_attention_hooks(self):
        """Register forward hooks to capture raw attention weights."""
        self._attention_data: Dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self._attention_data[name] = output[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "dropout"):
                module.register_forward_hook(make_hook(name))

    # ------------------------------------------------------------------
    # Sparsity methods
    # ------------------------------------------------------------------

    def set_sparsity_level(self, sparsity_level: float):
        assert 0.0 <= sparsity_level <= 1.0
        self.current_sparsity_level = sparsity_level

    @staticmethod
    def _apply_topk_sparsity(
        attn: torch.Tensor,
        k_percent: float,
    ) -> torch.Tensor:
        """Top-k masking on ``[*, H, S, S]`` — fully batched.

        Works on any leading-batch shape by flattening everything except the
        last two (query, key) dimensions.
        """
        *leading, H, S, _ = attn.shape
        flat = attn.reshape(-1, S, S)          # [*, S, S] → [N, S, S]
        k = max(1, int(S * k_percent))

        topk_vals, topk_idx = torch.topk(flat, k=k, dim=-1)   # [N, S, k]
        sparse = torch.zeros_like(flat)
        sparse.scatter_(-1, topk_idx, topk_vals)
        sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse.reshape(*leading, H, S, S)

    @staticmethod
    def _apply_sparsemax(
        attn: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Temperature-sharpened sparsemax approximation — fully batched."""
        temperature = 1.0 / (alpha + 0.1)
        sharpened = attn ** temperature
        return sharpened / (sharpened.sum(dim=-1, keepdim=True) + 1e-8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attention_maps: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with sparsity injection.

        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            return_hidden_states: include all hidden states in output
            return_attention_maps: apply sparsity and include attention maps

        Returns:
            dict with ``last_hidden_state``, optionally ``hidden_states``
            and ``attention_maps`` (list of L tensors [B, H, S, S]).
        """
        self._attention_data = {}

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )

        result: Dict[str, object] = {"last_hidden_state": outputs.last_hidden_state}

        if return_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        if return_attention_maps:
            # Stack all layers → [L, B, H, S, S], process as one batch
            stacked = torch.stack(outputs.attentions, dim=0)   # [L, B, H, S, S]
            L, B, H, S, _ = stacked.shape

            # Merge L and B into one batch dim for a single kernel dispatch
            merged = stacked.reshape(L * B, H, S, S)
            sparse_merged = self._sparsify(merged, self.current_sparsity_level)
            sparse_stacked = sparse_merged.reshape(L, B, H, S, S)

            # Unstack back to list of [B, H, S, S] — same API as before
            result["attention_maps"] = list(sparse_stacked.unbind(0))

        return result


def create_sparse_model(model_name: str = "gpt2", sparsity_type: str = "topk"):
    return SparseAttentionWrapper(model_name, sparsity_type)
