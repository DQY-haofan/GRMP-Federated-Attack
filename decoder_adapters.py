# decoder_adapters.py
# Pluggable transfer of backbone weights from HF SeqCLS (Fed fine-tuned) to CausalLM for generate().

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Type

import torch.nn as nn


class DecoderAdapter(ABC):
    """Maps a fine-tuned ``ForSequenceClassification`` inner model into a ``ForCausalLM`` backbone."""

    @staticmethod
    @abstractmethod
    def matches(model_name: str) -> bool:
        """Return True if this adapter handles ``model_name`` (HF id or path)."""

    @abstractmethod
    def transfer_backbone(
        self,
        seq_cls_inner: nn.Module,
        causal_lm: nn.Module,
    ) -> None:
        """
        Copy shared backbone weights from the classification model into ``causal_lm``.

        ``seq_cls_inner`` is typically ``GPTNeoXForSequenceClassification`` (possibly merged from PEFT).
        ``causal_lm`` is ``GPTNeoXForCausalLM``. ``lm_head`` stays as loaded from ``from_pretrained``.
        """


class PythiaNeoXAdapter(DecoderAdapter):
    """EleutherAI Pythia / GPT-NeoX sequence-classification -> causal LM."""

    _NEEDLE = "pythia"

    @staticmethod
    def matches(model_name: str) -> bool:
        m = (model_name or "").lower()
        return PythiaNeoXAdapter._NEEDLE in m or "gpt-neox" in m

    def transfer_backbone(self, seq_cls_inner: nn.Module, causal_lm: nn.Module) -> None:
        inner = seq_cls_inner
        if hasattr(inner, "merge_and_unload"):
            inner = inner.merge_and_unload()

        src_sd = inner.state_dict()
        dst_sd = causal_lm.state_dict()
        to_load = {}
        for k, v in src_sd.items():
            if not k.startswith("gpt_neox."):
                continue
            if k not in dst_sd or dst_sd[k].shape != v.shape:
                continue
            to_load[k] = v.to(device=dst_sd[k].device, dtype=dst_sd[k].dtype)

        if not to_load:
            raise RuntimeError(
                "PythiaNeoXAdapter: no gpt_neox.* keys matched between SeqCLS and CausalLM. "
                "Check model_name and transformers versions."
            )
        causal_lm.load_state_dict(to_load, strict=False)


class Qwen2Adapter(DecoderAdapter):
    """Qwen2 / Qwen2.5 sequence-classification -> causal LM (shared ``model.*`` backbone)."""

    @staticmethod
    def matches(model_name: str) -> bool:
        m = (model_name or "").lower()
        return "qwen2" in m

    def transfer_backbone(self, seq_cls_inner: nn.Module, causal_lm: nn.Module) -> None:
        inner = seq_cls_inner
        if hasattr(inner, "merge_and_unload"):
            inner = inner.merge_and_unload()

        src_sd = inner.state_dict()
        dst_sd = causal_lm.state_dict()
        to_load = {}
        for k, v in src_sd.items():
            if not k.startswith("model."):
                continue
            if k not in dst_sd or dst_sd[k].shape != v.shape:
                continue
            to_load[k] = v.to(device=dst_sd[k].device, dtype=dst_sd[k].dtype)

        if not to_load:
            raise RuntimeError(
                "Qwen2Adapter: no model.* keys matched between SeqCLS and CausalLM. "
                "Check model_name and transformers versions."
            )
        causal_lm.load_state_dict(to_load, strict=False)


# Registry: first match wins (order matters for overlapping patterns).
ADAPTER_REGISTRY: List[Type[DecoderAdapter]] = [
    Qwen2Adapter,
    PythiaNeoXAdapter,
]


def resolve_adapter(model_name: str) -> DecoderAdapter:
    """
    Select an adapter for the given Hugging Face model id.

    Raises:
        ValueError: If no registered adapter matches.
    """
    for cls in ADAPTER_REGISTRY:
        if cls.matches(model_name):
            return cls()
    registered = ", ".join(c.__name__ for c in ADAPTER_REGISTRY)
    raise ValueError(
        f"No DecoderAdapter registered for model_name={model_name!r}. "
        f"Implement a new adapter class, append it to ADAPTER_REGISTRY in decoder_adapters.py, "
        f"and implement transfer_backbone for that architecture. "
        f"Currently registered: {registered}"
    )
