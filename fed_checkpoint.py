# fed_checkpoint.py
# Save global NewsClassifierModel after federated learning for downstream experiments.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from models import NewsClassifierModel, get_model_architecture


def _build_checkpoint_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    model_name = config.get("model_name", "distilbert-base-uncased")
    use_lora = bool(config.get("use_lora", False))
    meta: Dict[str, Any] = {
        "model_name": model_name,
        "num_labels": int(config.get("num_labels", 4)),
        "use_lora": use_lora,
        "architecture": get_model_architecture(model_name),
        "experiment_name": config.get("experiment_name", "experiment"),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if use_lora:
        meta["lora_r"] = config.get("lora_r", 16)
        meta["lora_alpha"] = config.get("lora_alpha", 32)
        meta["lora_dropout"] = config.get("lora_dropout", 0.1)
        tm = config.get("lora_target_modules")
        meta["lora_target_modules"] = tm if tm is None else list(tm)
    return meta


def save_global_model_checkpoint(
    server,
    config: Dict[str, Any],
    results_dir: Path,
    subdir: Optional[str] = None,
) -> Optional[Path]:
    """
    Persist server.global_model weights + metadata for downstream loading.

    Writes:
      - ``{subdir}/checkpoint_metadata.json``
      - ``{subdir}/global_model.pt``  (torch dict with ``state_dict``)
      - ``{subdir}/peft_adapter/``     (optional, when use_lora and PEFT save_pretrained exists)

    Args:
        server: Server instance with ``global_model`` (NewsClassifierModel).
        config: Experiment config dict (must include model_name, num_labels, use_lora, ...).
        results_dir: Results directory (same as experiment JSON).
        subdir: Subfolder under results_dir; default from config ``global_checkpoint_subdir``
                or ``global_checkpoint``.

    Returns:
        Path to checkpoint directory, or None if saving is disabled / failed.
    """
    if not config.get("save_global_checkpoint", False):
        return None

    results_dir = Path(results_dir)
    subdir = subdir or config.get("global_checkpoint_subdir", "global_checkpoint")
    ckpt_dir = results_dir / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_model: NewsClassifierModel = server.global_model
    meta = _build_checkpoint_metadata(config)

    meta_path = ckpt_dir / "checkpoint_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # CPU state_dict for portable checkpoints
    sd = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    torch.save({"state_dict": sd, "metadata": meta}, ckpt_dir / "global_model.pt")

    inner = global_model.model
    if meta["use_lora"] and hasattr(inner, "save_pretrained"):
        adapter_dir = ckpt_dir / "peft_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        try:
            inner.save_pretrained(str(adapter_dir))
        except Exception as e:
            print(f"  [fed_checkpoint] Warning: PEFT save_pretrained failed: {e}")

    print(f"  Global model checkpoint saved under: {ckpt_dir}")
    return ckpt_dir
