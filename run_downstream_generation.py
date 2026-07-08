#!/usr/bin/env python3
"""
Load a FedLLM SeqCLS checkpoint (NewsClassifierModel), transfer backbone to CausalLM,
classify probes with the SeqCLS head, then generate an explanation with the CausalLM.

Example:
  python run_downstream_generation.py \
    --checkpoint results/global_checkpoint \
    --probes "data/AG News Datasets/ag_news_business_30.json" \
    --output results/downstream_gen.jsonl \
    --stable
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoder_adapters import resolve_adapter
from models import NewsClassifierModel

DEFAULT_MAX_NEW_TOKENS = 128
STABLE_MAX_NEW_TOKENS = 128
STABLE_REPETITION_PENALTY = 1.15

AG_NEWS_ID2LABEL_FALLBACK = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
AG_NEWS_LABEL2ID_FALLBACK = {v: k for k, v in AG_NEWS_ID2LABEL_FALLBACK.items()}

_CATEGORY_CANONICAL = {
    "world": "World",
    "sports": "Sports",
    "business": "Business",
    "sci/tech": "Sci/Tech",
    "science/tech": "Sci/Tech",
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def category_to_label_id(category: Optional[str]) -> Optional[int]:
    if category is None:
        return None
    return AG_NEWS_LABEL2ID_FALLBACK.get(category)


def label_id_to_category(label_id: Optional[int]) -> Optional[str]:
    if label_id is None:
        return None
    return AG_NEWS_ID2LABEL_FALLBACK.get(int(label_id))


def normalize_category_name(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return _CATEGORY_CANONICAL.get(str(raw).strip().lower())


def normalize_dataset_label_id(raw: Any, category_hint: Optional[str] = None) -> Optional[int]:
    if raw is None:
        return category_to_label_id(normalize_category_name(category_hint))
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return category_to_label_id(normalize_category_name(category_hint))
    if 0 <= val <= 3:
        return val
    if 1 <= val <= 4:
        return val - 1
    return category_to_label_id(normalize_category_name(category_hint))


def seq_cls_to_ag_category(label_id: int, label_str: str, num_labels: int) -> str:
    """Map SeqCLS prediction to canonical AG News category name."""
    if num_labels == 4:
        c = AG_NEWS_ID2LABEL_FALLBACK.get(label_id)
        if c:
            return c
    ls = normalize_category_name(label_str)
    return ls or ""


# ---------------------------------------------------------------------------
# Checkpoint / model helpers
# ---------------------------------------------------------------------------

def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_paths(checkpoint: Path) -> Tuple[Path, Path]:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        return checkpoint / "global_model.pt", checkpoint / "checkpoint_metadata.json"
    return checkpoint, checkpoint.parent / "checkpoint_metadata.json"


def _load_metadata(pack: Dict[str, Any], meta_path: Path) -> Dict[str, Any]:
    meta = pack.get("metadata")
    if meta is None and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if not meta:
        raise ValueError("Missing metadata: expected 'metadata' in .pt or checkpoint_metadata.json next to checkpoint.")
    return meta


def build_news_classifier(meta: Dict[str, Any]) -> NewsClassifierModel:
    use_lora = bool(meta.get("use_lora", False))
    kw: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "num_labels": int(meta["num_labels"]),
        "use_lora": use_lora,
    }
    if use_lora:
        kw["lora_r"] = meta.get("lora_r", 16)
        kw["lora_alpha"] = meta.get("lora_alpha", 32)
        kw["lora_dropout"] = meta.get("lora_dropout", 0.1)
        tm = meta.get("lora_target_modules")
        kw["lora_target_modules"] = None if tm is None else list(tm)
    return NewsClassifierModel(**kw)


def load_news_classifier(pt_path: Path, meta_path: Path) -> Tuple[NewsClassifierModel, Dict[str, Any]]:
    pack = _torch_load(pt_path)
    meta = _load_metadata(pack, meta_path)
    model = build_news_classifier(meta)
    incompatible = model.load_state_dict(pack["state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"  Warning: missing_keys when loading NewsClassifierModel: {len(incompatible.missing_keys)} keys")
    if incompatible.unexpected_keys:
        print(f"  Warning: unexpected_keys when loading NewsClassifierModel: {len(incompatible.unexpected_keys)} keys")
    model.eval()
    return model, meta


def build_causal_lm(base_model_name: str, device: torch.device) -> torch.nn.Module:
    causal = AutoModelForCausalLM.from_pretrained(base_model_name)
    causal.to(device)
    causal.eval()
    return causal


def build_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Probes JSON must be a list of objects with id, news_text, and optional question")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "news_text" not in item:
            raise ValueError(f"Probe at index {i} must be an object with 'news_text'")
        if "question" not in item:
            item["question"] = ""
        raw_label = item.get("dataset_label_id")
        if raw_label is None:
            raw_label = item.get("ag_news_label_id", item.get("gold_ag_label"))
        raw_category = item.get("dataset_category")
        if raw_category is None:
            raw_category = item.get("ag_news_category", item.get("gold_category"))
        norm_category = normalize_category_name(raw_category)
        norm_label_id = normalize_dataset_label_id(raw_label, norm_category)
        if norm_category is None:
            norm_category = label_id_to_category(norm_label_id)
        if norm_label_id is None and norm_category is not None:
            norm_label_id = category_to_label_id(norm_category)
        if norm_label_id is not None:
            item["dataset_label_id"] = norm_label_id
        if norm_category is not None:
            item["dataset_category"] = norm_category
    return data


# ---------------------------------------------------------------------------
# SeqCLS prediction
# ---------------------------------------------------------------------------

def _classifier_config_from_news(news: NewsClassifierModel):
    """Resolve HuggingFace PretrainedConfig for id2label (handles PEFT wrapper)."""
    m = news.model
    cfg = getattr(m, "config", None)
    if cfg is not None and getattr(cfg, "id2label", None):
        return cfg
    base = getattr(m, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", base)
        cfg = getattr(inner, "config", None)
        if cfg is not None:
            return cfg
    return cfg


def get_id2label_map(news: NewsClassifierModel, num_labels: int) -> Dict[int, str]:
    cfg = _classifier_config_from_news(news)
    if cfg is not None and getattr(cfg, "id2label", None):
        mapped = {}
        for k, v in cfg.id2label.items():
            idx = int(k)
            canon = seq_cls_to_ag_category(idx, str(v), num_labels)
            mapped[idx] = canon or AG_NEWS_ID2LABEL_FALLBACK.get(idx, str(v))
        return mapped
    if num_labels == 4:
        return dict(AG_NEWS_ID2LABEL_FALLBACK)
    return {i: str(i) for i in range(num_labels)}


def seq_cls_argmax_one(
    news: NewsClassifierModel,
    tokenizer,
    text: str,
    id2label: Dict[int, str],
    max_length: int = 512,
) -> Tuple[int, str]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    dev = next(news.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = news(enc["input_ids"], enc["attention_mask"])
    pid = int(logits.argmax(dim=-1).item())
    label = id2label.get(pid, str(pid))
    return pid, label


def collect_seq_cls_predictions(
    news: NewsClassifierModel,
    tokenizer,
    probes: List[Dict[str, Any]],
    num_labels: int,
    max_length: int = 512,
) -> List[Tuple[int, str]]:
    id2label = get_id2label_map(news, num_labels)
    out: List[Tuple[int, str]] = []
    for p in probes:
        pid, lab = seq_cls_argmax_one(news, tokenizer, p["news_text"], id2label, max_length=max_length)
        out.append((pid, lab))
    return out


# ---------------------------------------------------------------------------
# Reason prompt & generation
# ---------------------------------------------------------------------------

def reason_prompt(news_text: str, category: str) -> str:
    """Build the prompt that asks the CausalLM to explain why the article fits *category*."""
    return (
        f"The following news article has been classified as: {category}.\n"
        "Explain why this classification is appropriate based on the article content.\n\n"
        f"Article:\n{news_text}\n\n"
        "Explanation:\n"
    )


def clean_reason_text(text: str) -> str:
    rb = (text or "").strip()
    rb = re.sub(r"^(reason|explanation)\s*:\s*", "", rb, flags=re.IGNORECASE)
    rb = rb.strip().strip("\"'").strip()
    return re.sub(r"\s+", " ", rb).strip()


def generate_completion(
    causal,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    repetition_penalty: Optional[float] = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    inp_len = int(inputs["input_ids"].shape[1])
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = max(temperature, 1e-5)
    else:
        gen_kw["do_sample"] = False
    if repetition_penalty is not None and repetition_penalty > 0:
        gen_kw["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        out = causal.generate(**inputs, **gen_kw)
    gen_ids = out[0, inp_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downstream explanation generation: SeqCLS classifies, CausalLM explains."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to global_model.pt or directory containing global_model.pt + checkpoint_metadata.json",
    )
    parser.add_argument(
        "--probes", type=Path, default=Path("data/AG News Datasets/ag_news_business_30.json"),
        help="JSON list of {id, news_text, ...}",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Override HF model id for CausalLM/tokenizer (default: model_name from checkpoint metadata)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help=f"Max new tokens for reason generation (default: {DEFAULT_MAX_NEW_TOKENS}; under --stable: {STABLE_MAX_NEW_TOKENS})",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (greedy decode)")
    parser.add_argument(
        "--stable", action="store_true",
        help=f"Greedy decode, max_new_tokens={STABLE_MAX_NEW_TOKENS}, repetition_penalty={STABLE_REPETITION_PENALTY}",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=None,
        help="Passed to generate(); when --stable and omitted, uses %.2f" % STABLE_REPETITION_PENALTY,
    )
    parser.add_argument(
        "--cls-max-length", type=int, default=512,
        help="Tokenizer max_length for SeqCLS classification",
    )
    args = parser.parse_args()

    _repo = Path(__file__).resolve().parent
    checkpoint_arg = args.checkpoint
    if not checkpoint_arg.is_absolute():
        checkpoint_arg = _repo / checkpoint_arg

    probes_path = args.probes
    if not probes_path.is_absolute():
        probes_path = _repo / probes_path
    if not probes_path.is_file():
        for alt in (
            _repo / "data" / "AG News Datasets" / args.probes.name,
            _repo / "data" / args.probes.name,
        ):
            if alt.is_file():
                probes_path = alt
                break

    device = torch.device(args.device)
    pt_path, meta_path = resolve_checkpoint_paths(checkpoint_arg)
    if not pt_path.is_file():
        print(f"Checkpoint file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    if not probes_path.is_file():
        print(f"Probes file not found: {probes_path}", file=sys.stderr)
        sys.exit(1)

    probes = load_probes(probes_path)
    news, meta = load_news_classifier(pt_path, meta_path)
    base_name = args.base_model or meta["model_name"]
    if args.base_model and args.base_model != meta["model_name"]:
        print(
            f"  Note: --base-model {args.base_model!r} overrides metadata model_name {meta['model_name']!r}; "
            "ensure architecture matches the saved weights."
        )
    num_labels = int(meta["num_labels"])

    do_sample = not args.greedy
    repetition_penalty: Optional[float] = args.repetition_penalty
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    elif args.stable:
        max_new_tokens = STABLE_MAX_NEW_TOKENS
    else:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    if args.stable:
        do_sample = False
        if repetition_penalty is None:
            repetition_penalty = STABLE_REPETITION_PENALTY

    # --- SeqCLS classification ---
    tokenizer = build_tokenizer(base_name)
    seq_predictions = collect_seq_cls_predictions(
        news, tokenizer, probes, num_labels, max_length=args.cls_max_length
    )
    print(f"  SeqCLS argmax predictions: {len(seq_predictions)} probes")

    # --- CausalLM reason generation ---
    adapter = resolve_adapter(meta["model_name"])
    causal = build_causal_lm(base_name, device)
    adapter.transfer_backbone(news.model, causal)

    print(f"  Checkpoint: {pt_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, p in enumerate(probes):
            cat_id, cat_label = seq_predictions[i]
            cat = seq_cls_to_ag_category(cat_id, cat_label, num_labels)
            if not cat:
                cat = label_id_to_category(cat_id) or "World"

            prompt = reason_prompt(p["news_text"], cat)
            raw_reason = generate_completion(
                causal, tokenizer, prompt, device,
                max_new_tokens, do_sample, args.temperature,
                repetition_penalty=repetition_penalty,
            )
            cleaned = clean_reason_text(raw_reason)
            completion = f"Category: {cat}\nReason: {cleaned}"

            row: Dict[str, Any] = {
                "probe_id": p.get("id", i + 1),
                "news_text": p["news_text"],
                "seq_cls_category_id": cat_id,
                "seq_cls_category": cat,
                "completion": completion,
                "reason_raw": raw_reason,
                "reason_prompt": prompt,
            }
            if "dataset_label_id" in p:
                row["dataset_label_id"] = p["dataset_label_id"]
                row["dataset_category"] = p.get("dataset_category") or label_id_to_category(p["dataset_label_id"])
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    del causal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  Wrote {len(probes)} lines to {args.output}")


if __name__ == "__main__":
    main()
