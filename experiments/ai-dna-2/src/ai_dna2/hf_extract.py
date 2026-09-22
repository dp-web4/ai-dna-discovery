"""Hugging Face hidden-state extraction helpers for AI-DNA 2.0.

Torch/Transformers are imported lazily so the Phase A harness remains NumPy-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_depths(text: str) -> tuple[float, ...]:
    depths = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not depths:
        raise ValueError("at least one normalized layer depth is required")
    if any(d <= 0.0 or d > 1.0 for d in depths):
        raise ValueError("layer depths must satisfy 0 < depth <= 1")
    if len(set(depths)) != len(depths):
        raise ValueError("layer depths must be unique")
    return depths


def select_layer_indices(num_hidden_layers: int, depths: Iterable[float]) -> dict[str, int]:
    """Map normalized depths to hidden_states tuple indices."""

    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive")
    out: dict[str, int] = {}
    for depth in depths:
        d = float(depth)
        if d <= 0.0 or d > 1.0:
            raise ValueError("layer depths must satisfy 0 < depth <= 1")
        index = int(round(d * num_hidden_layers))
        index = max(1, min(num_hidden_layers, index))
        out[f"{d:.2f}"] = index
    return out


def load_anchor_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for field in ("concept_id", "domain", "label", "definition", "split"):
                if not record.get(field):
                    raise ValueError(f"anchor line {line_no} missing {field}")
            concept_id = str(record["concept_id"])
            if concept_id in seen:
                raise ValueError(f"duplicate concept_id: {concept_id}")
            seen.add(concept_id)
            records.append(record)
    if len(records) < 3:
        raise ValueError("anchor bank must contain at least three concepts")
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("_")


def build_prompt(record: dict) -> str:
    return (
        f"Concept: {record['label']}\n"
        f"Definition: {record['definition']}\n"
        "Represent the meaning of this concept."
    )


def extract_hf_model(
    *,
    model_id: str,
    anchors_path: str | Path,
    output_dir: str | Path,
    revision: str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    depths: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    pooling: str = "last",
    max_tokens: int = 192,
) -> tuple[Path, Path]:
    """Extract selected hidden-state vectors and write NPZ + JSON manifest."""

    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be 'last' or 'mean'")
    if max_tokens < 8:
        raise ValueError("max_tokens is implausibly small")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "model extraction requires torch and transformers; install the model extras first"
        ) from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if device == "cpu" and dtype != "float32":
        raise ValueError("CPU pilot must use float32 for portable, predictable numerics")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    anchors_path = Path(anchors_path)
    records = load_anchor_records(anchors_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"trust_remote_code": False}
    if revision:
        load_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_map[dtype],
        **load_kwargs,
    )
    model.to(device)
    model.eval()

    config = model.config
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_layers < 1:
        raise RuntimeError("model config does not expose num_hidden_layers")
    selected = select_layer_indices(num_layers, depths)

    collected: dict[str, list[np.ndarray]] = {key: [] for key in selected}
    token_counts: list[int] = []

    for record in records:
        prompt = build_prompt(record)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=True,
        )
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        token_count = int(inputs["attention_mask"][0].sum().item())
        token_counts.append(token_count)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != num_layers + 1:
            raise RuntimeError(
                f"unexpected hidden-state count: got {0 if hidden_states is None else len(hidden_states)}, "
                f"expected {num_layers + 1}"
            )

        for key, layer_index in selected.items():
            state = hidden_states[layer_index][0]
            if pooling == "last":
                vector = state[token_count - 1]
            else:
                vector = state[:token_count].mean(dim=0)
            collected[key].append(vector.detach().float().cpu().numpy())

        del outputs, hidden_states, inputs

    arrays = {
        f"depth_{key.replace('.', '')}": np.stack(vectors).astype(np.float32)
        for key, vectors in collected.items()
    }

    slug = safe_model_slug(model_id)
    activation_path = output_dir / f"{slug}.npz"
    manifest_path = output_dir / f"{slug}.manifest.json"
    np.savez_compressed(activation_path, **arrays)

    commit_hash = getattr(config, "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    manifest = {
        "schema": "ai-dna-2.hidden-state-manifest.v1",
        "model_id": model_id,
        "requested_revision": revision,
        "model_commit_hash": commit_hash,
        "tokenizer_commit_hash": tokenizer_commit,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "architectures": list(getattr(config, "architectures", []) or []),
        "model_type": getattr(config, "model_type", None),
        "parameter_count": parameter_count,
        "num_hidden_layers": num_layers,
        "hidden_size": int(getattr(config, "hidden_size", arrays[next(iter(arrays))].shape[1])),
        "selected_layers": selected,
        "pooling": pooling,
        "device": device,
        "dtype": dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "anchor_file": str(anchors_path),
        "anchor_sha256": file_sha256(anchors_path),
        "sample_ids": [str(record["concept_id"]) for record in records],
        "domains": [str(record["domain"]) for record in records],
        "splits": [str(record["split"]) for record in records],
        "token_counts": token_counts,
        "max_tokens": max_tokens,
        "activation_file": activation_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return activation_path, manifest_path
