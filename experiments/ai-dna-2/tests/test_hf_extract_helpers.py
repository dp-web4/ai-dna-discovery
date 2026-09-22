from pathlib import Path

import pytest

from ai_dna2.hf_extract import (
    file_sha256,
    load_anchor_records,
    parse_depths,
    safe_model_slug,
    select_layer_indices,
)


def test_depth_parsing_and_selection():
    depths = parse_depths("0.25,0.50,0.75,1.00")
    assert depths == (0.25, 0.5, 0.75, 1.0)
    assert select_layer_indices(24, depths) == {
        "0.25": 6,
        "0.50": 12,
        "0.75": 18,
        "1.00": 24,
    }


def test_depth_validation():
    with pytest.raises(ValueError):
        parse_depths("0,0.5")
    with pytest.raises(ValueError):
        parse_depths("0.5,0.5")


def test_anchor_bank_is_valid_and_split():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "anchors" / "pilot.jsonl"
    records = load_anchor_records(path)
    assert len(records) == 40
    assert len({r["concept_id"] for r in records}) == 40
    assert {r["split"] for r in records} == {
        "train", "heldout_concept", "heldout_domain"
    }
    assert len(file_sha256(path)) == 64


def test_model_slug_is_path_safe():
    assert safe_model_slug("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen__Qwen2.5-0.5B-Instruct"
