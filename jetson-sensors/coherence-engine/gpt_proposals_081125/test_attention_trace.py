import os, json, pathlib, time
from coherence_engine.attention.attention_trace import emit

def test_emit_creates_file(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTENTION_TRACE", "1")
    monkeypatch.setenv("ATTENTION_TRACE_PATH", str(path))
    emit("salience", {"motion":0.7}, {"bg":-0.2}, "unit-test", {"vision":0.5}, {"vision":0.6})
    assert path.exists()
    data = [json.loads(l) for l in path.read_text().splitlines()]
    assert data and data[0]["policy"] == "salience"
