"""GGUF baseline differential and embedding scan, on synthetic files."""
import os
import struct

import numpy as np
import pytest

from weightsentry import numeric


def write_gguf(path, tensors, meta=None):
    """Minimal valid GGUF v3: F32 tensors only, so nothing needs quantizing."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    meta = meta or {}
    kv = b""
    for k, v in meta.items():
        kv += s(k) + struct.pack("<I", 8) + s(str(v))
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(meta)) + kv
    infos, blobs, off = b"", b"", 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        infos += s(name) + struct.pack("<I", a.ndim)
        infos += struct.pack("<" + "Q" * a.ndim, *a.shape)
        infos += struct.pack("<I", 0) + struct.pack("<Q", off)   # type 0 = F32
        blobs += a.tobytes()
        off += a.nbytes
    body = head + infos
    pad = (-len(body)) % 32
    with open(path, "wb") as fh:
        fh.write(body + b"\0" * pad + blobs)


@pytest.fixture
def base_and_tuned(tmp_path):
    rng = np.random.RandomState(0)
    base = {
        "blk.0.attn_q.weight": rng.randn(32, 32).astype(np.float32),
        "blk.0.attn_norm.weight": np.abs(rng.randn(32).astype(np.float32)),
        "blk.0.attn_q.bias": rng.randn(32).astype(np.float32),
        "token_embd.weight": rng.randn(16, 32).astype(np.float32),
    }
    tuned = {k: v.copy() for k, v in base.items()}
    tuned["blk.0.attn_q.weight"] += 0.01          # a real weight change
    b = tmp_path / "base.gguf"; write_gguf(str(b), base, {"general.name": "base"})
    t = tmp_path / "tuned.gguf"; write_gguf(str(t), tuned, {"general.name": "tuned"})
    return str(b), str(t)


def test_lora_merge_pattern_is_recognised(base_and_tuned):
    b, t = base_and_tuned
    rep = numeric.gguf_baseline_diff(t, b)
    ev = rep.evidence
    assert ev["changed"] == 1
    assert ev["same"] >= 2
    assert rep.verdict == "clean", rep.findings


def test_touching_a_norm_is_flagged(base_and_tuned, tmp_path):
    b, t = base_and_tuned
    idx, _kv, _ = numeric.gguf_index(t)
    tset, _kv2, _ = numeric.gguf_index(b)
    # rewrite the tuned file with the norm also changed
    base = {
        "blk.0.attn_q.weight": np.zeros((32, 32), np.float32),
        "blk.0.attn_norm.weight": np.zeros(32, np.float32),
        "blk.0.attn_q.bias": np.zeros(32, np.float32),
        "token_embd.weight": np.zeros((16, 32), np.float32),
    }
    tuned = {k: v.copy() for k, v in base.items()}
    tuned["blk.0.attn_q.weight"] += 1.0
    tuned["blk.0.attn_norm.weight"] += 1.0
    b2 = tmp_path / "b2.gguf"; write_gguf(str(b2), base)
    t2 = tmp_path / "t2.gguf"; write_gguf(str(t2), tuned)
    rep = numeric.gguf_baseline_diff(str(t2), str(b2))
    assert rep.verdict == "review", rep.findings
    assert any("norm" in f for f in rep.findings), rep.findings


def test_shape_mismatch_is_dangerous(tmp_path):
    a = {"blk.0.attn_q.weight": np.zeros((32, 32), np.float32)}
    b = {"blk.0.attn_q.weight": np.zeros((16, 32), np.float32)}
    pa = tmp_path / "a.gguf"; write_gguf(str(pa), a)
    pb = tmp_path / "b.gguf"; write_gguf(str(pb), b)
    rep = numeric.gguf_baseline_diff(str(pa), str(pb))
    assert rep.verdict == "dangerous", rep.findings


def test_float_noise_is_not_a_change(base_and_tuned, tmp_path):
    base = {"blk.0.attn_norm.weight": np.ones(32, np.float32)}
    tuned = {"blk.0.attn_norm.weight": np.nextafter(np.ones(32, np.float32),
                                                    np.zeros(32, np.float32))}
    pb = tmp_path / "b.gguf"; write_gguf(str(pb), base)
    pt = tmp_path / "t.gguf"; write_gguf(str(pt), tuned)
    rep = numeric.gguf_baseline_diff(str(pt), str(pb))
    assert rep.evidence["changed"] == 0
    assert rep.evidence["float_noise_only"] == 1


def test_embedding_scan_without_base_is_informational(base_and_tuned):
    _b, t = base_and_tuned
    rep = numeric.gguf_embedding_outliers(t)
    # F32 embeddings are not a K-quant, so the scan bows out instead of guessing
    assert rep.verdict in ("review", "clean")
