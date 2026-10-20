"""Numeric layer: the checks must fire on a constructed implant and stay quiet
on an ordinary adapter. Both cases are built here, in memory, with numpy only.
"""
import os

import numpy as np
import pytest

from weightsentry import numeric

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def write_adapter(path, modules, r=8, base_dim=64):
    """Write a small safetensors file by hand: no torch, no safetensors dep."""
    import json
    import struct
    header, blobs, off = {}, [], 0
    for name, (A, B) in modules.items():
        for tag, W in (("lora_A", A), ("lora_B", B)):
            key = f"{name}.{tag}.weight"
            raw = np.ascontiguousarray(W, dtype=np.float32).tobytes()
            header[key] = {"dtype": "F32", "shape": list(W.shape),
                           "data_offsets": [off, off + len(raw)]}
            blobs.append(raw)
            off += len(raw)
    hb = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)


def ordinary_modules(n=16, r=8, dim=64, seed=0):
    rng = np.random.RandomState(seed)
    return {f"model.layers.{i}.self_attn.q_proj":
            (rng.randn(r, dim).astype(np.float32) * 0.02,
             rng.randn(dim, r).astype(np.float32) * 0.002)
            for i in range(n)}


def test_ordinary_adapter_is_clean(tmp_path):
    p = tmp_path / "clean.safetensors"
    write_adapter(str(p), ordinary_modules())
    rep = numeric.audit_lora(str(p))
    assert rep.verdict == "clean", rep.findings
    assert rep.evidence["modules"] == 16


def test_implanted_module_is_caught(tmp_path):
    mods = ordinary_modules()
    # one module carries 20x the update magnitude of every other module
    A, B = mods["model.layers.3.self_attn.q_proj"]
    mods["model.layers.3.self_attn.q_proj"] = (A, (B * 20.0).astype(np.float32))
    p = tmp_path / "implant.safetensors"
    write_adapter(str(p), mods)
    rep = numeric.audit_lora(str(p))
    assert rep.verdict == "review", rep.findings
    assert any("layers.3.self_attn.q_proj" in f for f in rep.findings), rep.findings


def test_zeroed_b_matrix_is_dangerous(tmp_path):
    mods = ordinary_modules()
    A, B = mods["model.layers.0.self_attn.q_proj"]
    mods["model.layers.0.self_attn.q_proj"] = (A, np.zeros_like(B))
    p = tmp_path / "zeroed.safetensors"
    write_adapter(str(p), mods)
    rep = numeric.audit_lora(str(p))
    assert rep.verdict == "dangerous", rep.findings
    assert any("exactly zero" in f for f in rep.findings)


def test_truncated_file_is_dangerous(tmp_path):
    mods = ordinary_modules()
    p = tmp_path / "cut.safetensors"
    write_adapter(str(p), mods)
    raw = open(str(p), "rb").read()
    open(str(p), "wb").write(raw[: len(raw) // 2])
    rep = numeric.audit_lora(str(p))
    assert rep.verdict == "dangerous", rep.findings


def test_flat_cohort_does_not_produce_outliers(tmp_path):
    """Every module identical: MAD collapses to zero. That must not flag."""
    rng = np.random.RandomState(1)
    A = rng.randn(8, 64).astype(np.float32) * 0.02
    B = rng.randn(64, 8).astype(np.float32) * 0.002
    mods = {f"model.layers.{i}.mlp.up_proj": (A.copy(), B.copy()) for i in range(12)}
    p = tmp_path / "flat.safetensors"
    write_adapter(str(p), mods)
    rep = numeric.audit_lora(str(p), z_threshold=6.0)
    assert rep.verdict == "clean", rep.findings


def test_lora_module_stats_never_forms_delta_w():
    rng = np.random.RandomState(2)
    A = rng.randn(4, 512).astype(np.float32)
    B = rng.randn(4096, 4).astype(np.float32)
    st = numeric.lora_module_stats(A, B)
    # energy must equal ||B@A||_F computed the slow way
    slow = float(np.linalg.norm(B @ A))
    assert abs(st["energy"] - slow) / slow < 1e-4
    assert 0.0 <= st["top1"] <= 1.0
