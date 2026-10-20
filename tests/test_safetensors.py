"""safetensors audit: a well-formed file is clean, a doctored one is not."""
import json
import struct

import pytest

from weightsentry import safetensors


def write_safetensors(path, tensors, meta=None, trailing=b""):
    header = dict(tensors)
    if meta is not None:
        header["__metadata__"] = meta
    hb = json.dumps(header).encode()
    data = b""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(data)
        fh.write(trailing)


def test_clean(tmp_path):
    p = tmp_path / "ok.safetensors"
    # one F32 tensor of 4 elements = 16 bytes, declared correctly
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    hb = json.dumps(header).encode()
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\0" * 16)
    r = safetensors.inspect(str(p))
    assert r.verdict == "clean", r.problems


def test_trailing_data_is_flagged(tmp_path):
    p = tmp_path / "trail.safetensors"
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    hb = json.dumps(header).encode()
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\0" * 16)
        fh.write(b"SMUGGLED-PAYLOAD")
    r = safetensors.inspect(str(p))
    assert any("trailing" in x for x in r.problems), r.problems


def test_offsets_past_eof(tmp_path):
    p = tmp_path / "evil.safetensors"
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 99999999]}}
    hb = json.dumps(header).encode()
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\0" * 16)
    r = safetensors.inspect(str(p))
    assert r.verdict == "dangerous", r.problems


def test_shape_size_mismatch(tmp_path):
    p = tmp_path / "bad.safetensors"
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
    hb = json.dumps(header).encode()
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\0" * 8)
    r = safetensors.inspect(str(p))
    assert any("span" in x for x in r.problems), r.problems
