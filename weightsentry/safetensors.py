"""safetensors structural audit.

Layout: <u64 header_len><json header><raw tensor bytes>. There is no code
path on load, so the audit is about consistency, not execution:
  - header must be valid JSON with tensor offsets inside the file
  - metadata must not carry a payload (a 'pt' format tag is normal)
  - offsets/sizes must not overlap or run past EOF (smuggled data / corruption)
  - optional: read tensor statistics without importing torch
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

METADATA_KEYS_EXPECTED = {"format", "format_version", "model_type", "architectures",
                          "peft_type", "base_model_name_or_path", "r", "lora_alpha"}


@dataclass
class SafetensorsReport:
    path: str
    size: int = 0
    header_bytes: int = 0
    tensors: int = 0
    dtypes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    verdict: str = "clean"

    def as_dict(self):
        return {
            "path": self.path, "size": self.size, "header_bytes": self.header_bytes,
            "tensors": self.tensors, "dtypes": self.dtypes,
            "metadata": {k: (v[:120] if isinstance(v, str) else v) for k, v in self.metadata.items()},
            "problems": self.problems, "verdict": self.verdict,
        }


def inspect(path: str, stat_tensors: bool = False) -> SafetensorsReport:
    r = SafetensorsReport(path=path, size=os.path.getsize(path))
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            r.problems.append("file shorter than 8-byte header length")
            r.verdict = "dangerous"
            return r
        n = struct.unpack("<Q", raw)[0]
        r.header_bytes = n
        if n == 0 or n > r.size or n > 100 * 1024 * 1024:
            r.problems.append(f"implausible header length {n}")
            r.verdict = "dangerous"
            return r
        header = fh.read(n)
    try:
        spec = json.loads(header)
    except Exception as exc:
        r.problems.append(f"header not JSON: {exc}")
        r.verdict = "dangerous"
        return r
    if not isinstance(spec, dict):
        r.problems.append("header is not an object")
        r.verdict = "dangerous"
        return r

    meta = spec.pop("__metadata__", {}) or {}
    r.metadata = meta
    r.tensors = len(spec)
    r.dtypes = sorted({v.get("dtype") for v in spec.values()})

    data_start = 8 + n
    covered = 0
    end_min = 0
    for name, info in spec.items():
        offs = info.get("data_offsets")
        if not (isinstance(offs, list) and len(offs) == 2):
            r.problems.append(f"{name}: bad data_offsets")
            continue
        a, b = offs
        if b < a:
            r.problems.append(f"{name}: end < begin")
        if a < end_min:
            r.problems.append(f"{name}: overlaps previous tensor")
        end_min = max(end_min, b)
        covered = max(covered, b)
        shape = info.get("shape", [])
        dt = info.get("dtype")
        expect = 1
        for d in shape:
            expect *= d
        if dt in DTYPE_BYTES:
            want = expect * DTYPE_BYTES[dt]
            if want != b - a:
                r.problems.append(f"{name}: span {b-a}B != {want}B for {dt}{shape}")

    if data_start + covered > r.size:
        r.problems.append(f"tensors run {data_start+covered-r.size}B past EOF")
    trailing = r.size - (data_start + covered)
    if trailing > 0:
        r.problems.append(f"{trailing}B of trailing data after last tensor")

    if meta and not set(meta) <= METADATA_KEYS_EXPECTED:
        unexpected = sorted(set(meta) - METADATA_KEYS_EXPECTED)
        r.problems.append(f"unexpected metadata keys: {unexpected}")

    if r.problems:
        r.verdict = "dangerous" if any("past EOF" in p or "not JSON" in p for p in r.problems) else "review"
    return r
