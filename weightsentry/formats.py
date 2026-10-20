"""Format sniffing: is this artifact inert data, or a loader program?"""
from __future__ import annotations

import os
from dataclasses import dataclass

# executable_on_load: loading the file can run attacker code by format design.
FORMATS = {
    "gguf":        dict(executable_on_load=False, note="tensor + metadata container; parsed, never executed"),
    "safetensors": dict(executable_on_load=False, note="8-byte header + raw tensor bytes; no code path"),
    "onnx":        dict(executable_on_load=False, note="protobuf graph; ops resolved by the runtime, custom domains are the risk"),
    "zip-torch":   dict(executable_on_load=True,  note="zip container whose data.pkl is unpickled by torch.load"),
    "pickle":      dict(executable_on_load=True,  note="raw pickle: the opcode stream is a program"),
    "tflite":      dict(executable_on_load=False, note="flatbuffer graph"),
    "unknown":     dict(executable_on_load=None,  note="not recognised"),
}


@dataclass
class Sniffed:
    path: str
    size: int
    fmt: str
    executable_on_load: bool | None
    note: str
    detail: dict


def sniff(path: str) -> Sniffed:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(64)
    ext = os.path.splitext(path)[1].lower()
    detail: dict = {}

    if head[:4] == b"GGUF":
        fmt = "gguf"
        detail["magic"] = "GGUF"
    elif head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        fmt = "zip-torch"
    elif head[:4] == b"\x08\x00\x12\x05" or head[:4] == b"ONNX" or ext == ".onnx":
        fmt = "onnx"
    elif head[:4] == b"TFL3":
        fmt = "tflite"
    elif head[:1] in (b"\x80", b"\x89", b"c", b"}") and ext in (".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".bin", ".joblib"):
        fmt = "pickle"
    elif ext in (".safetensors",):
        fmt = "safetensors"
    elif ext in (".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"):
        fmt = "pickle"
    else:
        fmt = "unknown"

    meta = FORMATS[fmt]
    return Sniffed(path=path, size=size, fmt=fmt,
                   executable_on_load=meta["executable_on_load"],
                   note=meta["note"], detail=detail)
