"""Pickle opcode auditing -- classify a serialized blob WITHOUT loading it.

The threat: torch.load()/pickle.load() execute whatever callable the pickle
names, at load time, with the loader's privileges. A .pt file is a program.

This module never calls pickle.load. It walks the opcode stream with
pickletools.genops and reports what the blob *would* call.
"""
from __future__ import annotations

import io
import pickletools
import zipfile
from dataclasses import dataclass, field

# Opcodes that can construct or invoke a callable.
EXEC_OPS = {
    "GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "REDUCE",
    "BUILD", "NEWOBJ", "NEWOBJ_EX", "EXT1", "EXT2", "EXT4",
}

# Callables that appear in ordinary checkpoints. Not "safe" -- just expected.
BENIGN_PREFIXES = (
    "torch.", "collections.OrderedDict", "collections.defaultdict",
    "numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype",
    "numpy._core.multiarray._reconstruct", "_codecs.encode",
    "builtins.set", "builtins.frozenset", "builtins.bytearray",
    "builtins.complex", "builtins.range", "builtins.slice",
    "torch.nn.parameter", "argparse.Namespace", "copyreg._reconstructor",
    "copy_reg._reconstructor", "pandas.", "sklearn.", "pathlib.",
    "transformers.", "accelerate.", "llamafactory.", "peft.", "datasets.",
    "deepspeed.", "safetensors.", "tokenizers.", "einops.", "omegaconf.",
)

# Substrings that turn a benign-looking import into an exfil primitive.
SUSPICIOUS_TOKENS = (
    "system", "subprocess", "popen", "posix", "nt.", "os.fork", "exec",
    "eval", "compile", "socket", "urllib", "httplib", "http.client",
    "requests", "ftplib", "smtplib", "telnetlib", "paramiko", "pty",
    "webbrowser", "ctypes", "runpy", "importlib", "builtins.getattr",
    "builtins.__import__", "boto3", "gsutil", "base64.b64decode",
    "marshal.loads", "pickle.loads", "shutil", "tempfile", "socket.socket",
)


@dataclass
class PickleVerdict:
    source: str
    ok: bool = True
    error: str | None = None
    exec_ops: list[tuple[str, int]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    suspicious: list[str] = field(default_factory=list)
    verdict: str = "clean"       # clean | review | dangerous

    def as_dict(self):
        return {
            "source": self.source, "ok": self.ok, "error": self.error,
            "exec_ops": len(self.exec_ops), "calls": self.calls[:40],
            "suspicious": self.suspicious, "verdict": self.verdict,
        }


def _judge(calls, exec_ops, source) -> PickleVerdict:
    v = PickleVerdict(source=source, exec_ops=exec_ops, calls=sorted(set(calls)))
    for name in v.calls:
        low = name.lower()
        if any(t in low for t in SUSPICIOUS_TOKENS):
            v.suspicious.append(name)
        elif not name.startswith(BENIGN_PREFIXES):
            v.suspicious.append(name)
    # STACK_GLOBAL with a computed name, or a REDUCE of an unknown callable:
    # both are how obfuscated loaders hide the string from naive greps.
    if "STACK_GLOBAL" in {op for op, _ in exec_ops} and not v.calls:
        v.suspicious.append("<STACK_GLOBAL with computed name>")
    if v.suspicious:
        v.verdict = "dangerous" if any(
            any(t in s.lower() for t in ("system", "subprocess", "popen", "socket",
                                         "urllib", "requests", "exec", "eval",
                                         "ctypes", "__import__", "paramiko"))
            for s in v.suspicious) else "review"
    return v


def scan_pickle_bytes(data: bytes, source: str = "<bytes>") -> PickleVerdict:
    """Walk the opcode stream of one pickle. The blob is never executed."""
    calls, ops = [], []
    # Protocol >=4 hides the callable behind STACK_GLOBAL: the module and the
    # attribute are pushed as plain strings first, so follow the string stack.
    stack: list[str] = []
    STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                  "STRING", "SHORT_BINSTRING", "BINSTRING"}
    try:
        # genops stops at STOP; anything after STOP is unreachable payload.
        for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            if op.name in EXEC_OPS:
                ops.append((op.name, _pos))
            if op.name in ("GLOBAL", "INST") and isinstance(arg, str):
                calls.append(arg.replace("\n", ".").replace(" ", "."))
            elif op.name in STRING_OPS and isinstance(arg, str):
                stack.append(arg)
                del stack[:-8]
            elif op.name == "STACK_GLOBAL":
                if len(stack) >= 2:
                    calls.append("%s.%s" % (stack[-2], stack[-1]))
                elif len(stack) == 1:
                    calls.append("?.%s" % stack[-1])
                else:
                    calls.append("<stack-global>")
            elif op.name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "MARK", "POP",
                             "POP_MARK", "DICT", "LIST", "EMPTY_TUPLE"):
                stack.clear() if op.name in ("POP", "POP_MARK", "DICT", "LIST") else None
    except Exception as exc:                        # truncated / not a pickle
        v = PickleVerdict(source=source, ok=False, error=f"{type(exc).__name__}: {exc}")
        v.verdict = "review"
        return v
    return _judge(calls, ops, source)


def scan_file(path: str) -> list[PickleVerdict]:
    """Scan a .pt/.pth/.ckpt/.bin/.pkl. Zips (torch>=1.6) are opened member-wise."""
    out: list[PickleVerdict] = []
    with open(path, "rb") as fh:
        head = fh.read(8)
    is_zip = head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    if is_zip:
        try:
            with zipfile.ZipFile(path) as zf:
                members = [m for m in zf.namelist() if m.endswith((".pkl", ".pickle"))]
                if not members:
                    out.append(PickleVerdict(source=path + "::<zip:no-pickle>",
                                             ok=False, error="zip without .pkl member",
                                             verdict="review"))
                for m in members:
                    out.append(scan_pickle_bytes(zf.read(m), f"{path}::{m}"))
        except Exception as exc:
            out.append(PickleVerdict(source=path, ok=False,
                                     error=f"{type(exc).__name__}: {exc}", verdict="review"))
        return out
    with open(path, "rb") as fh:
        out.append(scan_pickle_bytes(fh.read(), path))
    return out
