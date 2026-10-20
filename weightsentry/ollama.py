"""Ollama store integrity: re-hash every blob against its manifest digest.

An Ollama model is a manifest (JSON) pointing at content-addressed blobs.
A tampered weight file breaks the digest; a tampered *manifest* can point a
model name at any blob, including one pulled from somewhere else. Both are
cheap to check offline, so check them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

DEFAULT_ROOTS = [
    os.path.expanduser(r"~\.ollama\models"),
    os.path.expandvars(r"%LOCALAPPDATA%\Ollama\models"),
]


@dataclass
class Blob:
    digest: str
    media: str
    declared: int
    actual: int
    sha_ok: bool | None = None      # None = not hashed (too big / skipped)
    zerofill: bool | None = None    # True = sampled chunks are all zero bytes


@dataclass
class OllamaModel:
    name: str
    manifest: str
    root: str = ""
    blobs: list[Blob] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(b.declared for b in self.blobs)

    def verdict(self) -> str:
        if self.errors or any(b.sha_ok is False for b in self.blobs):
            return "dangerous"
        if any(b.sha_ok is None for b in self.blobs):
            return "review"
        return "clean"


def find_roots(extra=None):
    """Explicit roots win. Defaults are only used when nothing was passed,
    so a caller auditing a fixture store is not silently given the real one."""
    if extra:
        return [r for r in extra if os.path.isdir(r)]
    roots = []
    for r in DEFAULT_ROOTS:
        if r and os.path.isdir(r) and r not in roots:
            roots.append(r)
    return roots


def list_models(roots=None) -> list[OllamaModel]:
    out = []
    for root in find_roots(roots):
        mdir = os.path.join(root, "manifests")
        if not os.path.isdir(mdir):
            continue
        for dp, _dns, fns in os.walk(mdir):
            for f in fns:
                path = os.path.join(dp, f)
                rel = os.path.relpath(path, mdir).replace("\\", "/")
                name = rel.rsplit("/", 1)[0] + ":" + f
                m = OllamaModel(name=name, manifest=path, root=root)
                try:
                    spec = json.load(open(path, encoding="utf-8"))
                except Exception as exc:
                    m.errors.append(f"manifest unreadable: {exc}")
                    out.append(m)
                    continue
                blobs_dir = os.path.join(root, "blobs")
                for layer in spec.get("layers", []) + [spec.get("config", {})]:
                    digest = (layer or {}).get("digest")
                    if not digest:
                        continue
                    bpath = os.path.join(blobs_dir, digest.replace(":", "-"))
                    media = (layer.get("mediaType") or "").split(".")[-1]
                    declared = layer.get("size", -1)
                    if not os.path.exists(bpath):
                        m.blobs.append(Blob(digest, media, declared, -1, False))
                        m.errors.append(f"missing blob {digest[:19]}")
                        continue
                    m.blobs.append(Blob(digest, media, declared, os.path.getsize(bpath)))
                out.append(m)
    return out


def is_zerofill(path: str, samples: int = 12, chunk: int = 4096) -> bool:
    """A blob of nothing but zero bytes is a placeholder, not a model.

    Interrupted or disk-full pulls leave these behind, and they read exactly
    like a tampered weight file until you look at the bytes.
    """
    size = os.path.getsize(path)
    if size == 0:
        return True
    import random
    random.seed(0)
    with open(path, "rb") as fh:
        for _ in range(samples):
            fh.seek(random.randrange(0, max(1, size - chunk)))
            if any(fh.read(chunk)):
                return False
    return True


def verify_blob(path: str, alg: str = "sha256", chunk: int = 1 << 22) -> str:
    h = hashlib.new(alg)
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify_model(m: OllamaModel, root: str | None = None, max_bytes: int | None = None) -> OllamaModel:
    root = root or m.root
    blobs_dir = os.path.join(root, "blobs")
    for b in m.blobs:
        bpath = os.path.join(blobs_dir, b.digest.replace(":", "-"))
        if not os.path.exists(bpath):
            b.sha_ok = False
            continue
        if max_bytes is not None and b.actual > max_bytes:
            b.sha_ok = None
            continue
        alg, want = b.digest.split(":", 1)
        b.sha_ok = verify_blob(bpath, alg) == want
        if not b.sha_ok and b.media == "model":
            b.zerofill = is_zerofill(bpath)
    return m
