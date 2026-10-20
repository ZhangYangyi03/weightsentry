"""weightsentry CLI: audit local model artifacts and report what they can do to you."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, formats, ollama, rce, safetensors

MODEL_EXTS = {".gguf", ".safetensors", ".onnx", ".pt", ".pth", ".ckpt", ".pkl",
              ".pickle", ".h5", ".tflite", ".joblib", ".bin"}
SKIP_DIRS = {"node_modules", "site-packages", ".git", ".gradle", "dump_syms"}


def discover(roots, max_depth=8):
    seen, found = set(), []
    for root in roots:
        if not os.path.exists(root):
            continue
        if os.path.isfile(root):
            found.append(os.path.abspath(root))
            continue
        base_depth = root.rstrip("\\/").count(os.sep)
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d.lower() not in SKIP_DIRS]
            if dp.count(os.sep) - base_depth > max_depth:
                dns[:] = []
                continue
            for f in fns:
                if os.path.splitext(f)[1].lower() in MODEL_EXTS:
                    p = os.path.abspath(os.path.join(dp, f))
                    if p.lower() not in seen:
                        seen.add(p.lower())
                        found.append(p)
    return sorted(found)


def audit_file(path, hash_limit=None):
    """One artifact -> one report dict."""
    s = formats.sniff(path)
    rep = {"path": path, "size": s.size, "format": s.fmt,
           "executable_on_load": s.executable_on_load, "note": s.note,
           "findings": [], "verdict": "clean"}
    if s.fmt == "safetensors":
        st = safetensors.inspect(path)
        rep["detail"] = st.as_dict()
        rep["findings"] += st.problems
        rep["verdict"] = st.verdict
    elif s.fmt in ("pickle", "zip-torch"):
        verdicts = rce.scan_file(path)
        rep["detail"] = [v.as_dict() for v in verdicts]
        for v in verdicts:
            rep["findings"] += [f"{v.source}: suspicious {x}" for x in v.suspicious]
            if v.verdict == "dangerous":
                rep["verdict"] = "dangerous"
            elif v.verdict == "review" and rep["verdict"] == "clean":
                rep["verdict"] = "review"
    elif s.fmt == "gguf":
        rep["detail"] = read_gguf_head(path)
    elif s.fmt == "unknown":
        rep["verdict"] = "review"
        rep["findings"].append("unrecognised container")
    return rep


def read_gguf_head(path, max_kv=12):
    """Read the GGUF metadata header. No inference, no mmap, no model load."""
    import struct
    out = {"version": None, "tensor_count": None, "kv": {}}
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            return {"error": "not GGUF"}
        (ver,) = struct.unpack("<I", fh.read(4))
        out["version"] = ver
        tt, nkv = struct.unpack("<QQ", fh.read(16))
        out["tensor_count"] = tt
        names = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
                 6: "f32", 7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def rd_str():
            (ln,) = struct.unpack("<Q", fh.read(8))
            return fh.read(ln).decode("utf-8", "replace")

        def rd_val(t, depth=0):
            if t == 8:
                return rd_str()
            if t == 9 and depth < 3:
                (et,) = struct.unpack("<I", fh.read(4))
                (cnt,) = struct.unpack("<Q", fh.read(8))
                return [rd_val(et, depth + 1) for _ in range(min(cnt, 4))]
            fh.read(sizes.get(t, 4))
            return None

        for _ in range(min(nkv, 256)):
            try:
                k = rd_str()
                (vt,) = struct.unpack("<I", fh.read(4))
                v = rd_val(vt)
            except Exception:
                break
            if len(out["kv"]) < max_kv or any(s in k for s in ("general", "tokenizer.chat_template")):
                out["kv"][k] = v
    return out


def audit_ollama(roots=None, hash_blobs=True, max_bytes=None):
    out = []
    for m in ollama.list_models(roots):
        if hash_blobs:
            m = ollama.verify_model(m, max_bytes=max_bytes)
        out.append({
            "name": m.name, "manifest": m.manifest, "total_bytes": m.total,
            "errors": m.errors, "verdict": m.verdict(),
            "blobs": [{"media": b.media, "digest": b.digest[:23],
                       "declared_mb": round(b.declared / 2**20, 2),
                       "sha256_match": b.sha_ok,
                       "zerofill": b.zerofill} for b in m.blobs],
        })
    return out


def run_numeric(files, base=None):
    """Numeric checks: LoRA structure always, GGUF differential when a base is given."""
    try:
        from . import numeric
    except ImportError as exc:
        return [{"error": f"numeric layer unavailable: {exc}"}]
    out = []
    for p in files:
        fmt = formats.sniff(p).fmt
        try:
            if fmt == "safetensors":
                rep = numeric.audit_lora(p)
                if rep.evidence.get("modules"):
                    out.append(rep.as_dict())
            elif fmt == "gguf":
                if base:
                    out.append(numeric.gguf_baseline_diff(p, base).as_dict())
                    out.append(numeric.gguf_embedding_outliers(p, base).as_dict())
                else:
                    out.append(numeric.gguf_embedding_outliers(p).as_dict())
        except Exception as exc:                 # a numeric check must never abort a scan
            out.append({"path": p, "kind": "numeric", "verdict": "review",
                        "findings": [f"{type(exc).__name__}: {exc}"], "evidence": {}})
    return out


def summarize(report):
    counts = {"clean": 0, "review": 0, "dangerous": 0}
    for r in report["artifacts"]:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in report.get("numeric", []):
        if "verdict" in r:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="weightsentry",
                                 description="Static supply-chain audit for local model artifacts.")
    ap.add_argument("paths", nargs="*", help="files or directories to audit")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-ollama", action="store_true", help="skip the Ollama store")
    ap.add_argument("--no-hash", action="store_true", help="skip blob re-hashing (fast)")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--numeric", action="store_true",
                    help="also run the numeric checks (needs numpy)")
    ap.add_argument("--base", metavar="GGUF",
                    help="base model to diff a finetuned GGUF against; enables the "
                         "strongest numeric check")
    ap.add_argument("--fail-on", choices=["never", "review", "dangerous"], default="dangerous",
                    help="exit non-zero at this verdict (default: dangerous)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    roots = a.paths or [os.path.expanduser("~")]
    files = discover(roots, a.max_depth)
    report = {
        "tool": "weightsentry", "version": __version__,
        "scanned_roots": roots,
        "artifacts": [audit_file(p) for p in files],
        "ollama": [] if a.no_ollama else audit_ollama(hash_blobs=not a.no_hash),
        "numeric": run_numeric(files, a.base) if (a.numeric or a.base) else [],
    }
    report["summary"] = summarize(report)

    if a.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_plain(report)

    rank = {"clean": 0, "review": 1, "dangerous": 2}
    worst = max([rank[r["verdict"]] for r in report["artifacts"]] +
                [rank.get(m["verdict"], 0) for m in report["ollama"]] +
                [rank.get(n.get("verdict"), 0) for n in report.get("numeric", [])] + [0])
    if a.fail_on == "never":
        return 0
    return 2 if worst > rank[a.fail_on] else 0


def print_plain(report):
    print("weightsentry %s" % report["version"])
    print("")
    print("%-9s %-12s %-11s %s" % ("verdict", "format", "size", "path"))
    print("%-9s %-12s %-11s %s" % ("-" * 9, "-" * 12, "-" * 11, "-" * 40))
    for r in report["artifacts"]:
        print("%-9s %-12s %-11s %s" % (r["verdict"], r["format"],
                                       "%.1f MB" % (r["size"] / 2**20), r["path"]))
        for f in r["findings"][:6]:
            print("          ! %s" % f)
    if not report["artifacts"]:
        print("(no model artifacts found under the given roots)")
    if report["ollama"]:
        print("")
        print("ollama store")
        print("%-9s %-40s %10s  %s" % ("verdict", "model", "size", "digests re-hashed"))
        for m in report["ollama"]:
            checked = "no" if any(b["sha256_match"] is None for b in m["blobs"]) else "yes"
            print("%-9s %-40s %8.2f GB  %s" % (m["verdict"], m["name"],
                                               m["total_bytes"] / 2**30, checked))
            for e in m["errors"]:
                print("          ! %s" % e)
            for b in m["blobs"]:
                if b["sha256_match"] is False:
                    why = "all-zero placeholder (interrupted pull)" if b.get("zerofill") \
                        else "content does not match its digest"
                    print("          ! %s: %s" % (b["media"], why))
    if report.get("numeric"):
        print("")
        print("numeric checks")
        for r in report["numeric"]:
            if "error" in r:
                print("  ! %s" % r["error"])
                continue
            print("%-9s %-16s %s" % (r["verdict"], r["kind"], r["path"]))
            ev = r.get("evidence") or {}
            if r["kind"] == "lora" and ev:
                print("          %d modules, rank %s, ||dW|| %.3e..%.3e, "
                      "top-1 energy %.0f%% median" % (
                          ev.get("modules", 0), ev.get("rank"),
                          ev.get("energy_min", 0), ev.get("energy_max", 0),
                          ev.get("top1_median", 0) * 100))
            if r["kind"] == "gguf-vs-base" and ev:
                print("          %d identical, %d changed, %d requantized, "
                      "%d float-noise only" % (
                          ev.get("same", 0), ev.get("changed", 0),
                          ev.get("requantized", 0), ev.get("float_noise_only", 0)))
            for f in r.get("findings", [])[:6]:
                print("          ! %s" % f)

    s = report["summary"]
    print("")
    print("summary: %d clean, %d review, %d dangerous" % (s["clean"], s["review"], s["dangerous"]))


if __name__ == "__main__":
    sys.exit(main())
