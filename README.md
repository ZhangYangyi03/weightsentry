weightsentry
============

Static supply-chain audit for the model artifacts sitting on your own disk.

It answers one question per file: if something loads this, can it run code on
me, and does what is inside match what the header claims? It never loads the
model, never imports torch, and never unpickles a single byte. Everything is
done by reading bytes and walking opcode streams -- which is the only way to
inspect a hostile checkpoint without executing it.

    pip install -e .
    weightsentry ~           # audit everything model-shaped under $HOME
    weightsentry ./models --json > report.json

What it checks
--------------

  format          three model containers can execute code by design.
                  weightsentry says which is which instead of calling them
                  all "model files".

                  gguf         inert: tensor + metadata container
                  safetensors  inert: 8-byte header + raw bytes
                  onnx         inert graph; custom op domains are the risk
                  zip-torch    zip whose data.pkl torch.load unpickles
                  pickle       raw pickle: the opcode stream is a program

  pickle RCE      every GLOBAL / STACK_GLOBAL / REDUCE / BUILD / NEWOBJ is
                  extracted and the named callables are classified. os.system,
                  subprocess.Popen, socket, urllib, requests, ctypes,
                  builtins.__import__ and friends are dangerous; an unexpected
                  callable outside the known-benign torch/numpy set is review.
                  Truncated or non-pickle input is review, never a crash.

  safetensors     header JSON, tensor span vs dtype*shape, overlaps, offsets
                  past EOF, bytes of trailing data after the last tensor, and
                  unexpected __metadata__ keys.

  ollama blob     every blob in ~/.ollama/models is re-hashed against the
                  digest its manifest declares. A flipped byte, an appended
                  byte, or a manifest pointing at a blob from elsewhere all
                  show up. Manifest-level mediaType is reported, so a model
                  that ships an adapter layer is visible.

Exit codes
----------

  0  nothing above --fail-on
  2  a finding at or above --fail-on (default: dangerous)

so it drops into CI as a gate.

    weightsentry ~ --fail-on review

Known limits, stated rather than hidden
---------------------------------------

  * Numeric backdoors (BadNets, trojan triggers, sleeper agents) are invisible
    to this tool. A weight file that behaves differently on a trigger is
    structurally identical to one that does not. Detecting that needs trigger
    inversion or pruning-differential analysis against a reference model, which
    is a different tool with a different cost model.
  * Reloading trust is out of scope: it verifies that a blob matches its
    manifest, not that the digest came from the publisher you think it did.
    Pin upstream digests separately.
  * ONNX custom op domains can pull in native code at runtime; weightsentry
    reports the format but does not resolve op domain implementations.
  * GGUF is parsed for header metadata only.

License: Apache-2.0.

A real run, for calibration
---------------------------

On a Windows machine with two Qwen2.5 LoRA adapters, one merged 7B GGUF, two
Silero VAD ONNX models, and five Ollama models, every file was clean and one
finding was real:

    dangerous  safetensors  58.9 MB   .../paper4_CoLoRA/adapters/math_lora/adapter_model.safetensors
               ! tensors run 99765512B past EOF

That file has the same 392-tensor header as its sibling code_lora adapter,
which is 161,533,192 bytes and clean. math_lora is 61,767,680 bytes -- a
truncated write. Nothing was executed; the header arithmetic said so.

And on the Ollama side, three of five models were all-zero placeholders:
qwen2.5:0.5b, qwen2.5:3b and qwen2.5:14b each carry a `model` blob of exactly
the declared size filled entirely with 0x00. They fail their manifest digest,
but they are not tampered with -- they are interrupted pulls. That difference
is why the report says "all-zero placeholder (interrupted pull)" instead of
"tampered": a tool that cries wolf on a bad download teaches people to ignore
it. `ollama rm` + `ollama pull` fixes all three.

The full report is in examples/REPORT_example.json.

What this run does NOT claim
----------------------------

No numeric backdoor was found, and this tool could not have found one. What it
establishes is narrower and worth stating exactly: none of these artifacts can
execute code when loaded, and the one that is broken is broken visibly.
