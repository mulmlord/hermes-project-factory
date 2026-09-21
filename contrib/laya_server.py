#!/usr/bin/env python3
"""A local classification service for the Hermes security gate.

Hermes talks to this over HTTP rather than importing it, because a classifier
wants torch and transformers and Hermes deliberately wants neither. Run it
beside the inference server, on the same machine, and point
``security.classifier.base_url`` at it.

    pip install laya            # python >= 3.10; pulls torch + transformers
    python contrib/laya_server.py --port 6768

It answers the contract in :mod:`hermes.classify`::

    GET  /health   -> {"ok": true, "model": "..."}
    POST /classify {"question": ..., "options": [...],
                    "items": [{"id": "c0", "text": "..."}]}
                -> {"results": [{"id": "c0", "label": "safe",
                                 "confidence": 0.97, "escalate": false}]}

Two things worth knowing before you trust its answers.

The base checkpoints are near chance zero-shot on decisions they were not
tuned for -- the model card puts them below a majority-class baseline -- so
out of the box this is a source of noise, not of findings. It is worth running
once you have fine-tuned a checkpoint on labelled vulnerability data
(Devign, Big-Vul, DiverseVul, Juliet/SARD are the usual corpora) and measured
its false positive rate on your own code. Point ``--model`` at that checkpoint.

The model ships over-confident; the card reports mean ECE moving from 0.466 to
0.081 only after temperature refitting, and the multilingual checkpoint ships
with no fitted temperature at all. ``security.classifier.threshold`` means
nothing until that is done.

``escalate`` is this adapter's own idea, not a head the model exposes: it
marks an answer that named a defect without clearing ``--escalate-below``, so
the gate can still put it in front of a reviewer. Laya's own guidance is to
branch on the calibrated probability exactly like this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The SDK deadlocks against TensorFlow's abseil init if transformers decides to
# import both. Has to be set before laya pulls transformers in.
os.environ.setdefault("USE_TF", "0")

#: Short descriptions for the option markers Hermes sends. The SDK wants a
#: label -> description mapping, and the description is most of the signal for
#: a label the checkpoint has never been tuned on.
DESCRIPTIONS = {
    "safe": "no exploitable security defect in this code",
    "command injection": "a shell or process invocation built from untrusted input",
    "unsafe deserialization": "pickle, yaml.load, marshal or similar on untrusted data",
    "path traversal": "a filesystem path built from input without containment",
    "hardcoded credential": "a key, token, password or secret literal in source",
    "unsafe eval": "eval, exec or compile reached by untrusted input",
    "server-side request forgery": "an outbound request to a caller-controlled address",
    "sql injection": "a query assembled by string concatenation or formatting",
    "weak cryptography": "broken hash or cipher, or a predictable source of randomness",
}


class Model:
    """The checkpoint, loaded once and reused for every request."""

    def __init__(self, name: str, device: str | None = None):
        import laya  # imported late: startup cost belongs to the operator

        self.name = name
        kwargs = {"device": device} if device else {}
        self.agent = laya.load(name, **kwargs)

    def classify(self, question: str, options: list[str],
                 items: list[dict], escalate_below: float) -> list[dict]:
        """One forward pass per chunk. Chunks are separate states, and the
        SDK batches questions about one state rather than many states, so the
        batching here is over the request rather than inside a predict call.
        """
        questions = {"vulnerability": {
            "type": "choice",
            "instructions": question,
            "criteria": {opt: DESCRIPTIONS.get(opt, opt) for opt in options},
        }}
        results = []
        for item in items:
            text = str(item.get("text") or "")
            if not text.strip():
                continue
            try:
                answer = self.agent.predict(text, questions)["answers"]["vulnerability"]
                label = str(answer["choice"])
                confidence = float(answer["confidence"])
            except Exception as exc:  # noqa: BLE001 - one bad chunk is not an outage
                print(f"classify failed for {item.get('id')}: {exc}", file=sys.stderr)
                continue
            results.append({
                "id": item.get("id"),
                "label": label,
                "confidence": confidence,
                "escalate": label != "safe" and confidence < escalate_below,
            })
        return results


def handler(model: Model, escalate_below: float):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ok": True, "model": model.name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/classify":
                self._send(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(size) or b"{}")
                items = [i for i in payload.get("items") or [] if isinstance(i, dict)]
                results = model.classify(
                    str(payload.get("question") or ""),
                    [str(o) for o in payload.get("options") or []],
                    items, escalate_below)
            except Exception as exc:  # noqa: BLE001 - answer with the reason
                self._send(400, {"error": str(exc)[:400]})
                return
            self._send(200, {"results": results})

        def log_message(self, fmt: str, *args) -> None:
            pass  # the access log is noise next to Hermes' own

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="checkpoint to load; use your fine-tuned one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6768)
    parser.add_argument("--device", default=None, help="cuda, cpu, ...")
    parser.add_argument("--escalate-below", type=float, default=0.85,
                        help="a named defect under this confidence is marked "
                             "for review rather than reported outright")
    args = parser.parse_args()

    print(f"loading {args.model} ...", file=sys.stderr)
    model = Model(args.model, args.device)
    server = ThreadingHTTPServer((args.host, args.port),
                                 handler(model, args.escalate_below))
    print(f"listening on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
