# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Download the fixed Seed-TTS subset; measurement uses vllm bench serve."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent
manifest = json.loads((root / "input-manifest.json").read_text())
prefix = (
    "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval/resolve/"
    + manifest["dataset_revision"] + "/en/"
)
with urllib.request.urlopen(prefix + "meta.lst", timeout=90) as response:
    lines = response.read().decode().splitlines()
by_id = {line.split("|")[0].strip(): line for line in lines if "|" in line}
hashes = {row["id"]: row["sha256"] for row in manifest["references"]}
for split, ids in (("evaluation", manifest["evaluation_ids"]),
                   ("warmup", manifest["warmup_ids"])):
    dest = root / "inputs" / split / "en"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.lst").write_text("\n".join(by_id[i] for i in ids) + "\n")
    for row_id in ids:
        relative = by_id[row_id].split("|")[2].strip()
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(prefix + relative, timeout=90) as response:
                path.write_bytes(response.read())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[row_id], row_id
print("Prepared 64 evaluation and 32 disjoint warmup requests with verified reference audio.")
