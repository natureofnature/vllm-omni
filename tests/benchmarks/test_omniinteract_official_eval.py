"""Contract tests for the optional upstream OmniInteract evaluator wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    official, output = tmp_path / "official", tmp_path / "outputs"
    (official / "eval").mkdir(parents=True)
    (official / "eval" / "run_eval.py").write_text("")
    sample = output / "1q1a" / "videos__0001"
    sample.mkdir(parents=True)
    (sample / "wav_transcript.json").write_text("{}")
    annotation = tmp_path / "annotations" / "0001.json"
    annotation.parent.mkdir()
    annotation.write_text("[]")
    (output / "batch_summary.json").write_text(json.dumps({"total": 1, "success": 1, "failed": 0, "results": []}))
    (output / "official_eval_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "1q1a__videos__0001",
                "gt_json": str(annotation),
                "model_json": str(sample / "wav_transcript.json"),
                "scene_type": "multi_turn",
            }
        )
        + "\n"
    )
    return official, output, Path(__file__).resolve().parents[2] / "benchmarks/omniinteract/run_official_eval.py"


def _run(script: Path, official: Path, output: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-repo",
            str(official),
            "--output-root",
            str(output),
            *args,
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_official_eval_dry_run_builds_portable_pipeline(tmp_path: Path):
    official, output, script = _layout(tmp_path)
    result = _run(
        script,
        official,
        output,
        "--asr-model",
        "/models/asr",
        "--align-model",
        "/models/align",
        "--num-workers",
        "3",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert all(
        item in result.stdout
        for item in ("data_prep_batch.py", "--num_workers 3", "--force_asr", "--force_precise", "run_eval.py")
    )
    assert "precise_truncation.json" in (output / "official_eval_manifest.precise_truncation.jsonl").read_text()


def test_official_eval_fails_closed_and_hides_secret(tmp_path: Path):
    official, output, script = _layout(tmp_path)
    summary = output / "batch_summary.json"
    summary.write_text(json.dumps({"total": 2, "success": 1, "failed": 1, "results": []}))
    assert _run(script, official, output, "--skip-data-prep", "--dry-run").returncode != 0

    summary.write_text(json.dumps({"total": 1, "success": 1, "failed": 0, "results": []}))
    stale = output / "unified_eval" / "stale.unified_eval.json"
    stale.parent.mkdir()
    stale.write_text("{}")
    (official / "eval" / "run_eval.py").write_text("raise SystemExit(7)\n")
    secret = "never-print-this-judge-key"
    result = _run(script, official, output, "--skip-data-prep", "--judge-api-key", secret)
    assert result.returncode != 0 and secret not in result.stdout + result.stderr

    stale.parent.mkdir(exist_ok=True)
    stale.write_text("{}")
    evaluator = "import json,sys;from pathlib import Path;out=Path(sys.argv[sys.argv.index('--out_dir')+1]);"
    evaluator += "out.mkdir(parents=True,exist_ok=True);(out/'unified_eval_summary.json').write_text("
    evaluator += "json.dumps({'summary':{'num_items':1,'failed_or_skipped':0}}))"
    (official / "eval" / "run_eval.py").write_text(evaluator)
    assert _run(script, official, output, "--skip-data-prep", "--judge-api-key", secret).returncode == 0
    assert not stale.exists()
