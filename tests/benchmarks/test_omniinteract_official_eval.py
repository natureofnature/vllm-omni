"""Tests for the upstream OmniInteract evaluator wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_official_eval_wrapper_builds_portable_dry_run(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "omniinteract" / "run_official_eval.py"
    official_repo = tmp_path / "official"
    (official_repo / "eval").mkdir(parents=True)
    (official_repo / "eval" / "run_eval.py").write_text("", encoding="utf-8")

    output_root = tmp_path / "outputs"
    sample_dir = output_root / "1q1a" / "videos__0001"
    sample_dir.mkdir(parents=True)
    (output_root / "batch_summary.json").write_text(
        json.dumps({"total": 1, "success": 1, "failed": 0, "results": []}),
        encoding="utf-8",
    )
    (output_root / "official_eval_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "1q1a__videos__0001",
                "gt_json": str(tmp_path / "annotations" / "0001.json"),
                "model_json": str(sample_dir / "wav_transcript.json"),
                "scene_type": "multi_turn",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-repo",
            str(official_repo),
            "--output-root",
            str(output_root),
            "--asr-model",
            "/models/Qwen3-ASR-1.7B",
            "--align-model",
            "/models/Qwen3-ForcedAligner-0.6B",
            "--num-workers",
            "3",
            "--dry-run",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "eval/data_prep/data_prep_batch.py" in completed.stdout
    assert "--num_workers 3" in completed.stdout
    assert "--fail_fast" in completed.stdout
    assert "--force_asr" in completed.stdout
    assert "--force_precise" in completed.stdout
    assert "eval/run_eval.py" in completed.stdout
    assert "--skip_existing" not in completed.stdout
    manifest = output_root / "official_eval_manifest.precise_truncation.jsonl"
    assert manifest.is_file()
    assert "precise_truncation.json" in manifest.read_text(encoding="utf-8")


def test_official_eval_wrapper_rejects_failed_benchmark_samples(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "omniinteract" / "run_official_eval.py"
    official_repo = tmp_path / "official"
    (official_repo / "eval").mkdir(parents=True)
    (official_repo / "eval" / "run_eval.py").write_text("", encoding="utf-8")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (output_root / "batch_summary.json").write_text(
        json.dumps({"total": 2, "success": 1, "failed": 1, "results": []}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-repo",
            str(official_repo),
            "--output-root",
            str(output_root),
            "--skip-data-prep",
            "--dry-run",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "failed samples" in completed.stderr


def test_official_eval_wrapper_checks_fresh_evaluator_summary(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "omniinteract" / "run_official_eval.py"
    official_repo = tmp_path / "official"
    (official_repo / "eval").mkdir(parents=True)
    (official_repo / "eval" / "run_eval.py").write_text(
        """import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--manifest')
p.add_argument('--out_dir')
p.add_argument('--num_workers')
p.add_argument('--judge_api_url')
p.add_argument('--judge_api_model')
p.add_argument('--judge_api_key')
a = p.parse_args()
assert os.environ['JUDGE_API_KEY'] == 'test-key'
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / 'unified_eval_summary.json').write_text(json.dumps({
    'summary': {'num_items': 1, 'failed_or_skipped': 0}
}))
""",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    sample_dir = output_root / "1q1a" / "videos__0001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "wav_transcript.json").write_text("{}", encoding="utf-8")
    annotation = tmp_path / "annotations" / "0001.json"
    annotation.parent.mkdir()
    annotation.write_text("[]", encoding="utf-8")
    (output_root / "batch_summary.json").write_text(
        json.dumps({"total": 1, "success": 1, "failed": 0, "results": []}),
        encoding="utf-8",
    )
    (output_root / "official_eval_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "1q1a__videos__0001",
                "gt_json": str(annotation),
                "model_json": str(sample_dir / "wav_transcript.json"),
                "scene_type": "multi_turn",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stale = output_root / "unified_eval" / "stale.unified_eval.json"
    stale.parent.mkdir()
    stale.write_text("{}", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-repo",
            str(official_repo),
            "--output-root",
            str(output_root),
            "--skip-data-prep",
            "--judge-api-key",
            "test-key",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not stale.exists()


def test_official_eval_wrapper_does_not_leak_judge_key_on_failure(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "omniinteract" / "run_official_eval.py"
    official_repo = tmp_path / "official"
    (official_repo / "eval").mkdir(parents=True)
    (official_repo / "eval" / "run_eval.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    sample_dir = output_root / "1q1a" / "videos__0001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "wav_transcript.json").write_text("{}", encoding="utf-8")
    (output_root / "batch_summary.json").write_text(
        json.dumps({"total": 1, "success": 1, "failed": 0, "results": []}),
        encoding="utf-8",
    )
    (output_root / "official_eval_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "1q1a__videos__0001",
                "gt_json": str(tmp_path / "annotations" / "0001.json"),
                "model_json": str(sample_dir / "wav_transcript.json"),
                "scene_type": "multi_turn",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    secret = "never-print-this-judge-key"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--official-repo",
            str(official_repo),
            "--output-root",
            str(output_root),
            "--skip-data-prep",
            "--judge-api-key",
            secret,
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert secret not in completed.stdout
    assert secret not in completed.stderr
