"""Unit tests for OmniInteract duplex dataset/eval modules."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DS_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "omniinteract_dataset.py"
_DS_MODULE_NAME = "vllm_omni.benchmarks.data_modules.omniinteract_dataset"
if _DS_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_DS_MODULE_NAME, _DS_MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_DS_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

_EVAL_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "omniinteract_eval.py"
_EVAL_MODULE_NAME = "vllm_omni.benchmarks.data_modules.omniinteract_eval"
if _EVAL_MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_EVAL_MODULE_NAME, _EVAL_MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_EVAL_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (  # noqa: E402
    OmniInteractDataset,
    OmniInteractSampleRequest,
    resolve_omniinteract_root,
)
from vllm_omni.benchmarks.data_modules.omniinteract_eval import (  # noqa: E402
    compute_omniinteract_metrics,
    print_omniinteract_summary,
)
from vllm_omni.benchmarks.patch.patch import (  # noqa: E402
    _attach_omniinteract_to_request_func_input,
    get_samples,
)


def _write_minimal_1q1a_tree(root: Path) -> None:
    s1 = root / "1q1a"
    (s1 / "videos").mkdir(parents=True)
    (s1 / "annotations").mkdir(parents=True)
    (s1 / "videos" / "0001.mp4").write_bytes(b"fake-mp4")
    (s1 / "annotations" / "0001.json").write_text(
        json.dumps(
            [
                {
                    "question_time": "00:01",
                    "question_text": "What color is the cup?",
                    "answer_time": "00:04",
                    "answer_text": "red",
                    "question_type": "realtime",
                    "is_interrupted": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    (s1 / "video_json_map.json").write_text(
        json.dumps(
            {
                "total": 1,
                "entries": [
                    {
                        "video": "videos/0001.mp4",
                        "annotation": "annotations/0001.json",
                        "scene_type": "multi_turn",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_empty_optional_subsets(root: Path) -> None:
    (root / "1q1a_math" / "videos").mkdir(parents=True)
    (root / "1q1a_math" / "annotations").mkdir(parents=True)
    (root / "1q1a_math" / "video_json_map.json").write_text(json.dumps({"total": 0, "entries": []}), encoding="utf-8")
    (root / "1qna" / "videos_bench").mkdir(parents=True)
    (root / "1qna" / "annotations").mkdir(parents=True)


@pytest.fixture()
def omniinteract_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    _write_minimal_1q1a_tree(root)
    _write_empty_optional_subsets(root)
    return root


@pytest.fixture()
def mock_tokenizer(mocker):
    tok = mocker.MagicMock()
    tok.encode = lambda text, **kw: [0] * max(1, len(text.split()))
    tok.get_vocab.return_value = {"<pad>": 0}
    tok.all_special_ids = []
    tok.all_special_tokens = []
    tok.vocab_size = 1
    tok.__len__.return_value = 1
    return tok


def test_resolve_omniinteract_root_from_local_dataset_path(tmp_path: Path):
    data_root = tmp_path / "local_dataset"
    _write_minimal_1q1a_tree(data_root)
    resolved = resolve_omniinteract_root(str(data_root))
    assert resolved == data_root.resolve()


def test_resolve_omniinteract_root_extracts_local_tarball(tmp_path: Path):
    data_root = tmp_path / "archive_only"
    data_root.mkdir()
    payload = tmp_path / "payload"
    _write_minimal_1q1a_tree(payload)
    tar_path = data_root / "data.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for path in payload.rglob("*"):
            tf.add(path, arcname=path.relative_to(payload))
    resolved = resolve_omniinteract_root(str(data_root))
    assert (resolved / "1q1a" / "video_json_map.json").is_file()


def test_omniinteract_dataset_builds_full_video_duplex_sessions(omniinteract_root: Path, mock_tokenizer):
    ds = OmniInteractDataset(
        dataset_path=str(omniinteract_root),
        random_seed=0,
        disable_shuffle=True,
    )
    reqs = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)
    assert len(reqs) == 1
    req = reqs[0]
    assert isinstance(req, OmniInteractSampleRequest)
    assert req.omniinteract_subset == "1q1a"
    assert req.omniinteract_gold_answer == "red"
    assert req.omniinteract_scene_type == "multi_turn"
    assert req.omniinteract_video == "videos/0001.mp4"
    assert req.omniinteract_profile == "realtime"
    assert len(req.omniinteract_slots) == 1
    assert req.omniinteract_slots[0].answer_text == "red"
    assert req.omniinteract_slots[0].subset == "1q1a"
    assert req.omniinteract_slots[0].video_rel == "videos/0001.mp4"
    assert req.omniinteract_video_path.endswith("videos/0001.mp4")
    assert req.omniinteract_annotation_path.endswith("annotations/0001.json")


def test_omniinteract_dataset_rejects_unknown_subset(omniinteract_root: Path):
    with pytest.raises(ValueError, match="Unsupported OmniInteract subsets"):
        OmniInteractDataset(
            dataset_path=str(omniinteract_root),
            subsets=["unknown"],  # type: ignore[list-item]
        )


def test_omniinteract_dataset_attaches_realtime_session_fields(omniinteract_root: Path, mock_tokenizer):
    ds = OmniInteractDataset(
        dataset_path=str(omniinteract_root),
        random_seed=0,
        disable_shuffle=True,
    )
    [req] = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)
    request_input = SimpleNamespace(extra_body=None)
    _attach_omniinteract_to_request_func_input(req, request_input)
    assert request_input.omniinteract_slots == req.omniinteract_slots
    assert request_input.omniinteract_video_path == req.omniinteract_video_path
    assert request_input.omniinteract_annotation_path == req.omniinteract_annotation_path


def _omniinteract_benchmark_args(root: Path, **overrides):
    values = {
        "dataset_name": "omniinteract",
        "backend": "minicpmo-realtime",
        "dataset_path": str(root),
        "hf_name": None,
        "omniinteract_root": None,
        "omniinteract_subsets": "1q1a",
        "omniinteract_eval": False,
        "omniinteract_official_output_dir": None,
        "omniinteract_realtime_video_fps": 1.0,
        "omniinteract_realtime_no_pace": False,
        "omniinteract_realtime_chunk_ms": 200,
        "omniinteract_realtime_ref_audio": None,
        "omniinteract_realtime_timeout_s": 30.0,
        "no_oversample": True,
        "num_prompts": 1,
        "seed": 0,
        "request_id_prefix": "req-",
        "disable_shuffle": True,
        "output_len": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_omniinteract_benchmark_uses_unique_server_session_key(omniinteract_root: Path, mock_tokenizer):
    [request] = get_samples(_omniinteract_benchmark_args(omniinteract_root), mock_tokenizer)
    assert request.omniinteract_session_key == "1q1a:videos/0001.mp4:req-0"


def test_omniinteract_oversampling_keeps_server_sessions_isolated(omniinteract_root: Path, mock_tokenizer):
    requests = get_samples(
        _omniinteract_benchmark_args(omniinteract_root, num_prompts=3, no_oversample=False),
        mock_tokenizer,
    )

    assert len(requests) == 3
    assert len({request.request_id for request in requests}) == 3
    assert len({request.omniinteract_session_key for request in requests}) == 3


def test_omniinteract_accuracy_rejects_oversampling(omniinteract_root: Path, mock_tokenizer):
    args = _omniinteract_benchmark_args(
        omniinteract_root,
        omniinteract_official_output_dir="/tmp/official-output",
        no_oversample=False,
    )
    with pytest.raises(ValueError, match="require --no-oversample"):
        get_samples(args, mock_tokenizer)


def test_omniinteract_accuracy_rejects_unpaced_mode(omniinteract_root: Path, mock_tokenizer):
    args = _omniinteract_benchmark_args(
        omniinteract_root,
        omniinteract_official_output_dir="/tmp/official-output",
        omniinteract_realtime_no_pace=True,
    )
    with pytest.raises(ValueError, match="require realtime pacing"):
        get_samples(args, mock_tokenizer)


def test_omniinteract_realtime_rejects_nonofficial_proxy_eval(omniinteract_root: Path, mock_tokenizer):
    args = _omniinteract_benchmark_args(omniinteract_root, omniinteract_eval=True)
    with pytest.raises(ValueError, match="does not implement the official continuous-session"):
        get_samples(args, mock_tokenizer)


def test_omniinteract_1qna_loads_continuous_videos_bench_sessions(tmp_path: Path, mock_tokenizer):
    root = tmp_path / "data"
    _write_empty_optional_subsets(root)
    (root / "1q1a" / "videos").mkdir(parents=True, exist_ok=True)
    (root / "1q1a" / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "1q1a" / "video_json_map.json").write_text(json.dumps({"total": 0, "entries": []}), encoding="utf-8")

    video = root / "1qna" / "videos_bench" / "scene_a" / "clip.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-1qna-mp4")
    ann = root / "1qna" / "annotations" / "scene_a" / "clip.json"
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(
        json.dumps(
            [
                {
                    "question_time": "00:02",
                    "question_text": "What happens next?",
                    "answer_time": "00:05",
                    "answer_text": "a jump",
                    "question_type": "realtime",
                    "is_interrupted": False,
                },
                {
                    "question_time": "00:08",
                    "question_text": "And then?",
                    "answer_time": "00:10",
                    "answer_text": "a fall",
                    "question_type": "realtime",
                    "is_interrupted": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    ds = OmniInteractDataset(
        dataset_path=str(root),
        random_seed=0,
        disable_shuffle=True,
        subsets=["1qna"],
    )
    [req] = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)
    assert req.omniinteract_subset == "1qna"
    assert req.omniinteract_scene_type == "1qna"
    assert req.omniinteract_video.endswith("videos_bench/scene_a/clip.mp4")
    assert len(req.omniinteract_slots) == 2
    assert [slot.answer_text for slot in req.omniinteract_slots] == ["a jump", "a fall"]


def test_omniinteract_eval_counts_exact_and_soft_match():
    req = OmniInteractSampleRequest(
        prompt="q",
        prompt_len=1,
        expected_output_len=8,
        multi_modal_data=None,
        request_id="r0",
        omniinteract_gold_answer="red cup",
        omniinteract_subset="1q1a",
        omniinteract_question_type="realtime",
        omniinteract_video="videos/0001.mp4",
        omniinteract_profile="realtime",
    )

    class _Out:
        def __init__(self, success: bool, text: str, error: str = "") -> None:
            self.success = success
            self.generated_text = text
            self.error = error

    outputs = [_Out(True, "The answer is red cup.")]
    m = compute_omniinteract_metrics([req], outputs)
    assert m is not None
    assert m["omniinteract_evaluated"] == 1
    assert m["omniinteract_exact_count"] == 0
    assert m["omniinteract_soft_count"] == 1
    assert m["omniinteract_ia_qtf1"] == 1.0
    assert m["omniinteract_profiles"] == ["realtime"]
    assert m["omniinteract_official_compatible"] is False
    assert "omniinteract_ids" in m
    assert "omniinteract_nccs" in m


def test_omniinteract_summary_omits_legacy_match_metrics(capsys):
    metrics = {
        "omniinteract_evaluated": 1,
        "omniinteract_request_failed": 0,
        "omniinteract_exact_match": 0.0,
        "omniinteract_soft_match": 0.0,
        "omniinteract_ia_qtf1": 0.25,
        "omniinteract_ids": {
            "NOR": 0.5,
            "PAQ": 0.75,
            "CSM_SR": None,
            "CSM_AS_seconds": None,
        },
        "omniinteract_nccs": 0.0,
        "omniinteract_per_subset_exact": {"1q1a": 0.0},
        "omniinteract_per_subset": {"1q1a": {"exact": 0, "total": 1}},
    }

    print_omniinteract_summary(metrics)

    out = capsys.readouterr().out
    assert "HTTP failed:" not in out
    assert "Exact Match:" not in out
    assert "Soft Match (contains):" not in out
    assert "--- Exact Match by Subset ---" not in out
    assert "IDS.CSM-SR:" in out
    assert "IDS.CSM-AS(s):" in out


def test_omniinteract_ids_csm_uses_spill_timing_when_available():
    reqs = [
        OmniInteractSampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            multi_modal_data=None,
            request_id="r0",
            omniinteract_gold_answer="red",
            omniinteract_subset="1q1a",
            omniinteract_question_type="realtime",
            omniinteract_video="videos/0001.mp4",
            omniinteract_is_interrupted=True,
            omniinteract_profile="realtime",
        ),
        OmniInteractSampleRequest(
            prompt="q",
            prompt_len=1,
            expected_output_len=8,
            multi_modal_data=None,
            request_id="r1",
            omniinteract_gold_answer="blue",
            omniinteract_subset="1q1a",
            omniinteract_question_type="realtime",
            omniinteract_video="videos/0002.mp4",
            omniinteract_is_interrupted=True,
            omniinteract_profile="realtime",
        ),
    ]

    class _Out:
        def __init__(self, text: str, spill_seconds: float) -> None:
            self.success = True
            self.generated_text = text
            self.error = ""
            self.omniinteract_spill_seconds = spill_seconds

    metrics = compute_omniinteract_metrics(
        reqs,
        [_Out("red", 1.5), _Out("blue", 0.0)],
    )

    assert metrics is not None
    assert metrics["omniinteract_ids"]["CSM_SR"] == 0.5
    assert metrics["omniinteract_ids"]["CSM_AS_seconds"] == 0.75
    exp_interruption = metrics["omniinteract_paper_metrics"]["exp_interruption"]
    assert exp_interruption["interrupted_with_spill_timing_count"] == 2


def test_omniinteract_dataset_infers_nested_roles_in_one_session(tmp_path: Path, mock_tokenizer):
    root = tmp_path / "nested_data"
    s1 = root / "1q1a"
    (s1 / "videos").mkdir(parents=True)
    (s1 / "annotations").mkdir(parents=True)
    (s1 / "videos" / "0002.mp4").write_bytes(b"fake-mp4")
    (s1 / "annotations" / "0002.json").write_text(
        json.dumps(
            [
                {
                    "question_time": "00:01",
                    "question_text": "outer question?",
                    "answer_time": "00:10",
                    "answer_text": "outer answer",
                    "question_type": "proactive",
                    "is_interrupted": False,
                },
                {
                    "question_time": "00:03",
                    "question_text": "inner question?",
                    "answer_time": "00:05",
                    "answer_text": "inner answer",
                    "question_type": "realtime",
                    "is_interrupted": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    (s1 / "video_json_map.json").write_text(
        json.dumps(
            {
                "total": 1,
                "entries": [
                    {
                        "video": "videos/0002.mp4",
                        "annotation": "annotations/0002.json",
                        "scene_type": "nested",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_empty_optional_subsets(root)

    ds = OmniInteractDataset(dataset_path=str(root), random_seed=0, disable_shuffle=True, subsets=["1q1a"])
    reqs = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.omniinteract_scene_type == "nested"
    assert len(req.omniinteract_slots) == 2
    roles = {slot.nested_role for slot in req.omniinteract_slots}
    assert roles == {"outer", "inner"}
