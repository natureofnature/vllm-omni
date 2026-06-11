"""Unit tests for OmniInteract benchmark data/eval modules."""

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path

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


@pytest.fixture()
def omniinteract_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    # 1q1a subset
    s1 = root / "1q1a"
    (s1 / "videos").mkdir(parents=True)
    (s1 / "annotations").mkdir(parents=True)
    (s1 / "subvideos").mkdir(parents=True)
    (s1 / "videos" / "0001.mp4").write_bytes(b"fake-mp4")
    (s1 / "subvideos" / "0001_0.mp4").write_bytes(b"fake-subvideo")
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

    # empty subset dirs to satisfy default subset list
    (root / "1q1a_math" / "videos").mkdir(parents=True)
    (root / "1q1a_math" / "annotations").mkdir(parents=True)
    (root / "1q1a_math" / "video_json_map.json").write_text(json.dumps({"total": 0, "entries": []}), encoding="utf-8")
    (root / "1qna" / "videos_bench").mkdir(parents=True)
    (root / "1qna" / "annotations").mkdir(parents=True)
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


def _write_minimal_1q1a_tree(root: Path) -> None:
    s1 = root / "1q1a"
    (s1 / "videos").mkdir(parents=True)
    (s1 / "annotations").mkdir(parents=True)
    (s1 / "subvideos").mkdir(parents=True)
    (s1 / "videos" / "0001.mp4").write_bytes(b"fake-mp4")
    (s1 / "subvideos" / "0001_0.mp4").write_bytes(b"fake-subvideo")
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


def test_omniinteract_dataset_builds_chat_requests(omniinteract_root: Path, mock_tokenizer):
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
    assert req.omniinteract_video == "subvideos/0001_0.mp4"
    assert req.omni_chat_messages is not None
    user_msg = req.omni_chat_messages[1]["content"]
    assert user_msg[0]["type"] == "video_url"
    assert user_msg[1]["type"] == "text"
    assert req.omni_extra_body == {"mm_processor_kwargs": {"use_audio_in_video": True}}


def test_omniinteract_dataset_aura_mode_sends_audio_and_video(omniinteract_root: Path, mock_tokenizer):
    audio_dir = omniinteract_root / "1q1a" / "audios"
    audio_dir.mkdir()
    (audio_dir / "0001_0.wav").write_bytes(b"fake-wav")
    ref_audio = omniinteract_root / "ref.wav"
    ref_audio.write_bytes(b"fake-ref-wav")
    ds = OmniInteractDataset(
        dataset_path=str(omniinteract_root),
        random_seed=0,
        disable_shuffle=True,
        input_mode="aura",
        aura_tts_language="English",
        aura_tts_ref_audio=str(ref_audio),
        aura_tts_ref_text="reference transcript",
    )
    reqs = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.omni_chat_messages is not None
    user_msg = req.omni_chat_messages[1]["content"]
    assert user_msg[0]["type"] == "audio_url"
    assert user_msg[1]["type"] == "video_url"
    assert len(user_msg) == 2
    assert req.omniinteract_video == "subvideos/0001_0.mp4"
    assert req.omni_extra_body is not None
    assert req.omni_extra_body["modalities"] == ["text", "audio"]
    assert req.omni_extra_body["mm_processor_kwargs"] == {"use_audio_in_video": False}
    assert len(req.omni_extra_body["sampling_params_list"]) == 4
    assert req.omni_extra_body["additional_information"]["tts_task_type"] == "Base"
    assert req.omni_extra_body["additional_information"]["tts_language"] == "English"
    assert req.omni_extra_body["additional_information"]["tts_ref_audio"] == str(ref_audio.resolve())
    assert req.omni_extra_body["additional_information"]["tts_ref_text"] == "reference transcript"


def test_omniinteract_dataset_aura_mode_passes_custom_voice_speaker(omniinteract_root: Path, mock_tokenizer):
    audio_dir = omniinteract_root / "1q1a" / "audios"
    audio_dir.mkdir()
    (audio_dir / "0001_0.wav").write_bytes(b"fake-wav")
    ds = OmniInteractDataset(
        dataset_path=str(omniinteract_root),
        random_seed=0,
        disable_shuffle=True,
        input_mode="aura",
        aura_tts_task_type="CustomVoice",
        aura_tts_language="English",
        aura_tts_speaker="Ethan",
    )

    [req] = ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)

    assert req.omni_extra_body is not None
    additional_info = req.omni_extra_body["additional_information"]
    assert additional_info["tts_task_type"] == "CustomVoice"
    assert additional_info["tts_language"] == "English"
    assert additional_info["tts_speaker"] == "Ethan"
    assert "tts_ref_audio" not in additional_info
    assert "tts_ref_text" not in additional_info


def test_omniinteract_dataset_aura_mode_requires_base_tts_refs(omniinteract_root: Path, mock_tokenizer):
    audio_dir = omniinteract_root / "1q1a" / "audios"
    audio_dir.mkdir()
    (audio_dir / "0001_0.wav").write_bytes(b"fake-wav")
    ds = OmniInteractDataset(
        dataset_path=str(omniinteract_root),
        random_seed=0,
        disable_shuffle=True,
        input_mode="aura",
    )

    with pytest.raises(ValueError, match="requires both"):
        ds.sample(mock_tokenizer, num_requests=1, no_oversample=True)


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


def test_omniinteract_dataset_infers_nested_roles(tmp_path: Path, mock_tokenizer):
    root = tmp_path / "nested_data"
    s1 = root / "1q1a"
    (s1 / "videos").mkdir(parents=True)
    (s1 / "annotations").mkdir(parents=True)
    (s1 / "subvideos").mkdir(parents=True)
    (s1 / "videos" / "0002.mp4").write_bytes(b"fake-mp4")
    (s1 / "subvideos" / "0002_0.mp4").write_bytes(b"fake-subvideo-0")
    (s1 / "subvideos" / "0002_1.mp4").write_bytes(b"fake-subvideo-1")
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
    (root / "1q1a_math" / "videos").mkdir(parents=True)
    (root / "1q1a_math" / "annotations").mkdir(parents=True)
    (root / "1q1a_math" / "video_json_map.json").write_text(json.dumps({"total": 0, "entries": []}), encoding="utf-8")
    (root / "1qna" / "videos_bench").mkdir(parents=True)
    (root / "1qna" / "annotations").mkdir(parents=True)

    ds = OmniInteractDataset(dataset_path=str(root), random_seed=0, disable_shuffle=True)
    reqs = ds.sample(mock_tokenizer, num_requests=2, no_oversample=True)
    assert len(reqs) == 2
    assert reqs[0].omniinteract_scene_type == "nested"
    assert reqs[1].omniinteract_scene_type == "nested"
    roles = {reqs[0].omniinteract_nested_role, reqs[1].omniinteract_nested_role}
    assert roles == {"outer", "inner"}
