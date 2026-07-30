"""OmniInteract full-duplex dataset loader for ``vllm bench serve --omni``.

OmniInteract evaluates continuous streaming interaction over full videos:
https://github.com/Lucky-Lance/OmniInteract

One benchmark request is one full-video duplex session:

- ``1q1a`` / ``1q1a_math``: each ``video_json_map.json`` entry becomes one session
  over the full source video, with all annotation QA slots attached for eval.
- ``1qna``: each ``videos_bench/**/*.mp4`` becomes one continuous session with
  its matching annotation slots.

This matches the official MiniCPM-o batch path, which streams second-by-second
PCM/frames from the full video rather than independent clip requests.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import get_cached_tokenizer

logger = logging.getLogger(__name__)

OmniInteractSubset = Literal["1q1a", "1q1a_math", "1qna"]
_SUPPORTED_SUBSETS = {"1q1a", "1q1a_math", "1qna"}


@dataclass(frozen=True)
class OmniInteractQASlot:
    """Ground-truth interaction slot inside a continuous video session."""

    slot_index: int
    question_text: str
    answer_text: str
    question_time: str = ""
    answer_time: str = ""
    question_type: str = ""
    is_interrupted: bool | None = None
    nested_group_id: int | None = None
    nested_role: str = ""
    question_time_s: float | None = None
    answer_time_s: float | None = None
    subset: str = ""
    video_rel: str = ""
    scene_type: str = ""


@dataclass
class OmniInteractSampleRequest(SampleRequest):
    """One full-video duplex session plus attached QA slots."""

    omniinteract_gold_answer: str = ""
    omniinteract_subset: str = ""
    omniinteract_question_type: str = ""
    omniinteract_video: str = ""
    omniinteract_question_time: str = ""
    omniinteract_answer_time: str = ""
    omniinteract_is_interrupted: bool | None = None
    omniinteract_scene_type: str = ""
    omniinteract_nested_group_id: int | None = None
    omniinteract_nested_role: str = ""
    omniinteract_profile: str = "realtime"
    omniinteract_official_compatible: bool = False
    omniinteract_session_key: str = ""
    omniinteract_video_path: str = ""
    omniinteract_slots: list[OmniInteractQASlot] = field(default_factory=list)
    omniinteract_realtime_chunk_ms: int = 200
    omniinteract_realtime_video_fps: float = 1.0
    omniinteract_realtime_ref_audio: str | None = None
    omniinteract_realtime_pace: bool = True
    omniinteract_realtime_timeout_s: float = 120.0


@dataclass
class _OmniInteractSession:
    subset: str
    video_rel: str
    video_path: Path
    scene_type: str
    slots: list[OmniInteractQASlot]


def _parse_time_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        return minutes * 60.0 + seconds
    hours, minutes, seconds = nums
    return hours * 3600.0 + minutes * 60.0 + seconds


def _infer_nested_roles(ann: list[dict[str, Any]]) -> dict[int, tuple[int, str]]:
    rows: list[tuple[int, float, float]] = []
    for idx, qa in enumerate(ann):
        q_time = _parse_time_seconds(qa.get("question_time"))
        a_time = _parse_time_seconds(qa.get("answer_time"))
        if q_time is None or a_time is None:
            continue
        rows.append((idx, q_time, a_time))
    rows.sort(key=lambda r: (r[1], r[2], r[0]))

    nested_meta: dict[int, tuple[int, str]] = {}
    group_id = 1
    cursor = 0
    while cursor < len(rows):
        outer_idx, outer_q, outer_a = rows[cursor]
        inner_pos: int | None = None
        for cand_pos in range(cursor + 1, len(rows)):
            _, cand_q, cand_a = rows[cand_pos]
            if outer_q < cand_q < outer_a and cand_a <= outer_a:
                inner_pos = cand_pos
                break
        if inner_pos is None:
            cursor += 1
            continue
        inner_idx = rows[inner_pos][0]
        nested_meta[outer_idx] = (group_id, "outer")
        nested_meta[inner_idx] = (group_id, "inner")
        group_id += 1
        cursor = inner_pos + 1
    return nested_meta


def _slots_from_annotation(
    ann: list[dict[str, Any]],
    *,
    subset: str,
    video_rel: str,
    scene_type: str,
) -> list[OmniInteractQASlot]:
    nested_meta = _infer_nested_roles(ann) if scene_type == "nested" else {}
    slots: list[OmniInteractQASlot] = []
    for qa_idx, qa in enumerate(ann):
        q = str(qa.get("question_text") or "").strip()
        a = str(qa.get("answer_text") or "").strip()
        if not q or not a:
            continue
        nested_group_id, nested_role = nested_meta.get(qa_idx, (None, ""))
        question_time = str(qa.get("question_time") or "").strip()
        answer_time = str(qa.get("answer_time") or "").strip()
        slots.append(
            OmniInteractQASlot(
                slot_index=qa_idx,
                question_text=q,
                answer_text=a,
                question_time=question_time,
                answer_time=answer_time,
                question_type=str(qa.get("question_type") or "").strip(),
                is_interrupted=qa.get("is_interrupted"),
                nested_group_id=nested_group_id,
                nested_role=nested_role,
                question_time_s=_parse_time_seconds(question_time),
                answer_time_s=_parse_time_seconds(answer_time),
                subset=subset,
                video_rel=video_rel,
                scene_type=scene_type,
            )
        )
    return slots


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser().resolve()


def _tar_fingerprint(tar_path: Path) -> str:
    st = tar_path.stat()
    return f"v1:{st.st_size}:{int(st.st_mtime_ns)}"


def _resolve_data_dir_under(root: Path) -> Path:
    probe = root / "1q1a"
    if probe.is_dir():
        return root
    probe = root / "data" / "1q1a"
    if probe.is_dir():
        return root / "data"
    raise FileNotFoundError(f"Could not locate OmniInteract data dir under: {root}")


def _extract_tar_archive(tar_path: Path, cache_root: Path) -> Path:
    extracted_root = cache_root / "extracted"
    marker = cache_root / ".extracted"
    fp = _tar_fingerprint(tar_path)
    if marker.is_file() and extracted_root.is_dir():
        try:
            if marker.read_text(encoding="utf-8").strip() == fp:
                data_dir = _resolve_data_dir_under(extracted_root)
                logger.info("Reusing cached OmniInteract media at %s", data_dir)
                return data_dir
        except Exception:
            shutil.rmtree(extracted_root, ignore_errors=True)
            marker.unlink(missing_ok=True)

    cache_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting OmniInteract archive %s", tar_path)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=extracted_root, filter="data")
    marker.write_text(fp, encoding="utf-8")
    return _resolve_data_dir_under(extracted_root)


def _ensure_extracted_data_dir(root: Path) -> Path:
    try:
        return _resolve_data_dir_under(root)
    except FileNotFoundError:
        pass
    for name in ("data.tar.gz", "data.tar"):
        tar_path = root / name
        if tar_path.is_file():
            cache_root = root / ".vllm_omni_omniinteract_extracted"
            return _extract_tar_archive(tar_path, cache_root)
    raise FileNotFoundError(
        f"Could not locate OmniInteract data under {root}. "
        "Expected extracted 1q1a/ (or data/1q1a/) or data.tar.gz in that directory."
    )


def resolve_omniinteract_root(
    dataset_path: str | None = None,
    *,
    explicit_root: str | Path | None = None,
) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--omniinteract-root is not a directory: {root}")
        return _ensure_extracted_data_dir(root)
    if not dataset_path:
        raise ValueError("OmniInteract requires --dataset-path (HF repo id or local directory) or --omniinteract-root.")
    p = Path(dataset_path).expanduser()
    if p.exists() and p.is_dir():
        return _ensure_extracted_data_dir(p.resolve())
    return ensure_omniinteract_data_dir(dataset_path.strip())


def ensure_omniinteract_data_dir(repo_id: str) -> Path:
    rid = (repo_id or "").strip()
    if not rid:
        raise ValueError("repo_id is required for OmniInteract HF download")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "OmniInteract HF download requires huggingface_hub. "
            "Install it or pass --omniinteract-root with local extracted data."
        ) from e

    safe = rid.replace("/", "__").replace("\\", "_")
    cache_root = _hf_cache_root() / "vllm_omni" / "omniinteract_media" / safe
    tar_path: Path | None = None
    for name in ("data.tar.gz", "data.tar"):
        try:
            tar_path = Path(hf_hub_download(repo_id=rid, filename=name, repo_type="dataset"))
            break
        except Exception:
            continue
    if tar_path is None or not tar_path.is_file():
        raise FileNotFoundError(f"Could not download data.tar.gz from dataset repo {rid!r}.")
    return _extract_tar_archive(tar_path, cache_root)


class OmniInteractDataset(BenchmarkDataset):
    """OmniInteract full-video duplex sessions."""

    SUPPORTED_DATASET_PATHS: set[str] = {"lucky-lance/OmniInteract"}
    DEFAULT_HF_DATASET_ID = "lucky-lance/OmniInteract"
    DEFAULT_OUTPUT_LEN = 256
    IS_MULTIMODAL = True

    def __init__(
        self,
        dataset_path: str | None = None,
        random_seed: int = 0,
        data_root: str | None = None,
        subsets: list[OmniInteractSubset] | None = None,
        **kwargs: Any,
    ) -> None:
        self.dataset_path = dataset_path or self.DEFAULT_HF_DATASET_ID
        self.data_root_input = Path(data_root).expanduser().resolve() if data_root else None
        self.subsets = list(subsets or ["1q1a", "1q1a_math", "1qna"])
        unsupported = sorted(set(self.subsets) - _SUPPORTED_SUBSETS)
        if unsupported:
            raise ValueError(f"Unsupported OmniInteract subsets: {unsupported}. Expected: {sorted(_SUPPORTED_SUBSETS)}")
        self._data_root: Path | None = None
        self._sessions: list[_OmniInteractSession] = []

        super().__init__(
            dataset_path=self.dataset_path,
            random_seed=random_seed,
            **kwargs,
        )
        self.load_data()

    def _resolve_data_root(self) -> Path:
        if self._data_root is not None:
            return self._data_root
        dataset_ref = None if self.data_root_input is not None else self.dataset_path
        self._data_root = resolve_omniinteract_root(
            dataset_ref,
            explicit_root=self.data_root_input,
        )
        return self._data_root

    @staticmethod
    def _read_json(path: Path) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _iter_subset_sessions(self, data_root: Path, subset: OmniInteractSubset) -> list[_OmniInteractSession]:
        sessions: list[_OmniInteractSession] = []
        subset_root = data_root / subset
        if not subset_root.is_dir():
            logger.warning("Subset directory does not exist: %s", subset_root)
            return sessions

        if subset in ("1q1a", "1q1a_math"):
            map_path = subset_root / "video_json_map.json"
            map_data = self._read_json(map_path) if map_path.is_file() else {"entries": []}
            for item in map_data.get("entries", []):
                video_rel = str(item.get("video") or "").strip()
                ann_rel = str(item.get("annotation") or "").strip()
                scene_type = str(item.get("scene_type") or "multi_turn").strip().lower() or "multi_turn"
                if not video_rel or not ann_rel:
                    continue
                video_path = subset_root / video_rel
                ann_path = subset_root / ann_rel
                if not video_path.is_file() or not ann_path.is_file():
                    continue
                ann = self._read_json(ann_path)
                if not isinstance(ann, list):
                    continue
                slots = _slots_from_annotation(
                    ann,
                    subset=subset,
                    video_rel=video_rel,
                    scene_type=scene_type,
                )
                if not slots:
                    continue
                sessions.append(
                    _OmniInteractSession(
                        subset=subset,
                        video_rel=video_rel,
                        video_path=video_path,
                        scene_type=scene_type,
                        slots=slots,
                    )
                )
            return sessions

        # Official OmniInteract 1qna: one continuous session per videos_bench mp4.
        ann_root = subset_root / "annotations"
        video_root = subset_root / "videos_bench"
        if not ann_root.is_dir() or not video_root.is_dir():
            return sessions
        for video_path in sorted(video_root.rglob("*.mp4")):
            rel = video_path.relative_to(video_root)
            ann_path = (ann_root / rel).with_suffix(".json")
            if not ann_path.is_file():
                continue
            ann = self._read_json(ann_path)
            if not isinstance(ann, list):
                continue
            video_rel = str(video_path.relative_to(subset_root))
            slots = _slots_from_annotation(
                ann,
                subset=subset,
                video_rel=video_rel,
                scene_type="1qna",
            )
            if not slots:
                continue
            sessions.append(
                _OmniInteractSession(
                    subset=subset,
                    video_rel=video_rel,
                    video_path=video_path,
                    scene_type="1qna",
                    slots=slots,
                )
            )
        return sessions

    def load_data(self) -> None:
        root = self._resolve_data_root()
        all_sessions: list[_OmniInteractSession] = []
        for subset in self.subsets:
            all_sessions.extend(self._iter_subset_sessions(root, subset))
        if not all_sessions:
            raise ValueError(f"No OmniInteract duplex sessions found under {root} (subsets={self.subsets})")
        if not getattr(self, "disable_shuffle", False):
            import random

            rng = random.Random(self.random_seed)
            rng.shuffle(all_sessions)
        self._sessions = all_sessions
        self.data = self._sessions
        logger.info(
            "Loaded OmniInteract duplex sessions: root=%s subsets=%s sessions=%d slots=%d",
            root,
            self.subsets,
            len(all_sessions),
            sum(len(session.slots) for session in all_sessions),
        )

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int,
        output_len: int | None = None,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if output_len is None:
            output_len = self.DEFAULT_OUTPUT_LEN
        tok = get_cached_tokenizer(tokenizer)
        out: list[SampleRequest] = []
        for session_index, session in enumerate(self._sessions):
            if len(out) >= num_requests:
                break
            if not session.video_path.is_file():
                continue
            head = session.slots[0]
            prompt = f"OmniInteract duplex session: {session.subset}/{session.video_rel}"
            out.append(
                OmniInteractSampleRequest(
                    prompt=prompt,
                    prompt_len=len(tok.encode(prompt)),
                    expected_output_len=output_len,
                    multi_modal_data=None,
                    request_id=f"{request_id_prefix}{session_index}",
                    omniinteract_gold_answer=head.answer_text,
                    omniinteract_subset=session.subset,
                    omniinteract_question_type=head.question_type,
                    omniinteract_video=session.video_rel,
                    omniinteract_question_time=head.question_time,
                    omniinteract_answer_time=head.answer_time,
                    omniinteract_is_interrupted=head.is_interrupted,
                    omniinteract_scene_type=session.scene_type,
                    omniinteract_nested_group_id=head.nested_group_id,
                    omniinteract_nested_role=head.nested_role,
                    omniinteract_profile="realtime",
                    omniinteract_official_compatible=False,
                    omniinteract_session_key=f"{session.subset}:{session.video_rel}",
                    omniinteract_video_path=str(session.video_path.resolve()),
                    omniinteract_slots=list(session.slots),
                )
            )
        self.maybe_oversample_requests(out, num_requests, request_id_prefix, no_oversample)
        return out
