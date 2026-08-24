# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract realtime benchmark dataset discovery."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.transformers_utils.repo_utils import hf_fs

from vllm_omni.diffusion.model_loader import hub_prefetch

OMNIINTERACT_SUBSETS = ("1q1a", "1q1a_math", "1qna")
DEFAULT_OMNIINTERACT_REPO = "lucky-lance/OmniInteract"


@dataclass(frozen=True)
class OmniInteractCase:
    """One video/annotation pair from the official dataset."""

    subset: str
    video_rel: str
    video_path: Path
    annotation_path: Path
    scene_type: str


def _data_dir(root: Path) -> Path:
    for candidate in (root, root / "data"):
        if (candidate / "1q1a").is_dir():
            return candidate
    raise FileNotFoundError(f"OmniInteract data not found under {root}")


def _confined_path(root: Path, relative: object, *, field: str) -> Path:
    text = str(relative or "")
    path = Path(text)
    destination = (root / path).resolve()
    resolved_root = root.resolve()
    if not text or path.is_absolute() or ".." in path.parts or not destination.is_relative_to(resolved_root):
        raise ValueError(f"Unsafe OmniInteract {field} path: {text!r}")
    return destination


def _safe_extract(handle: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    members = handle.getmembers()
    for member in members:
        path = Path(member.name)
        destination = (root / path).resolve()
        if (
            path.is_absolute()
            or ".." in path.parts
            or not destination.is_relative_to(root)
            or not (member.isdir() or member.isfile())
        ):
            raise ValueError(f"Unsafe path in OmniInteract archive: {member.name!r}")
    handle.extractall(target, members=members)


def _archive_fingerprint(archive: Path) -> str:
    stat = archive.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _archive_lock_key(target: Path) -> str:
    digest = hashlib.sha256(str(target.resolve()).encode()).hexdigest()
    return f".omniinteract-{digest}"


def _extract_archive_unlocked(archive: Path, target: Path) -> Path:
    if target.is_symlink():
        raise ValueError(f"Refusing to extract through symlink: {target}")
    marker = target / ".source"
    fingerprint = _archive_fingerprint(archive)
    if marker.is_file() and marker.read_text().strip() == fingerprint:
        try:
            return _data_dir(target)
        except FileNotFoundError:
            pass
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    with tarfile.open(archive, "r:*") as handle:
        _safe_extract(handle, target)
    marker.write_text(fingerprint)
    return _data_dir(target)


def _extract_archive(archive: Path, target: Path) -> Path:
    with hub_prefetch._repo_prefetch_lock(
        _archive_lock_key(target),
        required=True,
        lock_dir=target.parent,
    ):
        return _extract_archive_unlocked(archive, target)


def resolve_omniinteract_root(
    data_root: str | None,
    dataset_repo: str = DEFAULT_OMNIINTERACT_REPO,
) -> Path:
    """Resolve an extracted local tree or download the official archive."""

    if data_root:
        local = Path(data_root).expanduser().resolve()
        if local.is_file():
            return _extract_archive(local, local.parent / f".{local.stem}.vllm_omni_extracted")
        if not local.is_dir():
            raise FileNotFoundError(f"--dataset-path does not exist: {local}")
        try:
            return _data_dir(local)
        except FileNotFoundError:
            for name in ("data.tar.gz", "data.tar"):
                local_archive = local / name
                if local_archive.is_file():
                    return _extract_archive(local_archive, local / ".vllm_omni_extracted")
            raise

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache"))
    cache_key = dataset_repo.replace("/", "__")
    target = cache_root / "vllm_omni" / "omniinteract" / cache_key
    with hub_prefetch._repo_prefetch_lock(
        _archive_lock_key(target),
        required=True,
        lock_dir=target.parent,
    ):
        archive_root = target.parent / "archives" / cache_key
        archive_root.mkdir(parents=True, exist_ok=True)
        filesystem = hf_fs()
        errors: list[str] = []
        downloaded_archive: Path | None = None
        for name in ("data.tar.gz", "data.tar"):
            try:
                downloaded_archive = archive_root / name
                if not downloaded_archive.is_file():
                    with tempfile.TemporaryDirectory(dir=archive_root) as temporary_dir:
                        candidate = Path(temporary_dir) / name
                        filesystem.download(f"datasets/{dataset_repo}/{name}", str(candidate))
                        if not candidate.is_file():
                            raise FileNotFoundError(f"Hugging Face did not download {name}")
                        candidate.replace(downloaded_archive)
                break
            except (OSError, RuntimeError, ValueError) as exc:  # noqa: PERF203 - both archive names are valid
                errors.append(f"{name}: {exc}")
                downloaded_archive = None
        if downloaded_archive is None:
            detail = "; ".join(errors)
            raise FileNotFoundError(f"Could not download OmniInteract from {dataset_repo!r}: {detail}")
        return _extract_archive_unlocked(downloaded_archive, target)


def _mapping_cases(root: Path, subset: str) -> list[OmniInteractCase]:
    mapping_path = root / "video_json_map.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing OmniInteract mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text())
    entries = mapping.get("entries") if isinstance(mapping, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Invalid OmniInteract mapping: {mapping_path}")
    cases: list[OmniInteractCase] = []
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid OmniInteract mapping row in {mapping_path}")
        video_rel = str(row.get("video") or "")
        annotation_rel = str(row.get("annotation") or "")
        video = _confined_path(root, video_rel, field="video")
        annotation = _confined_path(root, annotation_rel, field="annotation")
        if not video.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"OmniInteract mapping references missing files: {video_rel!r}, {annotation_rel!r}")
        cases.append(
            OmniInteractCase(
                subset=subset,
                video_rel=video_rel,
                video_path=video,
                annotation_path=annotation,
                scene_type=str(row.get("scene_type") or "multi_turn").lower(),
            )
        )
    return cases


def _one_to_many_cases(root: Path) -> list[OmniInteractCase]:
    videos = root / "videos_bench"
    annotations = root / "annotations"
    if not videos.is_dir() or not annotations.is_dir():
        raise FileNotFoundError(f"Invalid OmniInteract 1qna layout under {root}")
    cases: list[OmniInteractCase] = []
    for video in sorted(videos.rglob("*.mp4")):
        resolved_video = video.resolve()
        if not resolved_video.is_relative_to(videos.resolve()):
            raise ValueError(f"Unsafe OmniInteract video path: {video}")
        relative = video.relative_to(videos)
        annotation = (annotations / relative).with_suffix(".json").resolve()
        if not annotation.is_relative_to(annotations.resolve()):
            raise ValueError(f"Unsafe OmniInteract annotation path: {relative}")
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing OmniInteract annotation for {video}")
        cases.append(
            OmniInteractCase(
                subset="1qna",
                video_rel=str(video.relative_to(root)),
                video_path=resolved_video,
                annotation_path=annotation,
                scene_type="1qna",
            )
        )
    return cases


def discover_omniinteract_cases(
    root: Path,
    subsets: Sequence[str],
    *,
    num_prompts: int,
    seed: int = 0,
    disable_shuffle: bool = False,
) -> list[OmniInteractCase]:
    """Discover and deterministically select benchmark cases."""

    invalid = set(subsets) - set(OMNIINTERACT_SUBSETS)
    if invalid:
        raise ValueError(f"Unsupported OmniInteract subsets: {sorted(invalid)}")
    if not subsets:
        raise ValueError("At least one OmniInteract subset is required")
    if len(set(subsets)) != len(subsets):
        raise ValueError("OmniInteract subsets must not contain duplicates")
    if num_prompts < 0:
        raise ValueError("num_prompts must be non-negative")

    data = _data_dir(root.resolve())
    cases: list[OmniInteractCase] = []
    for subset in subsets:
        subset_root = data / subset
        subset_cases = _one_to_many_cases(subset_root) if subset == "1qna" else _mapping_cases(subset_root, subset)
        if not subset_cases:
            raise ValueError(f"No OmniInteract sessions found for requested subset {subset!r}")
        cases.extend(subset_cases)
    if not cases:
        raise ValueError(f"No OmniInteract sessions found under {data}")
    paths = [case.video_path for case in cases]
    if len(set(paths)) != len(paths):
        raise ValueError("OmniInteract dataset contains duplicate video paths")
    if not disable_shuffle:
        random.Random(seed).shuffle(cases)
    if num_prompts:
        if num_prompts > len(cases):
            raise ValueError(f"Requested {num_prompts} prompts but only {len(cases)} are available")
        cases = cases[:num_prompts]
    return cases


def case_manifest(case: OmniInteractCase, output_dir: Path) -> dict[str, Any]:
    """Build one portable manifest row for later official scoring."""

    return {
        "sample_id": f"{case.subset}__{output_dir.name}",
        "video": str(case.video_path),
        "gt_json": str(case.annotation_path.resolve()),
        "model_json": str((output_dir / "wav_transcript.json").resolve()),
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
    }
