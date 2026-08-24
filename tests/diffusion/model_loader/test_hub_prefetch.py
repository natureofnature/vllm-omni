# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import errno
import fcntl
import json
import os
import socket

import huggingface_hub
import pytest

from vllm_omni.diffusion.model_loader import hub_prefetch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_required_cache_lock_does_not_continue_after_fallback_timeout(tmp_path, monkeypatch):
    entered = False

    def unsupported_flock(*args):
        raise OSError(errno.ENOLCK, "flock unavailable")

    monkeypatch.setattr(fcntl, "flock", unsupported_flock)
    monkeypatch.setattr(hub_prefetch, "_dotfile_lock_acquire", lambda *args, **kwargs: False)

    with pytest.raises(TimeoutError, match="required cache lock"):
        with hub_prefetch._repo_prefetch_lock("dataset", required=True, lock_dir=tmp_path):
            entered = True

    assert entered is False


def test_required_cache_lock_recovers_expired_fallback_lease(tmp_path, monkeypatch):
    model = "dataset"
    lock_path = tmp_path / (hub_prefetch._safe_repo_filename(model) + ".dir")
    lock_path.mkdir()
    (lock_path / hub_prefetch._DOTFILE_LOCK_OWNER).write_text(
        json.dumps({"hostname": "abandoned-host", "pid": 12345, "token": "abandoned"})
    )
    expired = hub_prefetch.time.time() - 301
    os.utime(lock_path, (expired, expired))

    def unsupported_flock(*args):
        raise OSError(errno.ENOLCK, "flock unavailable")

    monkeypatch.setattr(fcntl, "flock", unsupported_flock)

    with hub_prefetch._repo_prefetch_lock(model, required=True, lock_dir=tmp_path):
        owner = json.loads((lock_path / hub_prefetch._DOTFILE_LOCK_OWNER).read_text())
        assert owner["hostname"] == socket.gethostname()
        assert owner["pid"] == os.getpid()
        assert owner["token"]

    assert not lock_path.exists()


def test_required_cache_lock_does_not_recover_live_fallback_lease(tmp_path):
    model = "dataset"
    lock_path = tmp_path / (hub_prefetch._safe_repo_filename(model) + ".dir")
    lock_path.mkdir()
    (lock_path / hub_prefetch._DOTFILE_LOCK_OWNER).write_text(
        json.dumps({"hostname": socket.gethostname(), "pid": os.getpid()})
    )

    assert not hub_prefetch._remove_stale_dotfile_lock(str(lock_path), stale_after_s=60)
    assert lock_path.is_dir()


def test_prefetch_subfolders_propagates_revision(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub_prefetch, "_repo_prefetch_lock", lambda _model: contextlib.nullcontext())

    hub_prefetch.prefetch_subfolders(
        "org/model",
        ["transformer"],
        revision="pinned-revision",
    )

    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "pinned-revision",
            "allow_patterns": ["transformer/*", "transformer/**", "*.json", "*.txt"],
        }
    ]


def test_from_pretrained_retry_preserves_revision(monkeypatch):
    factory_calls = []
    prefetch_calls = []

    def fake_factory(model, **kwargs):
        factory_calls.append((model, kwargs))
        if len(factory_calls) == 1:
            raise OSError("partial cache")
        return "loaded"

    def fake_prefetch(model, subfolders, **kwargs):
        prefetch_calls.append((model, tuple(subfolders), kwargs))

    monkeypatch.setattr(hub_prefetch, "prefetch_subfolders", fake_prefetch)
    monkeypatch.setattr(hub_prefetch.time, "sleep", lambda _seconds: None)

    result = hub_prefetch.from_pretrained_with_prefetch(
        fake_factory,
        "org/model",
        subfolder="transformer",
        prefetch_list=["transformer", "vae"],
        revision="pinned-revision",
        max_attempts=2,
    )

    assert result == "loaded"
    assert len(factory_calls) == 2
    assert all(call[1]["revision"] == "pinned-revision" for call in factory_calls)
    assert prefetch_calls == [
        (
            "org/model",
            ("transformer", "vae"),
            {"local_files_only": False, "revision": "pinned-revision"},
        )
    ]
