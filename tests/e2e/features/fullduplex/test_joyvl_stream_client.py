# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from examples.online_serving.joyvl_interaction.cli import stream_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        assert self.status_code < 400, self._payload


def test_stream_client_restarts_same_session_in_a_fresh_epoch(monkeypatch):
    state = {"epoch": 0}
    chat_epochs: list[int] = []

    def post(_url: str, *, json: dict[str, Any], timeout: int) -> _Response:
        assert timeout > 0
        if _url.endswith("/reset"):
            expected = json["session_epoch"]
            current = state["epoch"]
            if expected + 1 == current:
                return _Response(200, {"epoch": current, "advanced": False})
            if expected != current:
                return _Response(409, {"current_epoch": current})
            state["epoch"] = current + 1
            return _Response(200, {"epoch": state["epoch"], "advanced": True})

        assert json["session_epoch"] == state["epoch"]
        assert json["operation_id"] == "0"
        chat_epochs.append(json["session_epoch"])
        return _Response(
            200,
            {
                "interaction": {"frame_index": 1, "action": "response"},
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    monkeypatch.setattr(stream_client.requests, "post", post)
    monkeypatch.setattr(
        stream_client,
        "iter_frames",
        lambda _video_path, _fps: iter([(0.0, b"jpeg")]),
    )

    first = list(stream_client.stream("video.mp4", "http://server", "cli"))
    second = list(stream_client.stream("video.mp4", "http://server", "cli"))

    assert len(first) == 1
    assert len(second) == 1
    assert chat_epochs == [1, 2]


def test_stream_client_falls_back_to_in_process_reset_route(monkeypatch):
    urls: list[str] = []

    def post(url: str, *, json: dict[str, Any], timeout: int) -> _Response:
        urls.append(url)
        assert timeout > 0
        if url.endswith("/reset") and not url.endswith("/v1/session/reset"):
            return _Response(404, {"detail": "Not Found"})
        if url.endswith("/v1/session/reset"):
            assert json == {"session_id": "cli", "session_epoch": 0}
            return _Response(200, {"epoch": 1})

        assert json["session_epoch"] == 1
        return _Response(
            200,
            {
                "interaction": {"frame_index": 1, "action": "response"},
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    monkeypatch.setattr(stream_client.requests, "post", post)
    monkeypatch.setattr(
        stream_client,
        "iter_frames",
        lambda _video_path, _fps: iter([(0.0, b"jpeg")]),
    )

    ticks = list(stream_client.stream("video.mp4", "http://server", "cli"))

    assert len(ticks) == 1
    assert urls == [
        "http://server/reset",
        "http://server/v1/session/reset",
        "http://server/v1/chat/completions",
    ]
