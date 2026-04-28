from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_route_finished_only_handles_cfg_companion():
    orchestrator = object.__new__(Orchestrator)
    orchestrator.request_states = {"req-comp": OrchestratorRequestState(request_id="req-comp", final_stage_id=1)}
    orchestrator._companion_ids = {"req-comp"}
    orchestrator.output_async_queue = SimpleNamespace(put=AsyncMock())
    orchestrator._handle_cfg_companion_ready = AsyncMock()

    await Orchestrator._route_finished_only(orchestrator, 0, {"req-comp"})

    orchestrator._handle_cfg_companion_ready.assert_awaited_once_with("req-comp")
    orchestrator.output_async_queue.put.assert_not_awaited()
    assert "req-comp" not in orchestrator.request_states


@pytest.mark.asyncio
async def test_route_finished_only_cleans_up_final_request():
    orchestrator = object.__new__(Orchestrator)
    orchestrator.request_states = {"req-final": OrchestratorRequestState(request_id="req-final", final_stage_id=1)}
    orchestrator._companion_ids = set()
    orchestrator.output_async_queue = SimpleNamespace(put=AsyncMock())
    orchestrator._cleanup_companion_state = MagicMock()

    await Orchestrator._route_finished_only(orchestrator, 1, {"req-final"})

    orchestrator.output_async_queue.put.assert_awaited_once_with(
        {
            "type": "finished_only",
            "request_id": "req-final",
            "stage_id": 1,
            "finished": True,
        }
    )
    orchestrator._cleanup_companion_state.assert_called_once_with("req-final")
    assert "req-final" not in orchestrator.request_states
