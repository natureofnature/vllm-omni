import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_generate_accepts_request_after_repeated_cancellations():
    async def run_test():
        submitted_request_ids = []
        aborted_request_batches = []

        async def fake_add_request_async(*, request_id, prompt, sampling_params_list, final_stage_id, **kwargs):
            del prompt, sampling_params_list, final_stage_id, kwargs
            submitted_request_ids.append(request_id)

        async def fake_abort_async(request_ids):
            aborted_request_batches.append(list(request_ids))

        async def fake_process_results(request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts):
            del metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts
            if request_id.startswith("cancel-"):
                await asyncio.Future()
                return
            yield SimpleNamespace(
                stage_id=0,
                request_output=SimpleNamespace(outputs=[]),
                finished=True,
            )

        async def collect_outputs(request_id):
            outputs = []
            async for output in AsyncOmni.generate(
                omni,
                prompt={"prompt": "prompt"},
                request_id=request_id,
                sampling_params_list=[SimpleNamespace()],
                output_modalities=["image"],
            ):
                outputs.append(output)
            return outputs

        omni = object.__new__(AsyncOmni)
        omni._pause_cond = asyncio.Condition()
        omni._paused = False
        omni.engine = SimpleNamespace(
            num_stages=1,
            add_request_async=fake_add_request_async,
            abort_async=fake_abort_async,
        )
        omni.log_stats = False
        omni.request_states = {}
        omni._final_output_handler = lambda: None
        omni.resolve_sampling_params_list = lambda params: params
        omni._compute_final_stage_id = lambda output_modalities: 0
        omni._process_orchestrator_results = fake_process_results
        omni._log_summary_and_cleanup = lambda request_id: omni.request_states.pop(request_id, None)

        assert len(await collect_outputs("baseline")) == 1

        for idx in range(3):
            task = asyncio.create_task(collect_outputs(f"cancel-{idx}"))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert len(await collect_outputs("after-cancel")) == 1
        assert submitted_request_ids == [
            "baseline",
            "cancel-0",
            "cancel-1",
            "cancel-2",
            "after-cancel",
        ]
        assert aborted_request_batches == [
            ["cancel-0"],
            ["cancel-1"],
            ["cancel-2"],
        ]

    asyncio.run(run_test())


def test_process_orchestrator_results_finished_only_updates_metrics():
    async def run_test():
        import vllm_omni.entrypoints.async_omni as async_omni_module

        omni = object.__new__(AsyncOmni)
        queue = asyncio.Queue()
        await queue.put(
            {
                "type": "finished_only",
                "request_id": "req-1",
                "stage_id": 1,
                "finished": True,
            }
        )
        omni.request_states = {"req-1": SimpleNamespace(queue=queue)}

        finalize_calls = []

        metrics = SimpleNamespace(
            stage_first_ts=[None, None],
            stage_last_ts=[None, None],
            e2e_done=set(),
            on_finalize_request=lambda stage_id, req_id, start_ts: finalize_calls.append((stage_id, req_id, start_ts))
            or metrics.e2e_done.add(str(req_id)),
        )
        req_start_ts = {"req-1": 12.5}

        old_timeout = async_omni_module._FINAL_OUTPUT_DRAIN_TIMEOUT_S
        async_omni_module._FINAL_OUTPUT_DRAIN_TIMEOUT_S = 0.0
        try:
            outputs = []
            async for output in AsyncOmni._process_orchestrator_results(
                omni,
                "req-1",
                metrics,
                1,
                req_start_ts,
                3.14,
            ):
                outputs.append(output)
        finally:
            async_omni_module._FINAL_OUTPUT_DRAIN_TIMEOUT_S = old_timeout

        assert outputs == []
        assert finalize_calls == [(1, "req-1", 12.5)]
        assert "req-1" in metrics.e2e_done
        assert metrics.stage_first_ts[1] is not None
        assert metrics.stage_last_ts[1] is not None

    asyncio.run(run_test())
