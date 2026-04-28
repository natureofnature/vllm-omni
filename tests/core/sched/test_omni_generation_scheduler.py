from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_generation_scheduler import _extend_all_token_ids_if_available

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_extend_all_token_ids_if_available_extends_private_token_list():
    request = SimpleNamespace(_all_token_ids=[7, 8])

    _extend_all_token_ids_if_available(request, 3)

    assert request._all_token_ids == [7, 8, 0, 0, 0]


def test_extend_all_token_ids_if_available_skips_missing_or_non_list_storage():
    missing_attr = SimpleNamespace()
    tuple_backed = SimpleNamespace(_all_token_ids=(7, 8))

    _extend_all_token_ids_if_available(missing_attr, 2)
    _extend_all_token_ids_if_available(tuple_backed, 2)

    assert not hasattr(missing_attr, "_all_token_ids")
    assert tuple_backed._all_token_ids == (7, 8)
