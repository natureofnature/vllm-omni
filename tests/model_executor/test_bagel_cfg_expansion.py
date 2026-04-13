from vllm_omni.model_executor.stage_input_processors.bagel import expand_cfg_prompts, expand_cfg_prompts_think


class DummyParams:
    extra_args = {"negative_prompt": "bad"}


def _roles(expanded):
    return [ep.role for ep in expanded]


def test_expand_cfg_prompts_default_img2img_has_both_companions():
    prompt = {"prompt": "sys<|fim_middle|>user", "modalities": ["img2img"], "multi_modal_data": {"img2img": ["x"]}}
    assert _roles(expand_cfg_prompts(prompt, DummyParams())) == ["cfg_text", "cfg_img"]


def test_expand_cfg_prompts_local_cfg_text_on_dit_keeps_only_cfg_img():
    prompt = {
        "prompt": "sys<|fim_middle|>user",
        "modalities": ["img2img"],
        "multi_modal_data": {"img2img": ["x"]},
        "bagel_local_cfg_text_on_dit": True,
    }
    assert _roles(expand_cfg_prompts(prompt, DummyParams())) == ["cfg_img"]
    assert _roles(expand_cfg_prompts_think(prompt, DummyParams())) == ["cfg_img"]


def test_expand_cfg_prompts_local_cfg_text_on_dit_keeps_cfg_img_text_only():
    prompt = {
        "prompt": "sys<|fim_middle|>user",
        "modalities": ["img2img"],
        "multi_modal_data": {"img2img": ["x"]},
        "bagel_local_cfg_text_on_dit": True,
    }
    [cfg_img] = expand_cfg_prompts(prompt, DummyParams())
    assert cfg_img.role == "cfg_img"
    assert "multi_modal_data" not in cfg_img.prompt
