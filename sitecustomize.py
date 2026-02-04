"""Process-wide customizations for vLLM-Omni workers.

This module is auto-imported by Python's site machinery when present on
PYTHONPATH. It ensures BagelProcessor filters unsupported image kwargs
in spawned worker processes (spawn uses a fresh interpreter).
"""


def _patch_bagel_processor() -> None:
    """Filter invalid kwargs for Bagel image processor (e.g., truncation)."""
    try:
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.transformers_utils.processors.bagel import BagelProcessor

        def _patched_call(self, text=None, images=None, **kwargs):
            if images is not None:
                image_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in (
                        "use_fast",
                        "truncation",
                        "padding",
                        "max_length",
                        "add_special_tokens",
                    )
                }
                if "return_tensors" not in image_kwargs:
                    image_kwargs["return_tensors"] = "pt"
                pixel_values = self.image_processor(images, **image_kwargs)
            else:
                pixel_values = None
            text_inputs = self.tokenizer(text, **kwargs) if text is not None else None
            if pixel_values is not None and text_inputs is not None:
                combined = dict(text_inputs)
                combined["pixel_values"] = pixel_values["pixel_values"]
                return BatchFeature(combined)
            if pixel_values is not None:
                return pixel_values
            if text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            return BatchFeature({})

        BagelProcessor.__call__ = _patched_call
    except ImportError:
        # If BagelProcessor is unavailable, do nothing.
        pass


_patch_bagel_processor()
