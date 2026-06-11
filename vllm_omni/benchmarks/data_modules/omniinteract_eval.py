"""OmniInteract benchmark metrics (IA-QTF1 / IDS / NCCS) for vLLM bench."""

from __future__ import annotations

import math
import re
import string
from typing import Any

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import OmniInteractSampleRequest

_CJK_PUNCT = "，。！？；：（）【】《》“”‘’、"


def _normalize_text(text: str | None) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    table = str.maketrans("", "", string.punctuation + _CJK_PUNCT)
    t = t.translate(table)
    t = re.sub(r"\s+", "", t)
    return t


def _safe_ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _safe_metric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _metric_row(tp: float, fp: float, fn: float, num_slots: int) -> dict[str, Any]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "num_slots": num_slots,
        "Global_TP": tp,
        "Global_FP": fp,
        "Global_FN": fn,
        "Precision": precision,
        "Recall": recall,
        "IA_QTF1": _f1(precision, recall),
    }


def _metric_sub(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    tp = float(a.get("Global_TP", 0.0)) - float(b.get("Global_TP", 0.0))
    fp = float(a.get("Global_FP", 0.0)) - float(b.get("Global_FP", 0.0))
    fn = float(a.get("Global_FN", 0.0)) - float(b.get("Global_FN", 0.0))
    slots = int(a.get("num_slots", 0) or 0) - int(b.get("num_slots", 0) or 0)
    return _metric_row(tp=max(0.0, tp), fp=max(0.0, fp), fn=max(0.0, fn), num_slots=max(0, slots))


def compute_omniinteract_metrics(
    input_requests: list[Any],
    outputs: list[RequestFuncOutput],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, OmniInteractSampleRequest) for r in input_requests):
        return None

    exact = 0
    soft = 0
    evaluated = 0
    failed = 0
    per_subset: dict[str, dict[str, int]] = {}
    per_qtype: dict[str, dict[str, int]] = {}
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    by_scene: dict[str, dict[str, float]] = {}
    by_qtype_metric: dict[str, dict[str, float]] = {}
    nested_by_role: dict[str, dict[str, float]] = {}
    nested_pairs: dict[tuple[str, int], dict[str, float]] = {}
    interrupted_total = 0
    interrupted_no_output = 0
    interrupted_output_quality_sum = 0.0
    interrupted_output_count = 0
    interrupted_spill_timed_count = 0
    interrupted_spill_positive_count = 0
    interrupted_spill_seconds = 0.0
    items: list[dict[str, Any]] = []

    def _accumulate(store: dict[str, dict[str, float]], key: str, tp: float, fp: float, fn: float) -> None:
        row = store.setdefault(key, {"num_slots": 0.0, "Global_TP": 0.0, "Global_FP": 0.0, "Global_FN": 0.0})
        row["num_slots"] += 1.0
        row["Global_TP"] += tp
        row["Global_FP"] += fp
        row["Global_FN"] += fn

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, OmniInteractSampleRequest)
        subset = (req.omniinteract_subset or "unknown").strip() or "unknown"
        qtype = (req.omniinteract_question_type or "unknown").strip() or "unknown"
        scene = (req.omniinteract_scene_type or "").strip().lower() or ("1qna" if subset == "1qna" else "multi_turn")
        scene = "1qna" if scene in {"1qna", "1qna_long"} else scene
        per_subset.setdefault(subset, {"exact": 0, "soft": 0, "total": 0})
        per_qtype.setdefault(qtype, {"exact": 0, "soft": 0, "total": 0})

        if not out.success:
            failed += 1
            if req.omniinteract_is_interrupted:
                interrupted_total += 1
                interrupted_no_output += 1
            elif qtype:
                global_fn += 1.0
                _accumulate(by_scene, scene, 0.0, 0.0, 1.0)
                _accumulate(by_qtype_metric, qtype, 0.0, 0.0, 1.0)
            if include_per_item:
                items.append(
                    {
                        "request_id": req.request_id,
                        "subset": subset,
                        "question_type": qtype,
                        "video": req.omniinteract_video,
                        "scene_type": scene,
                        "nested_group_id": req.omniinteract_nested_group_id,
                        "nested_role": req.omniinteract_nested_role,
                        "error": (out.error or "")[:500],
                        "correct_exact": False,
                        "correct_soft": False,
                        "quality_score": 0.0,
                        "has_output": False,
                    }
                )
            continue

        pred_raw = out.generated_text or ""
        gold_raw = req.omniinteract_gold_answer or ""
        pred = _normalize_text(pred_raw)
        gold = _normalize_text(gold_raw)
        if not gold:
            continue

        evaluated += 1
        per_subset[subset]["total"] += 1
        per_qtype[qtype]["total"] += 1
        is_exact = pred == gold
        is_soft = bool(pred and gold and (pred in gold or gold in pred))
        quality_score = 1.0 if is_soft else 0.0
        has_output = bool(pred_raw.strip())
        if is_exact:
            exact += 1
            per_subset[subset]["exact"] += 1
            per_qtype[qtype]["exact"] += 1
        if is_soft:
            soft += 1
            per_subset[subset]["soft"] += 1
            per_qtype[qtype]["soft"] += 1

        if req.omniinteract_is_interrupted:
            interrupted_total += 1
            if not has_output:
                interrupted_no_output += 1
            else:
                interrupted_output_count += 1
                interrupted_output_quality_sum += quality_score
                spill_seconds = _safe_metric(getattr(out, "omniinteract_spill_seconds", None))
                if spill_seconds is None:
                    spill_seconds = _safe_metric(getattr(out, "spill_seconds", None))
                if spill_seconds is not None:
                    interrupted_spill_timed_count += 1
                    interrupted_spill_seconds += max(0.0, spill_seconds)
                    if spill_seconds > 0:
                        interrupted_spill_positive_count += 1
        else:
            fp = 1.0 if quality_score <= 0 else 0.0
            fn = 1.0 if quality_score <= 0 else 0.0
            global_tp += quality_score
            global_fp += fp
            global_fn += fn
            _accumulate(by_scene, scene, quality_score, fp, fn)
            _accumulate(by_qtype_metric, qtype, quality_score, fp, fn)
            role = (req.omniinteract_nested_role or "").strip().lower()
            if scene == "nested" and role in {"outer", "inner"}:
                _accumulate(nested_by_role, role, quality_score, fp, fn)
                if req.omniinteract_nested_group_id is not None:
                    pair_key = (req.omniinteract_video or "", int(req.omniinteract_nested_group_id))
                    nested_pairs.setdefault(pair_key, {})[role] = quality_score

        if include_per_item:
            items.append(
                {
                    "request_id": req.request_id,
                    "subset": subset,
                    "scene_type": scene,
                    "question_type": qtype,
                    "video": req.omniinteract_video,
                    "nested_group_id": req.omniinteract_nested_group_id,
                    "nested_role": req.omniinteract_nested_role,
                    "question_time": req.omniinteract_question_time,
                    "answer_time": req.omniinteract_answer_time,
                    "gold": gold_raw,
                    "predicted": pred_raw,
                    "gold_normalized": gold,
                    "predicted_normalized": pred,
                    "correct_exact": is_exact,
                    "correct_soft": is_soft,
                    "quality_score": quality_score,
                    "has_output": has_output,
                }
            )

    by_scene_metric = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in by_scene.items()
    }
    by_qtype_metric_out = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in by_qtype_metric.items()
    }
    nested_by_role_out = {
        name: _metric_row(v["Global_TP"], v["Global_FP"], v["Global_FN"], int(v["num_slots"]))
        for name, v in nested_by_role.items()
    }

    realtime_exclusive = _metric_sub(by_qtype_metric_out.get("realtime", {}), nested_by_role_out.get("inner", {}))
    proactive_exclusive = _metric_sub(by_qtype_metric_out.get("proactive", {}), nested_by_role_out.get("outer", {}))
    nested_metric = by_scene_metric.get("nested", _metric_row(0.0, 0.0, 0.0, 0))
    one_qna_metric = by_scene_metric.get("1qna", _metric_row(0.0, 0.0, 0.0, 0))
    one_q1a_tp = (
        float(realtime_exclusive.get("Global_TP", 0.0))
        + float(proactive_exclusive.get("Global_TP", 0.0))
        + float(nested_metric.get("Global_TP", 0.0))
    )
    one_q1a_fp = (
        float(realtime_exclusive.get("Global_FP", 0.0))
        + float(proactive_exclusive.get("Global_FP", 0.0))
        + float(nested_metric.get("Global_FP", 0.0))
    )
    one_q1a_fn = (
        float(realtime_exclusive.get("Global_FN", 0.0))
        + float(proactive_exclusive.get("Global_FN", 0.0))
        + float(nested_metric.get("Global_FN", 0.0))
    )
    one_q1a_slots = (
        int(realtime_exclusive.get("num_slots", 0) or 0)
        + int(proactive_exclusive.get("num_slots", 0) or 0)
        + int(nested_metric.get("num_slots", 0) or 0)
    )
    all_global = _metric_row(global_tp, global_fp, global_fn, int(one_q1a_slots + one_qna_metric.get("num_slots", 0)))

    num_pairs = 0
    success_pairs = 0
    missing_outer = 0
    ira_sum = 0.0
    for pair in nested_pairs.values():
        num_pairs += 1
        q1 = float(pair.get("outer", 0.0))
        q2 = float(pair.get("inner", 0.0))
        if q1 <= 0:
            missing_outer += 1
        if q1 > 0 and q2 > 0:
            success_pairs += 1
            ira_sum += math.sqrt(q1 * q2)
    nccs = _safe_div(ira_sum, num_pairs)

    nor = _safe_div(interrupted_no_output, interrupted_total)
    paq = _safe_div(interrupted_output_quality_sum, interrupted_output_count)
    csm_sr = (
        _safe_div(interrupted_spill_positive_count, interrupted_spill_timed_count)
        if interrupted_spill_timed_count
        else None
    )
    csm_as = (
        _safe_div(interrupted_spill_seconds, interrupted_spill_timed_count) if interrupted_spill_timed_count else None
    )

    out: dict[str, Any] = {
        "omniinteract_evaluated": evaluated,
        "omniinteract_request_failed": failed,
        "omniinteract_exact_match": _safe_ratio(exact, evaluated),
        "omniinteract_soft_match": _safe_ratio(soft, evaluated),
        "omniinteract_exact_count": exact,
        "omniinteract_soft_count": soft,
        "omniinteract_per_subset": per_subset,
        "omniinteract_per_question_type": per_qtype,
        "omniinteract_paper_metrics": {
            "exp_f1": {
                "realtime": realtime_exclusive,
                "proactive": proactive_exclusive,
                "nested": nested_metric,
                "one_q1a_global": _metric_row(one_q1a_tp, one_q1a_fp, one_q1a_fn, one_q1a_slots),
                "one_qna": one_qna_metric,
                "all_global": all_global,
            },
            "exp_interruption": {
                "NOR": nor,
                "PAQ": paq,
                "CSM_SR": csm_sr,
                "CSM_AS_seconds": csm_as,
                "interrupted_slot_count": interrupted_total,
                "interrupted_with_output_count": interrupted_output_count,
                "interrupted_with_spill_timing_count": interrupted_spill_timed_count,
            },
            "exp_nested": {
                "NCCS": nccs,
                "inner_IA_QTF1": float(nested_by_role_out.get("inner", {}).get("IA_QTF1", 0.0)),
                "outer_IA_QTF1": float(nested_by_role_out.get("outer", {}).get("IA_QTF1", 0.0)),
                "missed_outer": missing_outer,
                "num_pairs": num_pairs,
                "success_pairs": success_pairs,
            },
        },
        "omniinteract_ia_qtf1": float(all_global["IA_QTF1"]),
        "omniinteract_ids": {
            "NOR": nor,
            "PAQ": paq,
            "CSM_SR": csm_sr,
            "CSM_AS_seconds": csm_as,
        },
        "omniinteract_nccs": nccs,
        "omniinteract_metric_note": (
            "IA-QTF1/IDS/NCCS are estimated from per-request benchmark outputs. "
            "IDS CSM_SR/CSM_AS require continuous-turn spill timing; they are N/A "
            "unless outputs provide omniinteract_spill_seconds/spill_seconds."
        ),
    }

    out["omniinteract_per_subset_exact"] = {
        name: _safe_ratio(vals["exact"], vals["total"]) for name, vals in per_subset.items()
    }
    out["omniinteract_per_question_type_exact"] = {
        name: _safe_ratio(vals["exact"], vals["total"]) for name, vals in per_qtype.items()
    }
    if include_per_item:
        out["omniinteract_eval_items"] = items
    return out


def print_omniinteract_summary(metrics: dict[str, Any]) -> None:
    if (
        int(metrics.get("omniinteract_evaluated", 0) or 0) == 0
        and int(metrics.get("omniinteract_request_failed", 0) or 0) == 0
    ):
        return
    print("{s:{c}^{n}}".format(s=" OmniInteract QA metrics ", n=50, c="="))
    print("{:<40} {:<10}".format("Evaluated:", metrics.get("omniinteract_evaluated", 0)))
    if metrics.get("omniinteract_ia_qtf1") is not None:
        print("{:<40} {:<10.4f}".format("IA-QTF1 (estimated):", float(metrics.get("omniinteract_ia_qtf1"))))
    ids = metrics.get("omniinteract_ids") or {}
    if ids:
        print("{:<40} {:<10.4f}".format("IDS.NOR:", float(ids.get("NOR", 0.0))))
        print("{:<40} {:<10.4f}".format("IDS.PAQ:", float(ids.get("PAQ", 0.0))))
        csm_sr = ids.get("CSM_SR")
        csm_as = ids.get("CSM_AS_seconds")
        if csm_sr is None:
            print("{:<40} {:<10}".format("IDS.CSM-SR:", "N/A"))
        else:
            print("{:<40} {:<10.4f}".format("IDS.CSM-SR:", float(csm_sr)))
        if csm_as is None:
            print("{:<40} {:<10}".format("IDS.CSM-AS(s):", "N/A"))
        else:
            print("{:<40} {:<10.4f}".format("IDS.CSM-AS(s):", float(csm_as)))
    if metrics.get("omniinteract_nccs") is not None:
        print("{:<40} {:<10.4f}".format("NCCS (estimated):", float(metrics.get("omniinteract_nccs"))))
    print("=" * 50)
