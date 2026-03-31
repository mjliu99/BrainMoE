# llm/llm_utils.py
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from llm.gemini_llm_processor import extract_single_entry, text_to_vector


def _read_system_instructions(system_path: Optional[str] = None) -> str:
    if system_path is None:
        system_path = str(Path(__file__).with_name("system_instructions.txt"))
    return Path(system_path).read_text(encoding="utf-8")


def _infer_stage(
    isdt_json: Optional[Dict[str, Any]],
    subject_json: Optional[Dict[str, Any]],
) -> str:
    """
    Infer stage from payload content.
    Priority:
      1) explicit _stage
      2) graph_abnormal_* presence -> B_final
      3) fallback -> A_router
    """
    if isinstance(subject_json, dict):
        st = subject_json.get("_stage", None)
        if isinstance(st, str) and st.strip():
            return st.strip()

    if isinstance(isdt_json, dict):
        st = isdt_json.get("_stage", None)
        if isinstance(st, str) and st.strip():
            return st.strip()

    if isinstance(isdt_json, dict):
        if "graph_abnormal_report" in isdt_json or "graph_abnormal_prompt" in isdt_json:
            return "B_final"

    if isinstance(subject_json, dict):
        if "graph_abnormal_report" in subject_json or "graph_abnormal_prompt" in subject_json:
            return "B_final"

    return "A_router"


def _attach_stage(
    isdt_json: Dict[str, Any],
    subject_json: Dict[str, Any],
    force_stage: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return deep-copied payloads with a consistent _stage field attached.
    """
    a = copy.deepcopy(isdt_json) if isinstance(isdt_json, dict) else {}
    b = copy.deepcopy(subject_json) if isinstance(subject_json, dict) else {}

    stage = force_stage.strip() if isinstance(force_stage, str) and force_stage.strip() else _infer_stage(a, b)

    a["_stage"] = stage
    b["_stage"] = stage
    return a, b


def _normalize_llm_result(
    out: Optional[Dict[str, Any]],
    stage: str,
) -> Optional[Dict[str, Any]]:
    """
    Make downstream storage stable.
    Keep the core 3 fields used by the rest of the pipeline.
    """
    if out is None:
        return None

    if not isinstance(out, dict):
        return {
            "stage": stage,
            "initial_decision": 0,
            "reason": "",
            "graph_learning_rank": "GCN, 2-hop GCN, Graph Transformer, ChebNet",
        }

    rank = str(out.get("graph_learning_rank", "") or "").strip()
    if not rank:
        rank = "GCN, 2-hop GCN, Graph Transformer, ChebNet"

    try:
        initial_decision = int(out.get("initial_decision", 0) or 0)
    except Exception:
        initial_decision = 0

    reason = str(out.get("reason", "") or "").strip()

    normalized = {
        "stage": str(out.get("stage", stage) or stage),
        "initial_decision": initial_decision,
        "reason": reason,
        "graph_learning_rank": rank,
    }

    # optional convenience field for stage1 / router
    try:
        normalized["top_expert"] = rank.split(",")[0].strip()
    except Exception:
        normalized["top_expert"] = ""

    return normalized


def process_rows(
    isdt_json_list: List[Dict[str, Any]],
    subject_json_list: List[Dict[str, Any]],
    rows_to_process: Optional[int] = None,
    max_workers: Optional[int] = None,
    system_path: Optional[str] = None,
    retry: int = 2,
    retry_sleep: float = 1.5,
    force_stage: Optional[str] = None,
) -> List[Optional[Dict[str, Any]]]:
    """
    Calls LLM for each (isdt_json, subject_json).
    Returns list of dict outputs (or None if failed).

    New behavior:
      - auto-attaches _stage to both payload sides
      - normalizes returned fields for downstream stability
    """
    if rows_to_process is None:
        rows_to_process = len(subject_json_list)
    else:
        rows_to_process = min(rows_to_process, len(subject_json_list))

    rows_to_process = min(rows_to_process, len(isdt_json_list))
    system_instructions = _read_system_instructions(system_path)

    raw_pairs = list(zip(isdt_json_list[:rows_to_process], subject_json_list[:rows_to_process]))
    pairs = [_attach_stage(a, b, force_stage=force_stage) for a, b in raw_pairs]

    if max_workers is None:
        max_workers = min(4, len(pairs)) or 1
    max_workers = max(1, int(max_workers))

    def _call_with_retry(isdt_json, subject_json):
        last_err = None
        stage = _infer_stage(isdt_json, subject_json)

        for t in range(retry + 1):
            try:
                out = extract_single_entry(isdt_json, subject_json, system_instructions)
                return _normalize_llm_result(out, stage=stage)
            except Exception as e:
                last_err = e
                if t < retry:
                    time.sleep(retry_sleep * (t + 1))

        print(f"[LLM] failed after retries: {last_err}")
        return None

    results: List[Optional[Dict[str, Any]]] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_call_with_retry, a, b): idx
            for idx, (a, b) in enumerate(pairs)
        }
        for future in tqdm(as_completed(futures), total=len(pairs), desc="LLM rows"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[LLM] unexpected future error: {e}")
                results[idx] = None

    return results


def process_rows_from_texts(
    prompt_text_list: List[str],
    rows_to_process: Optional[int] = None,
    max_workers: Optional[int] = None,
    system_path: Optional[str] = None,
    retry: int = 2,
    retry_sleep: float = 1.5,
    force_stage: str = "B_final",
) -> List[Optional[Dict[str, Any]]]:
    """
    For legacy/simple prompt pipelines.

    We wrap raw text into graph_abnormal_prompt and mark them as B_final by default,
    because this helper is usually used for abnormal-report style prompting.
    """
    if rows_to_process is None:
        rows_to_process = len(prompt_text_list)
    else:
        rows_to_process = min(rows_to_process, len(prompt_text_list))

    isdt_json_list = [
        {"graph_abnormal_prompt": t}
        for t in prompt_text_list[:rows_to_process]
    ]
    subject_json_list = [{} for _ in range(rows_to_process)]

    return process_rows(
        isdt_json_list=isdt_json_list,
        subject_json_list=subject_json_list,
        rows_to_process=rows_to_process,
        max_workers=max_workers,
        system_path=system_path,
        retry=retry,
        retry_sleep=retry_sleep,
        force_stage=force_stage,
    )


def all_text_to_vector(
    json_list: List[Optional[Dict[str, Any]]],
    max_workers: Optional[int] = None,
    retry: int = 2,
    retry_sleep: float = 1.0,
) -> List[Optional[List[float]]]:
    """
    Embed the structured JSON outputs into vectors using text_to_vector().
    Returns list of vectors or None.

    Stage-aware embedding behavior is implemented inside text_to_vector().
    """
    if max_workers is None:
        max_workers = min(4, len(json_list)) or 1
    max_workers = max(1, int(max_workers))

    def _embed_with_retry(j):
        last_err = None
        for t in range(retry + 1):
            try:
                v = text_to_vector(j)
                return v
            except Exception as e:
                last_err = e
                if t < retry:
                    time.sleep(retry_sleep * (t + 1))
        print(f"[EMB] failed after retries: {last_err}")
        return None

    vectors: List[Optional[List[float]]] = [None] * len(json_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_embed_with_retry, j): idx for idx, j in enumerate(json_list)}
        for future in tqdm(as_completed(futures), total=len(json_list), desc="Embeddings"):
            idx = futures[future]
            try:
                vectors[idx] = future.result()
            except Exception as e:
                print(f"[EMB] unexpected future error: {e}")
                vectors[idx] = None

    return vectors