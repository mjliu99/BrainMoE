from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from llm.openrouter_llm_processor import extract_single_entry, text_to_vector


STAGE1_SYSTEM_INSTRUCTIONS = """
You are an expert assistant for identity-aware brain graph learning.

This is Stage A_router.
Your task is to analyze subject metadata together with identity-semantics discrete topology tokens (ISDT),
and recommend which graph-learning inductive biases are most suitable for this subject.

Important:
- This is router-stage reasoning, NOT final diagnosis.
- Do NOT diagnose the subject.
- Focus on structural heterogeneity, topology, modular organization, and long-range vs local interactions.

You MUST strictly follow these output rules:
1. Return ONLY a valid JSON object.
2. The JSON object must contain exactly these four keys:
   - initial_decision
   - reason
   - graph_learning_rank
   - stage
3. initial_decision must be the integer 0.
4. stage must be exactly "A_router".
5. graph_learning_rank must be a comma-separated ranking containing EXACTLY these four experts, each appearing once and only once:
   - GCN
   - MLP
   - Graph Transformer
   - ChebNet
6. Do NOT introduce any other expert names, aliases, abbreviations, or extra items.
   In particular, do NOT output 2-hop GCN or any custom expert name.
7. The first expert in graph_learning_rank is treated as the top expert, so rank them from best to worst.
8. reason must be concise, evidence-grounded, and based only on the provided topology / identity cues.
9. If the input evidence is weak or incomplete, still output a best-effort ranking using only the four allowed experts.

Output example:
{
  "initial_decision": 0,
  "reason": "The subject shows stronger long-range and cross-module interaction patterns, making Graph Transformer more suitable, followed by ChebNet for multi-scale topology.",
  "graph_learning_rank": "Graph Transformer, ChebNet, MLP, GCN",
  "stage": "A_router"
}
""".strip()


STAGE2_SYSTEM_INSTRUCTIONS = """
You are an expert assistant for psychiatric disorder prediction from identity-aware brain graphs.

This is Stage B_final.
Your task is to integrate subject metadata, ISDT signals, graph abnormality evidence,
and the Stage-1 router output to produce final reasoning.

Important:
- This is final-stage reasoning.
- Base the final decision primarily on the provided abnormal graph evidence, abnormal edge / ROI evidence,
  and summary abnormality statistics.
- Do NOT invent additional evidence.
- Do NOT defer the final decision to routing preference alone.

You MUST strictly follow these output rules:
1. Return ONLY a valid JSON object.
2. The JSON object must contain exactly these four keys:
   - initial_decision
   - reason
   - graph_learning_rank
   - stage
3. initial_decision must be either 0 or 1.
4. stage must be exactly "B_final".
5. graph_learning_rank must be a comma-separated ranking containing EXACTLY these four experts, each appearing once and only once:
   - GCN
   - MLP
   - Graph Transformer
   - ChebNet
6. Do NOT introduce any other expert names, aliases, abbreviations, or extra items.
   In particular, do NOT output 2-hop GCN or any custom expert name.
7. The first expert in graph_learning_rank is treated as the top expert, so rank them from best to worst.
8. reason must be concise, evidence-grounded, and based only on the provided abnormal graph evidence.
9. For initial_decision, use the abnormality evidence as the primary basis. Do NOT override the decision merely because of Stage-1 routing preference.
10. If the evidence is borderline or incomplete, still output a best-effort binary decision using only the provided evidence.

Output example:
{
  "initial_decision": 1,
  "reason": "The subject shows strong abnormal tail evidence and multiple highly deviant edges, supporting a disorder-positive prediction.",
  "graph_learning_rank": "Graph Transformer, ChebNet, MLP, GCN",
  "stage": "B_final"
}
""".strip()


def _choose_system_prompt(
    isdt_json: Optional[Dict[str, Any]],
    subject_json: Optional[Dict[str, Any]],
) -> str:
    stage = None
    if isinstance(subject_json, dict):
        stage = subject_json.get("_stage")
    if not stage and isinstance(isdt_json, dict):
        stage = isdt_json.get("_stage")

    if stage == "A_router":
        return STAGE1_SYSTEM_INSTRUCTIONS
    return STAGE2_SYSTEM_INSTRUCTIONS


def process_rows(
    isdt_rows: List[Dict[str, Any]],
    subject_rows: List[Dict[str, Any]],
    rows_to_process: Optional[int] = None,
    max_workers: int = 1,
    model: str = "xiaomi/mimo-v2-omni",
) -> List[Optional[Dict[str, Any]]]:
    if rows_to_process is None:
        rows_to_process = min(len(isdt_rows), len(subject_rows))

    n = min(rows_to_process, len(isdt_rows), len(subject_rows))
    results: List[Optional[Dict[str, Any]]] = [None] * n

    def _run_one(i: int):
        return i, extract_single_entry(
            isdt_json=isdt_rows[i],
            subject_json=subject_rows[i],
            system_instructions=_choose_system_prompt(isdt_rows[i], subject_rows[i]),
        )
    if max_workers <= 1:
        for i in range(n):
            _, out = _run_one(i)
            results[i] = out
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one, i) for i in range(n)]
        for fut in as_completed(futs):
            i, out = fut.result()
            results[i] = out

    return results


def all_text_to_vector(
    rows: List[Any],
    max_workers: int = 2,
) -> List[Optional[List[float]]]:
    n = len(rows)
    results: List[Optional[List[float]]] = [None] * n

    def _run_one(i: int):
        return i, text_to_vector(rows[i])

    if max_workers <= 1:
        for i in range(n):
            _, out = _run_one(i)
            results[i] = out
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one, i) for i in range(n)]
        for fut in as_completed(futs):
            i, out = fut.result()
            results[i] = out

    return results