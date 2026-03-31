from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, List

from openrouter import OpenRouter
from google import genai
from google.genai import types


# ----------------------------
# API keys
# ----------------------------
def _ensure_openrouter_key_env() -> None:
    if os.getenv("OPENROUTER_API_KEY"):
        return
    raise RuntimeError("No API key found. Please set OPENROUTER_API_KEY.")


def _ensure_gemini_api_key_env() -> None:
    if os.getenv("GOOGLE_API_KEY"):
        return
    if os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        return
    raise RuntimeError(
        "No Gemini API key found for embeddings. Please set GOOGLE_API_KEY "
        "(recommended) or GEMINI_API_KEY."
    )


_ensure_openrouter_key_env()
_ensure_gemini_api_key_env()

llm_client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
emb_client = genai.Client()


# ----------------------------
# Model names (env override)
# ----------------------------
def _normalize_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return name
    if name.startswith("models/"):
        return name
    return f"models/{name}"


LLM_MODEL = os.getenv("OPENROUTER_LLM_MODEL", "xiaomi/mimo-v2-pro").strip()
LLM_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "512"))
TEXT_EMBEDDING_MODEL = _normalize_model_name(
    os.getenv("GEMINI_EMB_MODEL", "gemini-embedding-001")
)
EMB_OUTPUT_DIM = int(os.getenv("GEMINI_EMB_DIM", "0"))

DEFAULT_GRAPH_RANK = "GCN, MLP, Graph Transformer, ChebNet"


# ----------------------------
# Retry helpers
# ----------------------------
_RETRY_DELAY_PATTERNS = [
    re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE),
    re.compile(r"'retryDelay':\s*'(\d+)s'"),
    re.compile(r"Retry-After:\s*(\d+)", re.IGNORECASE),
]


def _sleep_from_msg(msg: str, default: float = 10.0, cap: float = 60.0) -> float:
    for pat in _RETRY_DELAY_PATTERNS:
        m = pat.search(msg or "")
        if m:
            try:
                return min(float(m.group(1)), cap)
            except Exception:
                pass
    return min(float(default), cap)


def _is_rate_limit(msg: str) -> bool:
    if not msg:
        return False
    msg_u = msg.upper()
    return (
        ("RESOURCE_EXHAUSTED" in msg_u)
        or ("429" in msg_u)
        or ("RATE" in msg_u and "LIMIT" in msg_u)
    )


def _is_network_like(msg: str) -> bool:
    if not msg:
        return False
    msg_u = msg.upper()
    keys = ["TIMEOUT", "TIMED OUT", "CONNECTION", "UNAVAILABLE", "DEADLINE", "EOF", "RESET"]
    return any(k in msg_u for k in keys)


def _is_token_or_credit_error(msg: str) -> bool:
    if not msg:
        return False
    msg_u = msg.upper()
    return (
        ("PROMPT TOKENS LIMIT EXCEEDED" in msg_u)
        or ("REQUIRES MORE CREDITS" in msg_u)
        or ("CAN ONLY AFFORD" in msg_u)
    )


# ----------------------------
# JSON extraction / repair
# ----------------------------
def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_obj(text: str) -> Optional[str]:
    if not text:
        return None

    text = _strip_code_fences(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    js = _extract_first_json_obj(text)
    if not js:
        return None

    try:
        obj = json.loads(js)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


# ----------------------------
# Hard timeout wrapper
# ----------------------------
def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_s)


# ----------------------------
# Stage detection helpers
# ----------------------------
def _detect_stage(
    isdt_json: Optional[Dict[str, Any]] = None,
    subject_json: Optional[Dict[str, Any]] = None,
    llm_output: Optional[Dict[str, Any]] = None,
) -> str:
    if isinstance(subject_json, dict):
        st = subject_json.get("_stage", None)
        if isinstance(st, str):
            st = st.strip()
            if st == "A_router":
                return "A_router"
            if st:
                return "B_final"

    if isinstance(isdt_json, dict):
        st = isdt_json.get("_stage", None)
        if isinstance(st, str):
            st = st.strip()
            if st == "A_router":
                return "A_router"
            if st:
                return "B_final"

    if isinstance(llm_output, dict):
        reason = str(llm_output.get("reason", "") or "").lower()
        rank = str(llm_output.get("graph_learning_rank", "") or "").lower()

        if any(k in reason for k in ["p99", "p995", "frac230", "cnt230", "abnormal edge", "diagnos"]):
            return "B_final"
        if rank and any(k in reason for k in ["expert", "structural", "top-ranked", "graph transformer", "chebnet", "gcn", "mlp"]):
            return "A_router"

    return "unknown"


# ----------------------------
# Output normalization
# ----------------------------
def _normalize_llm_output(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stricter normalization:
    if both reason and rank are empty, return None instead of fake success.
    """
    if not isinstance(data, dict):
        return None

    rank = str(data.get("graph_learning_rank", "") or "").strip()
    reason = str(data.get("reason", "") or "").strip()

    if not rank and not reason:
        return None

    out: Dict[str, Any] = {}

    try:
        out["initial_decision"] = int(data.get("initial_decision", 0) or 0)
    except Exception:
        val = str(data.get("initial_decision", "0")).strip().lower()
        if val in {"1", "true", "patient", "adhd", "asd", "disorder", "positive"}:
            out["initial_decision"] = 1
        else:
            out["initial_decision"] = 0

    out["reason"] = reason
    out["graph_learning_rank"] = rank if rank else DEFAULT_GRAPH_RANK

    stage = data.get("stage", None)
    if isinstance(stage, str) and stage.strip():
        out["stage"] = stage.strip()

    return out


# ----------------------------
# OpenRouter response helpers
# ----------------------------
def _extract_text_from_openrouter_response(response: Any) -> str:
    """
    Prefer content, but fall back to reasoning.
    """
    if response is None:
        return ""

    try:
        choices = getattr(response, "choices", None)
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()

                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            txt = item.get("text") or item.get("content")
                            if txt:
                                parts.append(str(txt))
                        else:
                            txt = getattr(item, "text", None) or getattr(item, "content", None)
                            if txt:
                                parts.append(str(txt))
                    if parts:
                        return "\n".join(parts).strip()

                reasoning = getattr(msg, "reasoning", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()

        if isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                reasoning = message.get("reasoning", "")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
    except Exception:
        pass

    return str(response)


def _get_finish_reason_from_response(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if choices:
            fr = getattr(choices[0], "finish_reason", None)
            if fr is not None:
                return str(fr)
        if isinstance(response, dict):
            choices = response.get("choices")
            if choices and isinstance(choices, list):
                fr = choices[0].get("finish_reason", None)
                if fr is not None:
                    return str(fr)
    except Exception:
        pass
    return ""


# ----------------------------
# Prompt helpers
# ----------------------------
def _build_payload(isdt_json: Dict[str, Any], subject_json: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subjects": subject_json,
        "identity-semantics_discrete_tokenize": isdt_json,
    }


def _truncate_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


def _safe_shorten_for_retry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a VALID shortened payload dict for retry.
    Never cut the serialized JSON string in the middle.
    """
    out = {}

    subj = payload.get("subjects", {})
    isdt = payload.get("identity-semantics_discrete_tokenize", {})

    if isinstance(subj, dict):
        out_subj = {}
        for k, v in subj.items():
            if isinstance(v, (int, float, bool)) or v is None:
                out_subj[k] = v
            elif isinstance(v, str):
                out_subj[k] = _truncate_text(v, 80)
            else:
                out_subj[k] = _truncate_text(v, 120)
        out["subjects"] = out_subj
    else:
        out["subjects"] = subj

    if isinstance(isdt, dict):
        out_isdt = {}
        keep_keys = [
            "_stage",
            "graph_abnormal_prompt",
            "graph_abnormal_report",
            "stage1_router_output",
            "detection_task",
            "phenotype_summary",
            "node_identity",
        ]
        for k in keep_keys:
            if k not in isdt:
                continue
            v = isdt[k]

            if k == "graph_abnormal_prompt":
                out_isdt[k] = _truncate_text(v, 300)

            elif k == "graph_abnormal_report" and isinstance(v, dict):
                slim_report = {}
                for kk in [
                    "summary_stats",
                    "tail_stats",
                    "global_stats",
                    "top_abnormal_edges",
                    "top_abnormal_rois",
                ]:
                    if kk in v:
                        slim_report[kk] = v[kk]

                if "top_abnormal_edges" in slim_report and isinstance(slim_report["top_abnormal_edges"], list):
                    slim_report["top_abnormal_edges"] = slim_report["top_abnormal_edges"][:1]

                if "top_abnormal_rois" in slim_report and isinstance(slim_report["top_abnormal_rois"], list):
                    slim_report["top_abnormal_rois"] = slim_report["top_abnormal_rois"][:1]

                out_isdt[k] = slim_report

            elif k == "stage1_router_output" and isinstance(v, dict):
                slim_stage1 = {}
                for kk in ["graph_learning_rank", "reason", "stage"]:
                    if kk in v:
                        if kk == "reason":
                            slim_stage1[kk] = _truncate_text(v[kk], 100)
                        else:
                            slim_stage1[kk] = v[kk]
                out_isdt[k] = slim_stage1

            elif isinstance(v, str):
                out_isdt[k] = _truncate_text(v, 120)
            else:
                out_isdt[k] = v

        out["identity-semantics_discrete_tokenize"] = out_isdt
    else:
        out["identity-semantics_discrete_tokenize"] = isdt

    out["retry_mode"] = "shortened_valid_payload"
    return out


# ----------------------------
# Core LLM call (OpenRouter)
# ----------------------------
def extract_single_entry(
    isdt_json: Dict[str, Any],
    subject_json: Dict[str, Any],
    system_instructions: str,
    max_retries: int = 8,
    timeout_s: float = 60.0,
) -> Optional[Dict[str, Any]]:
    payload = _build_payload(isdt_json, subject_json)
    base_sleep = 2.0

    system_prompt = (
        system_instructions
        + "\n\nReturn ONLY valid JSON."
        + "\nDo not use markdown."
        + "\nDo not include analysis."
        + "\nDo not include chain-of-thought."
        + "\nKeep reason concise."
        + f"\nUse these experts only: {DEFAULT_GRAPH_RANK}."
        + "\nRequired keys: initial_decision, reason, graph_learning_rank, optional stage."
        + "\nIf the evidence is incomplete, still return your best JSON guess using only the available fields."
    )

    user_content = f"JSON input: {json.dumps(payload, ensure_ascii=False)}"

    for attempt in range(max_retries):
        try:
            def _do_call():
                return llm_client.chat.send(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=LLM_MAX_TOKENS,
                )

            response = _call_with_timeout(_do_call, timeout_s=timeout_s)
            text = _extract_text_from_openrouter_response(response)
            finish_reason = _get_finish_reason_from_response(response)

            data = _safe_json_loads(text or "")
            norm = _normalize_llm_output(data) if data is not None else None
            if norm is not None:
                return norm

            if finish_reason == "length" and attempt < max_retries - 1:
                retry_payload = _safe_shorten_for_retry(payload)
                user_content = f"JSON input: {json.dumps(retry_payload, ensure_ascii=False)}"
                wait_s = min(base_sleep * (2 ** min(attempt, 3)), 10.0)
                print(f"[LLM][retry-length] wait {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            raise RuntimeError(f"Model returned non-JSON / empty response. Raw: {text[:400]}")

        except Exception as e:
            msg = str(e)

            if "TIMEOUT" in msg.upper():
                wait_s = min(base_sleep * (2 ** attempt), 30.0)
                print(f"[LLM][timeout] wait {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            if _is_rate_limit(msg):
                wait_s = _sleep_from_msg(msg, default=30.0, cap=120.0)
                print(f"[LLM][429] retry in {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            if _is_network_like(msg):
                wait_s = min(base_sleep * (2 ** attempt), 30.0)
                print(f"[LLM][net] {msg[:120]}... wait {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            if _is_token_or_credit_error(msg):
                print(f"[LLM][fatal] {msg}")
                return None

            print(f"[LLM][fatal] {msg}")
            return None

    print("[LLM] Error: exceeded max_retries")
    return None


# ----------------------------
# Embedding text builder
# ----------------------------
def _build_embedding_text(data: Dict[str, Any]) -> str:
    stage = data.get("stage", None)
    if not isinstance(stage, str) or not stage.strip():
        stage = _detect_stage(llm_output=data)

    decision = int(data.get("initial_decision", 0) or 0)
    reason = str(data.get("reason", "") or "").strip()
    rank = str(data.get("graph_learning_rank", "") or "").strip()

    if not rank:
        rank = DEFAULT_GRAPH_RANK

    top_expert = rank.split(",")[0].strip() if rank else ""

    if stage == "A_router":
        return (
            f"Stage=A_router. "
            f"TopExpert={top_expert}. "
            f"ExpertRank={rank}. "
            f"StructuralReason={reason}."
        )

    return (
        f"Stage=B_final. "
        f"Decision={decision}. "
        f"DiagnosticReason={reason}. "
        f"ExpertRank={rank}."
    )


# ----------------------------
# Embedding call (Gemini)
# ----------------------------
def text_to_vector(
    json_data: Any,
    max_retries: int = 8,
    timeout_s: float = 60.0,
) -> Optional[List[float]]:
    if json_data is None:
        return None

    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except Exception:
        if isinstance(json_data, str):
            data = _safe_json_loads(json_data)
        else:
            data = None

    if not isinstance(data, dict):
        return None

    data = _normalize_llm_output(data)
    if data is None:
        return None

    text_to_embed = _build_embedding_text(data)
    base_sleep = 2.0

    for attempt in range(max_retries):
        try:
            def _do_call():
                cfg = None
                if EMB_OUTPUT_DIM and EMB_OUTPUT_DIM > 0:
                    try:
                        cfg = types.EmbedContentConfig(output_dimensionality=EMB_OUTPUT_DIM)
                    except Exception:
                        cfg = None

                return emb_client.models.embed_content(
                    model=TEXT_EMBEDDING_MODEL,
                    contents=text_to_embed,
                    config=cfg,
                )

            result = _call_with_timeout(_do_call, timeout_s=timeout_s)
            if result and getattr(result, "embeddings", None):
                vec = result.embeddings[0].values
                return list(vec) if vec is not None else None
            return None

        except Exception as e:
            msg = str(e)

            if "TIMEOUT" in msg.upper():
                wait_s = min(base_sleep * (2 ** attempt), 30.0)
                print(f"[EMB][timeout] wait {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            if _is_rate_limit(msg):
                wait_s = _sleep_from_msg(msg, default=30.0, cap=120.0)
                print(f"[EMB][429] retry in {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            if _is_network_like(msg):
                wait_s = min(base_sleep * (2 ** attempt), 30.0)
                print(f"[EMB][net] {msg[:120]}... wait {wait_s:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue

            print(f"[EMB][fatal] {msg}")
            return None

    print("[EMB] Error: exceeded max_retries")
    return None