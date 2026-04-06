"""
inference.py — SalesOps OpenEnv Baseline Agent

Strict log format required by the OpenEnv validator:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables (loaded from .env automatically):
    API_BASE_URL   — OpenAI-compatible base URL
                     Default: https://generativelanguage.googleapis.com/v1beta/openai/
    MODEL_NAME     — Model to use  (default: gemini-2.0-flash)
    HF_TOKEN       — API key for the provider (Gemini key, HF token, Groq key, etc.)
    IMAGE_NAME     — Docker image name (optional, metadata only)

Runtime budget: < 20 min, 2 vCPU / 8 GB RAM.
Exits 0 on success, 1 on fatal error (always emits [END]).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

# Load .env file automatically (safe — does nothing if file missing)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # env vars already set in shell take priority
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

from openai import OpenAI

from env import SalesOpsEnv
from models import Action, ActionType
from tasks import get_task_ids, get_task_meta

# ---------------------------------------------------------------------------
# Configuration from env vars
# ---------------------------------------------------------------------------

# Defaults point to Google Gemini free API (OpenAI-compatible)
API_BASE_URL: str = os.getenv(
    "API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "gemini-2.0-flash")
HF_TOKEN:     str = os.getenv("HF_TOKEN",     "")
IMAGE_NAME:   str = os.getenv("IMAGE_NAME",   "salesops-openenv:latest")

MAX_RETRIES   = 3
RETRY_DELAY_S = 2.0

# ---------------------------------------------------------------------------
# OpenAI client (OpenAI-compatible — works with any provider)
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "MISSING"),
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Logging helpers — STRICT FORMAT
# ---------------------------------------------------------------------------

def log_start(task_id: str, env_name: str, model: str) -> None:
    print(f"[START] task={task_id} env={env_name} model={model}", flush=True)


def log_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_field = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_field}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an enterprise sales operations AI agent working inside a CRM.

Given a lead observation, pick the SINGLE best next action to advance the deal.

ACTION TYPES and their allowed values:
  "classify_lead"       value must be one of: "hot", "warm", "cold", "enterprise_growth", "disqualified"
  "set_priority"        value must be one of: "critical", "high", "medium", "low"
  "assign_owner"        value must be one of: "sdr", "ae", "solution_engineer", "enterprise_ae", "compliance_officer", "legal"
  "request_more_info"   value must be: null
  "escalate_enterprise" value must be: null
  "notify_compliance"   value must be: null
  "schedule_demo"       value must be: null
  "close_lost"          value must be: null
  "close_won"           value must be: null

RULES:
1. First classify the lead, then set priority, then assign an owner.
2. If there are compliance_flags in the observation, call notify_compliance before schedule_demo.
3. If company_size is enterprise, call escalate_enterprise before schedule_demo.
4. Only call schedule_demo after classification, routing and any required compliance steps are done.
5. Never repeat an action you already took (check previous_actions).

EXAMPLE OUTPUT (copy this format exactly):
{"action_type": "classify_lead", "value": "hot", "reasoning": "High urgency score and pricing page visit indicate hot lead."}

RESPOND WITH ONLY THE JSON OBJECT. No markdown, no explanation, no extra text.
""").strip()


def _build_user_message(obs_dict: Dict[str, Any], step: int) -> str:
    return (
        f"Step {step}. Current observation:\n"
        + json.dumps(obs_dict, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def _call_llm(messages: List[Dict[str, str]]) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Use JSON object mode when the provider supports it (Gemini, GPT-4o, etc.)
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            err_str = str(exc)
            # If provider doesn't support response_format, retry without it
            if "response_format" in err_str.lower() or "json_object" in err_str.lower():
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=300,
                    )
                    return resp.choices[0].message.content or ""
                except Exception:
                    pass
            if attempt == MAX_RETRIES:
                print(f"[WARN] LLM call failed after {MAX_RETRIES} attempts: {exc}",
                      file=sys.stderr, flush=True)
                return None
            time.sleep(RETRY_DELAY_S * attempt)
    return None


def _parse_action(raw: Optional[str]) -> Optional[Action]:
    if not raw:
        return None
    try:
        text = raw.strip()

        # Strip markdown fences: ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json) and last line (```)
            inner = lines[1:]
            if inner and inner[-1].strip().startswith("```"):
                inner = inner[:-1]
            text = "\n".join(inner).strip()

        # If the model returned multiple JSON objects, take the first one
        if text.count("{") > 1:
            end = text.index("}") + 1
            text = text[:end]

        payload = json.loads(text)

        raw_at = payload.get("action_type", "")
        action_type = ActionType(raw_at.lower().strip())

        # value: treat the string "null", "", None all as None
        raw_val = payload.get("value")
        value = None if raw_val in (None, "null", "", "none") else str(raw_val).strip()

        reasoning = payload.get("reasoning", "")
        return Action(action_type=action_type, value=value, reasoning=reasoning)

    except Exception as exc:
        # Show what Gemini actually returned so we can debug
        preview = (raw or "")[:200].replace("\n", " ")
        print(f"[WARN] parse_action failed ({exc}). Raw: {preview}",
              file=sys.stderr, flush=True)
        return None


# ---------------------------------------------------------------------------
# Safe fallback action (never crashes, always valid)
# ---------------------------------------------------------------------------

def _fallback_action() -> Action:
    """Return a harmless explore action when LLM fails."""
    return Action(action_type=ActionType.REQUEST_MORE_INFO, value=None,
                  reasoning="LLM parse failure — requesting more info as safe fallback.")


# ---------------------------------------------------------------------------
# Single task run
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> Dict[str, Any]:
    env      = SalesOpsEnv()
    meta     = get_task_meta(task_id)
    env_name = SalesOpsEnv.ENV_NAME

    log_start(task_id, env_name, MODEL_NAME)

    obs      = env.reset(task_id)
    done     = False
    step_num = 0
    rewards: List[float] = []
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done:
        step_num += 1
        obs_dict = obs.model_dump(mode="json")

        # Remove available_actions list from dict to keep context tight
        obs_dict.pop("available_actions", None)

        messages.append({"role": "user", "content": _build_user_message(obs_dict, step_num)})

        raw_response = _call_llm(messages)
        action       = _parse_action(raw_response) or _fallback_action()

        # Add assistant turn so model sees its own history
        messages.append({
            "role":    "assistant",
            "content": raw_response or json.dumps(action.model_dump()),
        })

        result     = env.step(action)
        obs        = result.observation
        done       = result.done
        rewards.append(result.reward)

        log_step(
            step=step_num,
            action_str=action.to_log_str(),
            reward=result.reward,
            done=done,
            error=result.error,
        )

        # Safety: enforce max steps regardless
        if step_num >= meta.max_steps:
            break

    # Final score
    final = env.final_score()
    score   = final.total_score if final else 0.0
    success = final.success     if final else False

    log_end(success=success, steps=step_num, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps":   step_num,
        "score":   score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    task_ids  = get_task_ids()
    results   = []
    exit_code = 0

    # Allow running a single task via CLI: python inference.py task_easy_hot_lead
    if len(sys.argv) > 1:
        requested = sys.argv[1]
        if requested in task_ids:
            task_ids = [requested]
        else:
            print(
                f"[ERROR] Unknown task '{requested}'. "
                f"Valid: {task_ids}",
                file=sys.stderr,
            )
            # Always emit a proper [END] even on fatal error
            log_end(success=False, steps=0, score=0.0, rewards=[])
            sys.exit(1)

    for tid in task_ids:
        try:
            result = run_task(tid)
            results.append(result)
        except Exception as exc:
            # Always emit [END] even on crash
            print(f"[ERROR] Task {tid} crashed: {exc}", file=sys.stderr, flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            exit_code = 1

    # Summary across all tasks
    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        successes = sum(1 for r in results if r["success"])
        print(
            f"\n[SUMMARY] tasks={len(results)} successes={successes} "
            f"avg_score={avg_score:.4f}",
            flush=True,
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
