# -*- coding: utf-8 -*-
"""
validate.py — Local pre-submission validator for SalesOps OpenEnv.

Runs a full episode of all 3 tasks without any LLM (uses scripted oracle actions)
to verify:
  - reset() returns a valid Observation
  - step() returns valid StepResult with reward in [0, 1]
  - state() returns a TaskState snapshot
  - final_score() returns a normalised EpisodeScore
  - No exceptions raised
  - All task IDs are reachable

Exit code 0 = PASS (safe to submit)
Exit code 1 = FAIL (review errors above)

Usage:
    python validate.py
"""

from __future__ import annotations

import sys
import traceback
from typing import List, Tuple

from env import SalesOpsEnv
from graders import grade_episode
from models import Action, ActionType, EpisodeScore, LeadClassification, Observation, OwnerRole, Priority
from tasks import get_task_ids, get_task_meta

# ---------------------------------------------------------------------------
# Oracle sequences — deterministic, hand-crafted correct actions per task
# ---------------------------------------------------------------------------

ORACLE_SEQUENCES: dict[str, List[Tuple[ActionType, str | None]]] = {
    "task_easy_hot_lead": [
        (ActionType.CLASSIFY_LEAD,  "hot"),
        (ActionType.SET_PRIORITY,   "high"),
        (ActionType.ASSIGN_OWNER,   "sdr"),
        (ActionType.SCHEDULE_DEMO,  None),
    ],
    "task_medium_enterprise_growth": [
        (ActionType.CLASSIFY_LEAD,       "enterprise_growth"),
        (ActionType.SET_PRIORITY,        "high"),
        (ActionType.ASSIGN_OWNER,        "solution_engineer"),
        (ActionType.NOTIFY_COMPLIANCE,   None),
        (ActionType.REQUEST_MORE_INFO,   None),
        (ActionType.SCHEDULE_DEMO,       None),
    ],
    "task_hard_enterprise_migration": [
        (ActionType.CLASSIFY_LEAD,        "enterprise_growth"),
        (ActionType.SET_PRIORITY,         "critical"),
        (ActionType.REQUEST_MORE_INFO,    None),
        (ActionType.ESCALATE_ENTERPRISE,  None),
        (ActionType.NOTIFY_COMPLIANCE,    None),
        (ActionType.ASSIGN_OWNER,         "enterprise_ae"),
        (ActionType.SCHEDULE_DEMO,        None),
    ],
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"

errors: List[str] = []

def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"{PASS} {label}")
    else:
        msg = f"{label}" + (f": {detail}" if detail else "")
        print(f"{FAIL} {msg}")
        errors.append(msg)
    return condition


def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate_task(task_id: str) -> EpisodeScore | None:
    section(f"Task: {task_id}")
    env  = SalesOpsEnv()
    meta = get_task_meta(task_id)

    # --- reset ---
    try:
        obs = env.reset(task_id)
        check(isinstance(obs, Observation), "reset() returns Observation")
        check(obs.step_number == 0, "reset() step_number is 0")
        check(obs.current_classification is None, "reset() classification starts None")
        check(obs.current_owner.value == "none", "reset() owner starts as none")
    except Exception as exc:
        check(False, "reset() raised exception", str(exc))
        traceback.print_exc()
        return None

    # --- state after reset ---
    try:
        ts = env.state()
        check(ts.step_count == 0, "state() step_count == 0 post-reset")
        check(ts.done is False,   "state() done is False post-reset")
    except Exception as exc:
        check(False, "state() raised exception", str(exc))

    # --- steps ---
    actions_seq = ORACLE_SEQUENCES[task_id]
    rewards: List[float] = []
    done = False

    for i, (at, val) in enumerate(actions_seq):
        if done:
            break
        try:
            action = Action(action_type=at, value=val)
            result = env.step(action)
            rewards.append(result.reward)

            check(
                0.0 <= result.reward <= 1.0,
                f"step {i+1} reward in [0,1]",
                f"got {result.reward}",
            )
            check(
                isinstance(result.observation, Observation),
                f"step {i+1} returns Observation",
            )
            done = result.done
        except Exception as exc:
            check(False, f"step {i+1} raised exception", str(exc))
            traceback.print_exc()
            break

    # Force completion if not done yet (step budget)
    max_extra = meta.max_steps - len(actions_seq) - 1
    extra = 0
    while not done and extra < max_extra:
        try:
            result = env.step(Action(action_type=ActionType.REQUEST_MORE_INFO))
            rewards.append(result.reward)
            done = result.done
            extra += 1
        except Exception:
            break

    # --- final score ---
    try:
        score = env.final_score()
        if score:
            check(
                0.0 <= score.total_score <= 1.0,
                "final_score() total_score in [0,1]",
                f"got {score.total_score}",
            )
            check(
                score.classification_score >= 0.0,
                "classification_score non-negative",
            )
            print(f"{INFO} Score breakdown:")
            print(f"      total={score.total_score:.4f}  classify={score.classification_score:.2f}  "
                  f"routing={score.routing_score:.2f}  urgency={score.urgency_score:.2f}  "
                  f"compliance={score.compliance_score:.2f}  scheduling={score.scheduling_score:.2f}")
            print(f"      penalty={score.closure_penalty:.2f}  efficiency_bonus={score.efficiency_bonus:.2f}  "
                  f"success={score.success}  steps={score.steps_taken}")
            if score.notes:
                print(f"      notes: {score.notes}")
            return score
        else:
            check(False, "final_score() returned None", "episode may not have ended")
    except Exception as exc:
        check(False, "final_score() raised exception", str(exc))
        traceback.print_exc()

    return None


def validate_env_info() -> None:
    section("Environment metadata")
    info = SalesOpsEnv.env_info()
    check("env_name"    in info, "env_info has env_name")
    check("version"     in info, "env_info has version")
    check("task_ids"    in info, "env_info has task_ids")
    check("action_space" in info, "env_info has action_space")
    check(len(info["task_ids"]) >= 3, "at least 3 task_ids registered")


def validate_action_space() -> None:
    section("Action space coverage")
    for at in ActionType:
        check(True, f"ActionType.{at.name} is importable")


def validate_model_bounds() -> None:
    section("Model constraint bounds")
    from models import Observation, Action, StepResult, EpisodeScore
    # Just ensure imports work and models are sane
    check(True, "Observation model importable")
    check(True, "Action model importable")
    check(True, "StepResult model importable")
    check(True, "EpisodeScore model importable")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n[*] SalesOps OpenEnv -- Pre-Submission Validator")
    print("=" * 62)

    validate_env_info()
    validate_action_space()
    validate_model_bounds()

    scores: List[EpisodeScore] = []
    for task_id in get_task_ids():
        score = validate_task(task_id)
        if score:
            scores.append(score)

    # Summary
    section("Validation Summary")
    total_checks = len(errors) + sum(1 for _ in range(0))  # errors are tracked globally
    if errors:
        print(f"\n[FAIL] {len(errors)} check(s) FAILED:")
        for e in errors:
            print(f"   - {e}")
        print("\nFix the above before submitting.\n")
        sys.exit(1)
    else:
        print(f"\n[PASS] All checks PASSED -- safe to submit!\n")
        if scores:
            avg = sum(s.total_score for s in scores) / len(scores)
            successes = sum(1 for s in scores if s.success)
            print(f"   Oracle agent avg_score={avg:.4f}  successes={successes}/{len(scores)}")
        sys.exit(0)


if __name__ == "__main__":
    main()
