"""
env.py — SalesOps OpenEnv: Core Environment

Implements the three mandatory OpenEnv methods:
    reset()  → Observation
    step()   → StepResult
    state()  → TaskState

All transitions are deterministic given the same task_id.
No randomness is introduced post-reset.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from models import (
    Action, ActionType, EpisodeScore, LeadClassification, Observation,
    OwnerRole, Priority, StepResult, TaskState, WorkflowStage,
)
from tasks import get_initial_observation, get_task_ids, get_task_meta, TASK_REGISTRY
from graders import RewardShaper, grade_episode


# ---------------------------------------------------------------------------
# Transition helpers
# ---------------------------------------------------------------------------

_STAGE_AFTER_CLASSIFY: dict[LeadClassification, WorkflowStage] = {
    LeadClassification.HOT:               WorkflowStage.QUALIFYING,
    LeadClassification.WARM:              WorkflowStage.QUALIFYING,
    LeadClassification.COLD:              WorkflowStage.QUALIFYING,
    LeadClassification.ENTERPRISE_GROWTH: WorkflowStage.DISCOVERY,
    LeadClassification.DISQUALIFIED:      WorkflowStage.CLOSE_LOST,
}


def _parse_classification(value: Optional[str]) -> Optional[LeadClassification]:
    if value is None:
        return None
    try:
        return LeadClassification(value.lower())
    except ValueError:
        return None


def _parse_priority(value: Optional[str]) -> Optional[Priority]:
    if value is None:
        return None
    try:
        return Priority(value.lower())
    except ValueError:
        return None


def _parse_owner(value: Optional[str]) -> Optional[OwnerRole]:
    if value is None:
        return None
    try:
        return OwnerRole(value.lower())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SalesOpsEnv
# ---------------------------------------------------------------------------

class SalesOpsEnv:
    """
    Production-grade OpenEnv environment simulating an enterprise
    CRM / sales-ops workflow.

    Lifecycle:
        env = SalesOpsEnv()
        obs = env.reset(task_id)
        while not done:
            result = env.step(action)
            obs, reward, done = result.observation, result.reward, result.done
        score = env.final_score()
    """

    # Benchmark identity (referenced by inference.py and openenv.yaml)
    ENV_NAME    = "salesops-openenv-v1"
    ENV_VERSION = "1.0.0"

    def __init__(self) -> None:
        self._task_state: Optional[TaskState] = None
        self._shaper:     Optional[RewardShaper] = None

    # ------------------------------------------------------------------
    # Public contract: reset / step / state
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Initialise a clean, deterministic episode for the given task.

        Args:
            task_id: One of the registered task IDs (see tasks.get_task_ids()).

        Returns:
            Initial Observation (no agent actions applied yet).
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available: {get_task_ids()}"
            )

        meta    = get_task_meta(task_id)
        obs     = get_initial_observation(task_id)

        self._task_state = TaskState(
            task_id=task_id,
            observation=copy.deepcopy(obs),
            step_count=0,
            cumulative_reward=0.0,
            reward_history=[],
            done=False,
            success=False,
            repeated_action_counts={},
            wrong_close_issued=False,
            late_escalation_penalty_applied=False,
        )
        self._shaper = RewardShaper(meta)
        return copy.deepcopy(obs)

    def step(self, action: Action) -> StepResult:
        """
        Apply one agent action and advance the environment.

        Args:
            action: Typed Action from the agent.

        Returns:
            StepResult with updated Observation, step reward, done flag, and info dict.
        """
        if self._task_state is None:
            raise RuntimeError("Call reset() before step().")

        ts   = self._task_state
        meta = get_task_meta(ts.task_id)

        if ts.done:
            return StepResult(
                observation=copy.deepcopy(ts.observation),
                reward=0.0,
                done=True,
                info={"warning": "Episode already finished. Call reset()."},
            )

        obs_before = copy.deepcopy(ts.observation)

        # --- Apply transition ---
        error_msg: Optional[str] = None
        obs_after  = copy.deepcopy(obs_before)
        obs_after.step_number += 1

        transition_ok = self._apply_transition(action, obs_after)
        if not transition_ok:
            error_msg = f"Invalid action value '{action.value}' for '{action.action_type.value}'."

        obs_after.previous_actions.append(action.to_log_str())

        # --- Dense step reward ---
        step_reward = self._shaper.shape(action, obs_before, obs_after, done=False)
        step_reward = max(0.0, min(1.0, step_reward + 0.05))  # small base offset, clipped

        ts.step_count += 1
        ts.cumulative_reward += step_reward
        ts.reward_history.append(round(step_reward, 4))
        ts.observation = obs_after

        # --- Episode termination conditions ---
        done = self._check_done(obs_after, meta)
        ts.done = done

        if done:
            episode_score = grade_episode(obs_after, meta, ts.step_count)
            ts.success = episode_score.success
            info = {
                "episode_score": episode_score.model_dump(),
                "final_stage":   obs_after.current_stage.value,
            }
        else:
            info = {
                "step":        ts.step_count,
                "stage":       obs_after.current_stage.value,
                "cumulative":  round(ts.cumulative_reward, 4),
            }

        if error_msg:
            info["error"] = error_msg

        return StepResult(
            observation=copy.deepcopy(obs_after),
            reward=step_reward,
            done=done,
            info=info,
            error=error_msg,
        )

    def state(self) -> TaskState:
        """Return a snapshot of the current episode state (read-only copy)."""
        if self._task_state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._task_state)

    def final_score(self) -> Optional[EpisodeScore]:
        """Return the final graded score if the episode has ended, else None."""
        if self._task_state is None or not self._task_state.done:
            return None
        meta = get_task_meta(self._task_state.task_id)
        return grade_episode(
            self._task_state.observation,
            meta,
            self._task_state.step_count,
        )

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def _apply_transition(self, action: Action, obs: Observation) -> bool:
        """
        Mutate obs in-place according to the action.
        Returns True on valid transition, False on invalid value.
        """
        at = action.action_type

        if at == ActionType.CLASSIFY_LEAD:
            classification = _parse_classification(action.value)
            if classification is None:
                obs.current_classification = LeadClassification.WARM  # safe fallback
                return False
            obs.current_classification = classification
            obs.current_stage = _STAGE_AFTER_CLASSIFY.get(
                classification, WorkflowStage.QUALIFYING
            )
            if classification == LeadClassification.DISQUALIFIED:
                obs.current_stage = WorkflowStage.CLOSE_LOST

        elif at == ActionType.SET_PRIORITY:
            priority = _parse_priority(action.value)
            if priority is None:
                return False
            obs.current_priority = priority

        elif at == ActionType.ASSIGN_OWNER:
            owner = _parse_owner(action.value)
            if owner is None:
                return False
            obs.current_owner = owner
            # Stage promotion if classification already done
            if (
                obs.current_classification is not None and
                obs.current_stage == WorkflowStage.QUALIFYING
            ):
                obs.current_stage = WorkflowStage.DISCOVERY

        elif at == ActionType.REQUEST_MORE_INFO:
            obs.more_info_requested = True

        elif at == ActionType.ESCALATE_ENTERPRISE:
            obs.enterprise_escalated = True
            if obs.current_owner not in (
                OwnerRole.ENTERPRISE_AE, OwnerRole.COMPLIANCE_OFFICER
            ):
                obs.current_owner = OwnerRole.ENTERPRISE_AE
            obs.current_stage = WorkflowStage.DISCOVERY

        elif at == ActionType.NOTIFY_COMPLIANCE:
            obs.compliance_notified = True
            if obs.current_stage not in (
                WorkflowStage.COMPLIANCE_HOLD,
                WorkflowStage.DEMO_SCHEDULED,
                WorkflowStage.PROPOSAL,
            ):
                obs.current_stage = WorkflowStage.COMPLIANCE_HOLD

        elif at == ActionType.SCHEDULE_DEMO:
            obs.demo_scheduled = True
            if obs.current_stage not in (
                WorkflowStage.CLOSE_WON, WorkflowStage.CLOSE_LOST
            ):
                obs.current_stage = WorkflowStage.DEMO_SCHEDULED

        elif at == ActionType.CLOSE_LOST:
            obs.current_stage = WorkflowStage.CLOSE_LOST

        elif at == ActionType.CLOSE_WON:
            obs.current_stage = WorkflowStage.CLOSE_WON

        return True

    # ------------------------------------------------------------------
    # Termination conditions
    # ------------------------------------------------------------------

    def _check_done(self, obs: Observation, meta) -> bool:
        # Hard terminal states
        if obs.current_stage in (WorkflowStage.CLOSE_WON, WorkflowStage.CLOSE_LOST):
            return True

        # Step limit — prevent infinite loops
        if obs.step_number >= meta.max_steps:
            return True

        # Natural completion: all required sub-goals satisfied
        classification_done = obs.current_classification is not None
        routing_done        = obs.current_owner != OwnerRole.NONE
        priority_done       = obs.current_priority is not None
        compliance_done     = (not meta.requires_compliance) or obs.compliance_notified
        demo_done           = (not meta.requires_demo) or obs.demo_scheduled

        if all([classification_done, routing_done, priority_done, compliance_done, demo_done]):
            return True

        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def available_task_ids() -> List[str]:
        return get_task_ids()

    @staticmethod
    def env_info() -> dict:
        return {
            "env_name":    SalesOpsEnv.ENV_NAME,
            "version":     SalesOpsEnv.ENV_VERSION,
            "task_ids":    get_task_ids(),
            "action_space": [a.value for a in ActionType],
        }
