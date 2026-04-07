"""
graders.py — Deterministic, normalized reward graders for SalesOps OpenEnv.

Every grader:
  - Returns a float in [0.0, 1.0]
  - Is fully deterministic (no randomness)
  - Awards partial credit for partial progress
  - Applies penalties for destructive wrong actions
"""

from __future__ import annotations
from typing import Optional
from models import (
    Action, ActionType, EpisodeScore, LeadClassification, Observation,
    OwnerRole, Priority, TaskMetadata, WorkflowStage,
)


# ---------------------------------------------------------------------------
# Sub-dimension graders (each returns [0,1])
# ---------------------------------------------------------------------------

def _grade_classification(obs: Observation, expected: LeadClassification) -> float:
    """Did the agent classify correctly?"""
    if obs.current_classification is None:
        return 0.0
    if obs.current_classification == expected:
        return 1.0
    # Partial credit for adjacent classifications
    adjacency = {
        (LeadClassification.HOT,              LeadClassification.WARM):           0.4,
        (LeadClassification.WARM,             LeadClassification.HOT):            0.4,
        (LeadClassification.ENTERPRISE_GROWTH, LeadClassification.HOT):           0.3,
        (LeadClassification.HOT,              LeadClassification.ENTERPRISE_GROWTH): 0.3,
        (LeadClassification.WARM,             LeadClassification.COLD):           0.2,
        (LeadClassification.COLD,             LeadClassification.WARM):           0.2,
    }
    return adjacency.get((obs.current_classification, expected), 0.0)


def _grade_routing(obs: Observation, expected_owner: OwnerRole) -> float:
    """Did the agent assign the right owner role?"""
    if obs.current_owner == OwnerRole.NONE:
        return 0.0
    if obs.current_owner == expected_owner:
        return 1.0
    # Partial credit for reasonable wrong routing
    partial = {
        (OwnerRole.SDR,               OwnerRole.AE):                0.4,
        (OwnerRole.AE,                OwnerRole.SDR):                0.4,
        (OwnerRole.SOLUTION_ENGINEER, OwnerRole.AE):                 0.5,
        (OwnerRole.AE,                OwnerRole.SOLUTION_ENGINEER):  0.5,
        (OwnerRole.ENTERPRISE_AE,     OwnerRole.SOLUTION_ENGINEER):  0.5,
        (OwnerRole.SOLUTION_ENGINEER, OwnerRole.ENTERPRISE_AE):      0.5,
        (OwnerRole.ENTERPRISE_AE,     OwnerRole.AE):                 0.3,
    }
    return partial.get((obs.current_owner, expected_owner), 0.0)


def _grade_urgency(obs: Observation, expected_priority: Priority) -> float:
    """Was the urgency / priority set correctly?"""
    if obs.current_priority is None:
        return 0.0
    if obs.current_priority == expected_priority:
        return 1.0
    # Priority ladder: CRITICAL > HIGH > MEDIUM > LOW
    ladder = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
    exp_idx = ladder.index(expected_priority)
    got_idx = ladder.index(obs.current_priority)
    distance = abs(exp_idx - got_idx)
    # 1 step off → 0.5, 2 steps off → 0.2, 3+ → 0.0
    return {1: 0.5, 2: 0.2}.get(distance, 0.0)


def _grade_compliance(obs: Observation, meta: TaskMetadata) -> float:
    """Was compliance correctly handled when required?"""
    if not meta.requires_compliance:
        # Bonus if agent proactively notified anyway (forward-thinking)
        return 1.0 if obs.compliance_notified else 1.0
    # Compliance required
    if not obs.compliance_notified:
        return 0.0
    # Full credit only if notified before demo was scheduled
    # (penalize scheduling before compliance clearance)
    if obs.demo_scheduled and not obs.compliance_notified:
        return 0.2
    return 1.0


def _grade_scheduling(obs: Observation, meta: TaskMetadata) -> float:
    """Was a demo / discovery call scheduled when appropriate?"""
    if not meta.requires_demo:
        return 1.0
    if not obs.demo_scheduled:
        return 0.0
    # For hard tasks: scheduling before compliance = partial only
    if meta.requires_compliance and not obs.compliance_notified:
        return 0.35
    return 1.0


def _closure_penalty(obs: Observation, meta: TaskMetadata) -> float:
    """
    Apply a strong penalty if the agent chose close_lost on a lead that
    should have been pursued, or close_won without completing required steps.
    Returns a value in [0, 0.5] to be subtracted from total.
    """
    if obs.current_stage == WorkflowStage.CLOSE_LOST:
        expected_close = meta.expected_classification == LeadClassification.DISQUALIFIED
        if not expected_close:
            return 0.50   # Severe: closed a good lead as lost

    if obs.current_stage == WorkflowStage.CLOSE_WON:
        # Close won is valid only after demo & (compliance if required)
        if meta.requires_compliance and not obs.compliance_notified:
            return 0.40
        if meta.requires_demo and not obs.demo_scheduled:
            return 0.30

    return 0.0


def _efficiency_bonus(steps_taken: int, max_steps: int) -> float:
    """
    Small bonus for solving task in fewer steps than allowed.
    Max bonus: +0.10 (capped). Only applied if base score ≥ 0.80.
    """
    ratio = steps_taken / max_steps
    if ratio <= 0.40:
        return 0.10
    elif ratio <= 0.60:
        return 0.06
    elif ratio <= 0.80:
        return 0.03
    return 0.0


# ---------------------------------------------------------------------------
# Per-task graders (dispatch table)
# ---------------------------------------------------------------------------

def _compute_weighted_score(
    obs: Observation,
    meta: TaskMetadata,
    steps_taken: int,
) -> EpisodeScore:
    """
    Core grading engine. Combines sub-dimension scores using task-specific
    weights to produce a final normalized score in [0, 1].
    """
    w = meta.scoring_weights

    c_score  = _grade_classification(obs, meta.expected_classification)
    r_score  = _grade_routing(obs, meta.expected_owner)
    u_score  = _grade_urgency(obs, meta.expected_priority)
    co_score = _grade_compliance(obs, meta) if meta.requires_compliance else 1.0
    s_score  = _grade_scheduling(obs, meta)
    penalty  = _closure_penalty(obs, meta)

    # Weighted sum
    raw = (
        w.get("classification", 0) * c_score +
        w.get("routing",        0) * r_score +
        w.get("urgency",        0) * u_score +
        w.get("compliance",     0) * co_score +
        w.get("scheduling",     0) * s_score
    )

    raw = max(0.0, raw - penalty)

    # Efficiency bonus (only when already performing well)
    eff_bonus = 0.0
    if raw >= 0.80:
        eff_bonus = _efficiency_bonus(steps_taken, meta.max_steps)

    total = min(0.999, raw + eff_bonus)
    total = max(0.001, total)   # strictly open interval — platform requirement

    return EpisodeScore(
        task_id=meta.task_id,
        total_score=round(total, 4),
        classification_score=round(c_score, 4),
        routing_score=round(r_score, 4),
        urgency_score=round(u_score, 4),
        compliance_score=round(co_score, 4),
        scheduling_score=round(s_score, 4),
        closure_penalty=round(penalty, 4),
        steps_taken=steps_taken,
        efficiency_bonus=round(eff_bonus, 4),
        success=(total >= 0.80),
        notes=_build_notes(c_score, r_score, u_score, co_score, s_score, penalty, meta),
    )


def _build_notes(c, r, u, co, s, penalty, meta: TaskMetadata) -> str:
    issues = []
    if c < 0.80:
        issues.append(f"classification={c:.2f} (expected {meta.expected_classification.value})")
    if r < 0.80:
        issues.append(f"routing={r:.2f} (expected {meta.expected_owner.value})")
    if u < 0.80:
        issues.append(f"urgency={u:.2f} (expected {meta.expected_priority.value})")
    if meta.requires_compliance and co < 0.80:
        issues.append(f"compliance={co:.2f}")
    if meta.requires_demo and s < 0.80:
        issues.append("demo not scheduled")
    if penalty > 0:
        issues.append(f"wrong_closure_penalty={penalty:.2f}")
    return "; ".join(issues) if issues else "All dimensions satisfied."


# ---------------------------------------------------------------------------
# Dense step-level reward shaper
# ---------------------------------------------------------------------------

class RewardShaper:
    """
    Produces dense per-step rewards to guide the agent trajectory.
    All individual rewards are in [0, 1] before combination.
    The caller is responsible for clipping the final episode score.
    """

    def __init__(self, meta: TaskMetadata):
        self.meta = meta
        self._repeated: dict[str, int] = {}

    # ---- public API --------------------------------------------------------

    def shape(
        self,
        action: Action,
        obs_before: Observation,
        obs_after: Observation,
        done: bool,
    ) -> float:
        """
        Compute the step reward.
        Returns a float in [-0.10, 0.20].  (Small dense signal, not episode score.)
        """
        reward = 0.0
        at = action.action_type

        # 1. Repetition penalty
        key = at.value + (action.value or "")
        self._repeated[key] = self._repeated.get(key, 0) + 1
        if self._repeated[key] > 1:
            return -0.05 * min(self._repeated[key] - 1, 2)   # cap at -0.10

        # 2. Correct classification (immediate feedback)
        if at == ActionType.CLASSIFY_LEAD and action.value:
            try:
                chosen = LeadClassification(action.value)
                reward += 0.12 if chosen == self.meta.expected_classification else -0.04
            except ValueError:
                reward -= 0.05

        # 3. Routing reward
        if at == ActionType.ASSIGN_OWNER and action.value:
            try:
                chosen = OwnerRole(action.value)
                reward += 0.10 if chosen == self.meta.expected_owner else -0.04
            except ValueError:
                reward -= 0.05

        # 4. Priority reward
        if at == ActionType.SET_PRIORITY and action.value:
            try:
                chosen = Priority(action.value)
                reward += 0.08 if chosen == self.meta.expected_priority else -0.02
            except ValueError:
                reward -= 0.03

        # 5. Compliance reward (bonus for proactive)
        if at == ActionType.NOTIFY_COMPLIANCE:
            if self.meta.requires_compliance:
                reward += 0.15
            else:
                reward += 0.03   # small proactive bonus

        # 6. Escalation reward
        if at == ActionType.ESCALATE_ENTERPRISE:
            if self.meta.requires_escalation:
                reward += 0.12
            else:
                reward -= 0.02   # unnecessary escalation wastes resources

        # 7. Demo scheduling reward (higher when prerequisites met)
        if at == ActionType.SCHEDULE_DEMO:
            if not obs_before.demo_scheduled:
                prereqs_met = (
                    obs_before.current_classification is not None and
                    obs_before.current_owner != OwnerRole.NONE
                )
                compliance_ok = (
                    not self.meta.requires_compliance or
                    obs_before.compliance_notified
                )
                if prereqs_met and compliance_ok:
                    reward += 0.18
                elif prereqs_met:
                    reward += 0.08   # scheduled but skipped compliance
                else:
                    reward += 0.03   # premature schedule

        # 8. Wrong closure penalties
        if at == ActionType.CLOSE_LOST:
            if self.meta.expected_classification != LeadClassification.DISQUALIFIED:
                reward -= 0.10   # closing a good lead

        if at == ActionType.CLOSE_WON:
            if not obs_before.demo_scheduled:
                reward -= 0.08   # closing without a demo
            if self.meta.requires_compliance and not obs_before.compliance_notified:
                reward -= 0.10   # compliance bypass

        # 9. Late escalation penalty (should escalate before step 5 on hard tasks)
        if (
            at == ActionType.ESCALATE_ENTERPRISE and
            self.meta.requires_escalation and
            obs_before.step_number > 5
        ):
            reward -= 0.04   # late-escalation penalty

        # 10. Request more info — positive early, negative late
        if at == ActionType.REQUEST_MORE_INFO:
            if obs_before.step_number <= 3:
                reward += 0.06
            elif obs_before.step_number > 8:
                reward -= 0.03   # stalling

        return round(max(-0.10, min(0.20, reward)), 4)


# ---------------------------------------------------------------------------
# Public grading API
# ---------------------------------------------------------------------------

def grade_episode(obs: Observation, meta: TaskMetadata, steps_taken: int) -> EpisodeScore:
    """Entry point called by env.py at episode end."""
    return _compute_weighted_score(obs, meta, steps_taken)
