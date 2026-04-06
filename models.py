"""
models.py — Typed Pydantic models for SalesOps OpenEnv.

All data flowing through the environment is strictly typed.
Reward is always normalized to [0.0, 1.0].
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LeadSource(str, Enum):
    PRICING_PAGE   = "pricing_page"
    INBOUND_FORM   = "inbound_form"
    OUTBOUND       = "outbound"
    REFERRAL       = "referral"
    PARTNER        = "partner"
    LINKEDIN       = "linkedin"
    CONFERENCE     = "conference"
    UNKNOWN        = "unknown"


class CompanySize(str, Enum):
    STARTUP     = "startup"       # 1 – 50
    SMB         = "smb"           # 51 – 500
    MID_MARKET  = "mid_market"    # 501 – 2000
    ENTERPRISE  = "enterprise"    # 2001+


class Region(str, Enum):
    NORTH_AMERICA  = "north_america"
    EUROPE         = "europe"
    APAC           = "apac"
    LATAM          = "latam"
    MENA           = "mena"


class LeadClassification(str, Enum):
    HOT            = "hot"
    WARM           = "warm"
    COLD           = "cold"
    ENTERPRISE_GROWTH = "enterprise_growth"
    DISQUALIFIED   = "disqualified"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class OwnerRole(str, Enum):
    SDR                = "sdr"
    AE                 = "ae"
    SOLUTION_ENGINEER  = "solution_engineer"
    ENTERPRISE_AE      = "enterprise_ae"
    COMPLIANCE_OFFICER = "compliance_officer"
    LEGAL              = "legal"
    NONE               = "none"


class WorkflowStage(str, Enum):
    NEW             = "new"
    QUALIFYING      = "qualifying"
    DISCOVERY       = "discovery"
    DEMO_SCHEDULED  = "demo_scheduled"
    PROPOSAL        = "proposal"
    NEGOTIATION     = "negotiation"
    COMPLIANCE_HOLD = "compliance_hold"
    CLOSE_WON       = "close_won"
    CLOSE_LOST      = "close_lost"


class ComplianceFlag(str, Enum):
    GDPR          = "gdpr"
    HIPAA         = "hipaa"
    SOC2          = "soc2"
    FedRAMP       = "fedramp"
    SECURITY_REVIEW = "security_review"
    NONE          = "none"


class ActionType(str, Enum):
    CLASSIFY_LEAD      = "classify_lead"
    SET_PRIORITY       = "set_priority"
    ASSIGN_OWNER       = "assign_owner"
    REQUEST_MORE_INFO  = "request_more_info"
    ESCALATE_ENTERPRISE = "escalate_enterprise"
    NOTIFY_COMPLIANCE  = "notify_compliance"
    SCHEDULE_DEMO      = "schedule_demo"
    CLOSE_LOST         = "close_lost"
    CLOSE_WON          = "close_won"


# ---------------------------------------------------------------------------
# Core payload models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A single action the agent takes in the environment."""
    action_type: ActionType
    value: Optional[str] = Field(
        default=None,
        description="Specific value for the action (e.g. 'hot', 'sdr', 'critical')"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's free-text reasoning for this action (not used in grading)"
    )

    def to_log_str(self) -> str:
        parts = [self.action_type.value]
        if self.value:
            parts.append(f"value={self.value}")
        return ":".join(parts)


class Observation(BaseModel):
    """Full observable state of the lead / workflow at any step."""

    # --- lead profile ---
    lead_id: str
    lead_source: LeadSource
    company_name: str
    company_size: CompanySize
    region: Region
    industry: str
    contact_name: str
    contact_title: str

    # --- signals ---
    urgency_score: float = Field(ge=0.0, le=1.0, description="0=cold 1=very urgent")
    budget_confidence: float = Field(ge=0.0, le=1.0, description="0=unknown 1=confirmed")
    stakeholder_count: int = Field(ge=1, description="Number of known decision-makers")
    has_existing_contract: bool

    # --- requirements ---
    requested_integrations: List[str] = Field(default_factory=list)
    mentioned_competitors: List[str] = Field(default_factory=list)
    compliance_flags: List[ComplianceFlag] = Field(default_factory=list)

    # --- current workflow state ---
    current_stage: WorkflowStage
    current_classification: Optional[LeadClassification] = None
    current_priority: Optional[Priority] = None
    current_owner: OwnerRole = OwnerRole.NONE
    demo_scheduled: bool = False
    compliance_notified: bool = False
    more_info_requested: bool = False
    enterprise_escalated: bool = False

    # --- episode tracking ---
    step_number: int = 0
    previous_actions: List[str] = Field(default_factory=list)
    available_actions: List[ActionType] = Field(
        default_factory=lambda: list(ActionType)
    )

    # --- context notes (from CRM notes field) ---
    notes: str = ""


class StepResult(BaseModel):
    """Returned by env.step() after each action."""
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class EpisodeScore(BaseModel):
    """Final scoring breakdown returned at episode end."""
    task_id: str
    total_score: float = Field(ge=0.0, le=1.0)

    # Sub-dimension scores (all in [0,1])
    classification_score: float = 0.0
    routing_score: float = 0.0
    urgency_score: float = 0.0
    compliance_score: float = 0.0
    scheduling_score: float = 0.0
    closure_penalty: float = 0.0   # negative contribution absorbed into total

    steps_taken: int = 0
    efficiency_bonus: float = 0.0
    success: bool = False
    notes: str = ""


class TaskMetadata(BaseModel):
    """Metadata for a single task definition."""
    task_id: str
    name: str
    difficulty: str          # easy | medium | hard
    description: str
    expected_classification: LeadClassification
    expected_owner: OwnerRole
    expected_priority: Priority
    requires_compliance: bool
    requires_demo: bool
    requires_escalation: bool
    max_steps: int
    scoring_weights: Dict[str, float]
    tags: List[str] = Field(default_factory=list)


class TaskState(BaseModel):
    """Runtime mutable state for an active episode."""
    task_id: str
    observation: Observation
    step_count: int = 0
    cumulative_reward: float = 0.0
    reward_history: List[float] = Field(default_factory=list)
    done: bool = False
    success: bool = False
    repeated_action_counts: Dict[str, int] = Field(default_factory=dict)
    wrong_close_issued: bool = False
    late_escalation_penalty_applied: bool = False
