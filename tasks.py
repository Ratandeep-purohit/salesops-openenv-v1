"""
tasks.py — Deterministic task definitions for SalesOps OpenEnv.

Three tasks spanning easy → medium → hard with realistic enterprise lead data.
Every task is fully deterministic: same seed produces identical initial state.
"""

from __future__ import annotations
from models import (
    ActionType, CompanySize, ComplianceFlag, LeadClassification,
    LeadSource, Observation, OwnerRole, Priority, Region,
    TaskMetadata, WorkflowStage,
)


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _base_available_actions() -> list[ActionType]:
    return list(ActionType)


# ---------------------------------------------------------------------------
# Task 1 — EASY: Hot Pricing-Page Lead
# ---------------------------------------------------------------------------

TASK_EASY_META = TaskMetadata(
    task_id="task_easy_hot_lead",
    name="Hot Pricing-Page Lead Qualification",
    difficulty="easy",
    description=(
        "A SMB contact landed on the pricing page twice this week, "
        "filled out the inbound form, and explicitly asked for a trial. "
        "The agent must classify the lead as hot, assign an SDR, "
        "set medium-high priority, and schedule a discovery demo."
    ),
    expected_classification=LeadClassification.HOT,
    expected_owner=OwnerRole.SDR,
    expected_priority=Priority.HIGH,
    requires_compliance=False,
    requires_demo=True,
    requires_escalation=False,
    max_steps=10,
    scoring_weights={
        "classification": 0.30,
        "routing":        0.25,
        "urgency":        0.20,
        "scheduling":     0.25,
    },
    tags=["hot", "smb", "pricing_page", "trial_request"],
)

def _task_easy_initial_observation() -> Observation:
    return Observation(
        lead_id="LEAD-001-EASY",
        lead_source=LeadSource.PRICING_PAGE,
        company_name="Brightwave Solutions",
        company_size=CompanySize.SMB,
        region=Region.NORTH_AMERICA,
        industry="Marketing Technology",
        contact_name="Ashley Chen",
        contact_title="Head of Marketing",
        urgency_score=0.85,
        budget_confidence=0.70,
        stakeholder_count=1,
        has_existing_contract=False,
        requested_integrations=["Salesforce", "HubSpot"],
        mentioned_competitors=[],
        compliance_flags=[],
        current_stage=WorkflowStage.NEW,
        current_classification=None,
        current_priority=None,
        current_owner=OwnerRole.NONE,
        demo_scheduled=False,
        compliance_notified=False,
        more_info_requested=False,
        enterprise_escalated=False,
        step_number=0,
        previous_actions=[],
        available_actions=_base_available_actions(),
        notes=(
            "Ashley visited /pricing 3× in 5 days. Submitted 'Start my trial' form. "
            "Budget: ~$1 200/mo. No compliance complexity. Wants a live demo ASAP."
        ),
    )


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: Enterprise Growth with Custom Integration Request
# ---------------------------------------------------------------------------

TASK_MEDIUM_META = TaskMetadata(
    task_id="task_medium_enterprise_growth",
    name="Enterprise Growth — Custom Integration & Discovery",
    difficulty="medium",
    description=(
        "A mid-market SaaS company requests a custom API integration with "
        "their internal ERP and mentions a meaningful budget. "
        "The agent must classify as enterprise_growth, assign a Solution Engineer, "
        "confirm budget, and schedule a technical discovery call."
    ),
    expected_classification=LeadClassification.ENTERPRISE_GROWTH,
    expected_owner=OwnerRole.SOLUTION_ENGINEER,
    expected_priority=Priority.HIGH,
    requires_compliance=False,
    requires_demo=True,
    requires_escalation=False,
    max_steps=14,
    scoring_weights={
        "classification": 0.25,
        "routing":        0.30,
        "urgency":        0.15,
        "scheduling":     0.30,
    },
    tags=["enterprise_growth", "custom_integration", "mid_market", "discovery"],
)

def _task_medium_initial_observation() -> Observation:
    return Observation(
        lead_id="LEAD-002-MED",
        lead_source=LeadSource.REFERRAL,
        company_name="Nexora Systems",
        company_size=CompanySize.MID_MARKET,
        region=Region.EUROPE,
        industry="Enterprise SaaS",
        contact_name="Marcus Höfler",
        contact_title="VP of Operations",
        urgency_score=0.60,
        budget_confidence=0.65,
        stakeholder_count=3,
        has_existing_contract=False,
        requested_integrations=["SAP ERP", "Workday", "Custom REST API"],
        mentioned_competitors=["Salesforce", "Zoho"],
        compliance_flags=[ComplianceFlag.GDPR],
        current_stage=WorkflowStage.NEW,
        current_classification=None,
        current_priority=None,
        current_owner=OwnerRole.NONE,
        demo_scheduled=False,
        compliance_notified=False,
        more_info_requested=False,
        enterprise_escalated=False,
        step_number=0,
        previous_actions=[],
        available_actions=_base_available_actions(),
        notes=(
            "Marcus was referred by an existing customer. Three stakeholders involved. "
            "Budget mentioned: €80k/yr. Requires deep integration with SAP & Workday. "
            "Timeline: Q3 rollout. GDPR must be acknowledged. Competitor: Salesforce quote received."
        ),
    )


# ---------------------------------------------------------------------------
# Task 3 — HARD: Complex Enterprise Migration with Legal & Compliance Risk
# ---------------------------------------------------------------------------

TASK_HARD_META = TaskMetadata(
    task_id="task_hard_enterprise_migration",
    name="High-Stakes Enterprise Migration with Compliance & Legal Risk",
    difficulty="hard",
    description=(
        "A global enterprise (2 000+ employees) is migrating away from a legacy "
        "platform under a tight legal deadline. Multiple stakeholders, unclear "
        "budget authority, active security review, HIPAA + SOC2 obligations, "
        "and a C-suite sponsor who is about to leave. "
        "The agent must navigate: classify correctly, escalate to Enterprise AE, "
        "notify compliance, request clarification on budget/stakeholders, "
        "and schedule only after compliance clearance — in the right order."
    ),
    expected_classification=LeadClassification.ENTERPRISE_GROWTH,
    expected_owner=OwnerRole.ENTERPRISE_AE,
    expected_priority=Priority.CRITICAL,
    requires_compliance=True,
    requires_demo=True,
    requires_escalation=True,
    max_steps=20,
    scoring_weights={
        "classification": 0.20,
        "routing":        0.20,
        "urgency":        0.15,
        "compliance":     0.25,
        "scheduling":     0.20,
    },
    tags=[
        "enterprise", "migration", "hipaa", "soc2", "legal", "multi_stakeholder",
        "urgent_timeline", "security_review", "hard",
    ],
)

def _task_hard_initial_observation() -> Observation:
    return Observation(
        lead_id="LEAD-003-HARD",
        lead_source=LeadSource.CONFERENCE,
        company_name="Meridian Health Systems",
        company_size=CompanySize.ENTERPRISE,
        region=Region.NORTH_AMERICA,
        industry="Healthcare & Life Sciences",
        contact_name="Dr. Patricia Wen",
        contact_title="CTO",
        urgency_score=0.95,
        budget_confidence=0.30,
        stakeholder_count=7,
        has_existing_contract=True,
        requested_integrations=[
            "Epic EHR",
            "Azure AD",
            "On-prem Data Warehouse",
            "Custom FHIR API",
        ],
        mentioned_competitors=["Veeva", "Oracle Health"],
        compliance_flags=[
            ComplianceFlag.HIPAA,
            ComplianceFlag.SOC2,
            ComplianceFlag.SECURITY_REVIEW,
            ComplianceFlag.FedRAMP,
        ],
        current_stage=WorkflowStage.NEW,
        current_classification=None,
        current_priority=None,
        current_owner=OwnerRole.NONE,
        demo_scheduled=False,
        compliance_notified=False,
        more_info_requested=False,
        enterprise_escalated=False,
        step_number=0,
        previous_actions=[],
        available_actions=_base_available_actions(),
        notes=(
            "Meridian is a 3 200-employee health system migrating off legacy Veeva in 90 days. "
            "C-suite sponsor (CFO) leaving in 6 weeks — deal must progress now. "
            "7 stakeholders across IT, Legal, Finance, Clinical Ops. "
            "Budget owner unclear — IT says $400k, Finance says $200k. "
            "Active HIPAA audit underway. SOC2 Type II report requested. "
            "Security team requires pen-test results before any demo. "
            "FedRAMP moderate authorization required for government-adjacent workloads. "
            "Legal flagged PII data residency in CONUS only."
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict] = {
    TASK_EASY_META.task_id: {
        "meta":    TASK_EASY_META,
        "init_fn": _task_easy_initial_observation,
    },
    TASK_MEDIUM_META.task_id: {
        "meta":    TASK_MEDIUM_META,
        "init_fn": _task_medium_initial_observation,
    },
    TASK_HARD_META.task_id: {
        "meta":    TASK_HARD_META,
        "init_fn": _task_hard_initial_observation,
    },
}


def get_task_ids() -> list[str]:
    return list(TASK_REGISTRY.keys())


def get_task_meta(task_id: str) -> TaskMetadata:
    return TASK_REGISTRY[task_id]["meta"]


def get_initial_observation(task_id: str) -> Observation:
    return TASK_REGISTRY[task_id]["init_fn"]()
