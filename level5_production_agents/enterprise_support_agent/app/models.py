from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field


IntentType = Literal[
    "general_question",
    "billing_issue",
    "refund_request",
    "refund_status",
    "subscription_cancellation",
    "support_escalation",
    "unknown"
]


ToolName = Literal[
    "check_payment_history",
    "check_refund_eligibility",
    "check_refund_status",
    "cancel_subscription",
    "create_support_ticket",
    "none"
]


Priority = Literal[
    "low",
    "medium",
    "high"
]


ToolStatus = Literal[
    "not_called",
    "success",
    "failed"
]


ApprovalStatus = Literal[
    "not_required",
    "required",
    "approved",
    "rejected",
    "pending"
]


ValidationStatus = Literal[
    "valid",
    "invalid",
    "tool_failed",
    "not_required",
    "needs_human_approval",
    "needs_user_confirmation"
]


class AgentDecision(BaseModel):
    intent: IntentType
    requires_tool: bool
    tool_name: ToolName = "none"
    priority: Priority = "medium"
    reason: str


class ToolResult(BaseModel):
    tool_name: ToolName
    status: ToolStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ValidationResult(BaseModel):
    status: ValidationStatus
    is_valid: bool
    message: str
    requires_human_action: bool = False
    requires_user_confirmation: bool = False


class AgentState(BaseModel):
    user_query: str

    decision: Optional[AgentDecision] = None
    tool_result: Optional[ToolResult] = None
    validation_result: Optional[ValidationResult] = None

    approval_status: ApprovalStatus = "not_required"

    final_response: Optional[str] = None
    errors: List[str] = Field(default_factory=list)