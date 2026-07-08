import re
from typing import Callable, Dict

from app.models import AgentDecision, ToolResult


def extract_amount(user_query: str) -> float:
    match = re.search(r"\$?(\d+(?:\.\d{1,2})?)", user_query)

    if match:
        return float(match.group(1))

    return 50.0


def check_payment_history(user_query: str) -> ToolResult:
    query = user_query.lower()

    duplicate_charge_found = (
        "charged twice" in query
        or "double charged" in query
        or "duplicate charge" in query
    )

    return ToolResult(
        tool_name="check_payment_history",
        status="success",
        data={
            "duplicate_charge_found": duplicate_charge_found,
            "recent_charges": [
                {"date": "2026-06-01", "amount": 19.99},
                {"date": "2026-06-01", "amount": 19.99},
            ] if duplicate_charge_found else [
                {"date": "2026-06-01", "amount": 19.99}
            ]
        }
    )


def check_refund_eligibility(user_query: str) -> ToolResult:
    amount = extract_amount(user_query)

    requires_human_approval = amount > 100
    eligible_for_auto_refund = amount <= 100

    return ToolResult(
        tool_name="check_refund_eligibility",
        status="success",
        data={
            "refund_amount": amount,
            "eligible_for_auto_refund": eligible_for_auto_refund,
            "requires_human_approval": requires_human_approval,
            "policy": "Refunds above $100 require human approval."
        }
    )


def check_refund_status(user_query: str) -> ToolResult:
    return ToolResult(
        tool_name="check_refund_status",
        status="success",
        data={
            "refund_status": "processing",
            "estimated_completion": "3-5 business days",
            "message": "Refund has been initiated and is currently processing."
        }
    )


def cancel_subscription(user_query: str) -> ToolResult:
    return ToolResult(
        tool_name="cancel_subscription",
        status="success",
        data={
            "cancellation_status": "confirmation_required",
            "message": "User confirmation is required before cancelling the subscription."
        }
    )


def create_support_ticket(user_query: str) -> ToolResult:
    return ToolResult(
        tool_name="create_support_ticket",
        status="success",
        data={
            "ticket_created": True,
            "ticket_id": "SUP-1001",
            "message": "Support ticket created successfully."
        }
    )


TOOL_REGISTRY: Dict[str, Callable[[str], ToolResult]] = {
    "check_payment_history": check_payment_history,
    "check_refund_eligibility": check_refund_eligibility,
    "check_refund_status": check_refund_status,
    "cancel_subscription": cancel_subscription,
    "create_support_ticket": create_support_ticket,
}


def execute_tool(decision: AgentDecision, user_query: str) -> ToolResult:
    if not decision.requires_tool:
        return ToolResult(
            tool_name="none",
            status="not_called",
            data={
                "message": "No tool was required for this request."
            }
        )

    tool_function = TOOL_REGISTRY.get(decision.tool_name)

    if tool_function is None:
        return ToolResult(
            tool_name=decision.tool_name,
            status="failed",
            error_message=f"Tool '{decision.tool_name}' is not registered."
        )

    return tool_function(user_query)