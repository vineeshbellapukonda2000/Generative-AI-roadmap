from app.models import AgentDecision


def make_decision(user_query: str) -> AgentDecision:
    query = user_query.lower()

    if "refund status" in query or "where is my refund" in query:
        return AgentDecision(
            intent="refund_status",
            requires_tool=True,
            tool_name="check_refund_status",
            priority="medium",
            reason="User is asking about the status of an existing refund."
        )

    if "refund" in query:
        return AgentDecision(
            intent="refund_request",
            requires_tool=True,
            tool_name="check_refund_eligibility",
            priority="high",
            reason="User is requesting a refund, so eligibility must be checked first."
        )

    if "charged twice" in query or "double charged" in query or "payment issue" in query or "billing" in query:
        return AgentDecision(
            intent="billing_issue",
            requires_tool=True,
            tool_name="check_payment_history",
            priority="high",
            reason="User reported a billing or payment issue, so payment history must be checked."
        )

    if "cancel" in query and ("subscription" in query or "plan" in query):
        return AgentDecision(
            intent="subscription_cancellation",
            requires_tool=True,
            tool_name="cancel_subscription",
            priority="high",
            reason="User wants to cancel a subscription or plan."
        )

    if "speak to someone" in query or "human" in query or "support agent" in query:
        return AgentDecision(
            intent="support_escalation",
            requires_tool=True,
            tool_name="create_support_ticket",
            priority="medium",
            reason="User wants help from a support person."
        )

    if "password" in query or "reset" in query:
        return AgentDecision(
            intent="general_question",
            requires_tool=False,
            tool_name="none",
            priority="low",
            reason="User is asking a general help question that can be answered without a tool."
        )

    return AgentDecision(
        intent="unknown",
        requires_tool=False,
        tool_name="none",
        priority="low",
        reason="The system could not confidently classify the user request."
    )