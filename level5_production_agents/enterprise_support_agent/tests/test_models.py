from app.models import AgentDecision, ToolResult, AgentState


def test_agent_decision_model():
    decision = AgentDecision(
        intent="refund_request",
        requires_tool=True,
        tool_name="check_refund_eligibility",
        priority="high",
        reason="User requested a refund."
    )

    assert decision.intent == "refund_request"
    assert decision.requires_tool is True
    assert decision.tool_name == "check_refund_eligibility"


def test_tool_result_model():
    result = ToolResult(
        tool_name="check_refund_eligibility",
        status="success",
        data={
            "eligible": True,
            "amount": 50
        }
    )

    assert result.status == "success"
    assert result.data["eligible"] is True


def test_agent_state_model():
    state = AgentState(
        user_query="Refund my latest payment."
    )

    assert state.user_query == "Refund my latest payment."
    assert state.decision is None
    assert state.tool_result is None
    assert state.approval_status == "not_required"