from app.models import AgentState, ToolResult
from app.decision_agent import make_decision
from app.tools import execute_tool
from app.validator import validate_tool_result


def test_validate_auto_refund():
    query = "I want a refund of $50"

    state = AgentState(user_query=query)
    state.decision = make_decision(query)
    state.tool_result = execute_tool(state.decision, query)

    state = validate_tool_result(state)

    assert state.validation_result.status == "valid"
    assert state.validation_result.is_valid is True
    assert state.approval_status == "not_required"


def test_validate_refund_requires_human_approval():
    query = "I want a refund of $700"

    state = AgentState(user_query=query)
    state.decision = make_decision(query)
    state.tool_result = execute_tool(state.decision, query)

    state = validate_tool_result(state)

    assert state.validation_result.status == "needs_human_approval"
    assert state.validation_result.is_valid is True
    assert state.validation_result.requires_human_action is True
    assert state.approval_status == "required"


def test_validate_no_tool_required():
    query = "How do I reset my password?"

    state = AgentState(user_query=query)
    state.decision = make_decision(query)
    state.tool_result = execute_tool(state.decision, query)

    state = validate_tool_result(state)

    assert state.validation_result.status == "not_required"
    assert state.validation_result.is_valid is True


def test_validate_tool_failure():
    state = AgentState(user_query="Where is my refund?")

    state.tool_result = ToolResult(
        tool_name="check_refund_status",
        status="failed",
        error_message="Payment system timeout."
    )

    state = validate_tool_result(state)

    assert state.validation_result.status == "tool_failed"
    assert state.validation_result.is_valid is False
    assert "Payment system timeout." in state.errors


def test_validate_subscription_needs_confirmation():
    query = "Cancel my monthly plan"

    state = AgentState(user_query=query)
    state.decision = make_decision(query)
    state.tool_result = execute_tool(state.decision, query)

    state = validate_tool_result(state)

    assert state.validation_result.status == "needs_user_confirmation"
    assert state.validation_result.requires_user_confirmation is True