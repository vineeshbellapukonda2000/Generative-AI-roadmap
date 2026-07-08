from app.decision_agent import make_decision
from app.tools import execute_tool


def test_execute_refund_eligibility_tool():
    query = "I want a refund of $50"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "check_refund_eligibility"
    assert result.status == "success"
    assert result.data["eligible_for_auto_refund"] is True
    assert result.data["requires_human_approval"] is False


def test_refund_above_100_requires_human_approval():
    query = "I want a refund of $700"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "check_refund_eligibility"
    assert result.status == "success"
    assert result.data["refund_amount"] == 700
    assert result.data["requires_human_approval"] is True
    assert result.data["eligible_for_auto_refund"] is False


def test_execute_refund_status_tool():
    query = "Where is my refund?"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "check_refund_status"
    assert result.status == "success"
    assert result.data["refund_status"] == "processing"


def test_execute_payment_history_tool():
    query = "I was charged twice this month"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "check_payment_history"
    assert result.status == "success"
    assert result.data["duplicate_charge_found"] is True


def test_no_tool_required():
    query = "How do I reset my password?"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "none"
    assert result.status == "not_called"


def test_create_support_ticket_tool():
    query = "I want to speak to someone"

    decision = make_decision(query)
    result = execute_tool(decision, query)

    assert result.tool_name == "create_support_ticket"
    assert result.status == "success"
    assert result.data["ticket_created"] is True