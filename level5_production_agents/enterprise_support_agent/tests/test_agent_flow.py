from app.decision_agent import make_decision


def test_refund_request_decision():
    decision = make_decision("I want a refund for yesterday's charge")

    assert decision.intent == "refund_request"
    assert decision.requires_tool is True
    assert decision.tool_name == "check_refund_eligibility"
    assert decision.priority == "high"


def test_refund_status_decision():
    decision = make_decision("Where is my refund?")

    assert decision.intent == "refund_status"
    assert decision.requires_tool is True
    assert decision.tool_name == "check_refund_status"


def test_billing_issue_decision():
    decision = make_decision("I was charged twice this month")

    assert decision.intent == "billing_issue"
    assert decision.requires_tool is True
    assert decision.tool_name == "check_payment_history"


def test_subscription_cancellation_decision():
    decision = make_decision("Cancel my monthly plan")

    assert decision.intent == "subscription_cancellation"
    assert decision.requires_tool is True
    assert decision.tool_name == "cancel_subscription"


def test_general_question_decision():
    decision = make_decision("How do I reset my password?")

    assert decision.intent == "general_question"
    assert decision.requires_tool is False
    assert decision.tool_name == "none"


def test_unknown_decision():
    decision = make_decision("I need help with something random")

    assert decision.intent == "unknown"
    assert decision.requires_tool is False
    assert decision.tool_name == "none"