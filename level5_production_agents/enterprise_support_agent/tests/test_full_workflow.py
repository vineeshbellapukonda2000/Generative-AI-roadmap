from app.main import run_agent_workflow


def test_full_workflow_auto_refund():
    state = run_agent_workflow("I want a refund of $50")

    assert state.decision.intent == "refund_request"
    assert state.tool_result.tool_name == "check_refund_eligibility"
    assert state.validation_result.status == "valid"
    assert state.final_response is not None
    assert "eligible for automatic processing" in state.final_response


def test_full_workflow_refund_requires_human_approval():
    state = run_agent_workflow("I want a refund of $700")

    assert state.decision.intent == "refund_request"
    assert state.tool_result.data["refund_amount"] == 700
    assert state.validation_result.status == "needs_human_approval"
    assert state.approval_status == "required"
    assert "human approval" in state.final_response.lower()


def test_full_workflow_refund_status():
    state = run_agent_workflow("Where is my refund?")

    assert state.decision.intent == "refund_status"
    assert state.tool_result.tool_name == "check_refund_status"
    assert state.validation_result.status == "valid"
    assert "processing" in state.final_response.lower()


def test_full_workflow_subscription_confirmation():
    state = run_agent_workflow("Cancel my monthly plan")

    assert state.decision.intent == "subscription_cancellation"
    assert state.tool_result.tool_name == "cancel_subscription"
    assert state.validation_result.status == "needs_user_confirmation"
    assert "confirm" in state.final_response.lower()


def test_full_workflow_password_reset():
    state = run_agent_workflow("How do I reset my password?")

    assert state.decision.intent == "general_question"
    assert state.tool_result.status == "not_called"
    assert state.validation_result.status == "not_required"
    assert "reset your password" in state.final_response.lower()