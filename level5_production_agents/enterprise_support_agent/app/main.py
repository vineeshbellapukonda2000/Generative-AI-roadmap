from app.models import AgentState
from app.decision_agent import make_decision
from app.tools import execute_tool
from app.validator import validate_tool_result


def generate_final_response(state: AgentState) -> str:
    if state.validation_result is None:
        return "I could not validate the result, so I cannot safely complete this request."

    validation_status = state.validation_result.status
    tool_result = state.tool_result
    decision = state.decision

    if validation_status == "tool_failed":
        return (
            "I could not complete this request because the required system tool failed. "
            "Please try again later or contact support."
        )

    if validation_status == "not_required":
        if decision and decision.intent == "general_question":
            return (
                "You can reset your password by going to account settings, selecting "
                "'Forgot Password' or 'Reset Password', and following the verification steps."
            )

        return "No backend tool was required for this request."

    if validation_status == "needs_human_approval":
        refund_amount = tool_result.data.get("refund_amount") if tool_result and tool_result.data else None

        return (
            f"Your refund request for ${refund_amount} is valid for review, "
            "but it requires human approval before it can be processed."
        )

    if validation_status == "needs_user_confirmation":
        return (
            "Your subscription cancellation request requires confirmation before we continue. "
            "Please confirm if you want to cancel your subscription."
        )

    if tool_result and tool_result.tool_name == "check_refund_eligibility":
        refund_amount = tool_result.data.get("refund_amount")

        return (
            f"Your refund request for ${refund_amount} is eligible for automatic processing. "
            "No human approval is required."
        )

    if tool_result and tool_result.tool_name == "check_refund_status":
        refund_status = tool_result.data.get("refund_status")
        estimated_completion = tool_result.data.get("estimated_completion")

        return (
            f"Your refund is currently {refund_status}. "
            f"The estimated completion time is {estimated_completion}."
        )

    if tool_result and tool_result.tool_name == "check_payment_history":
        duplicate_found = tool_result.data.get("duplicate_charge_found")

        if duplicate_found:
            return (
                "I found a possible duplicate charge in your recent payment history. "
                "This should be reviewed by the billing team."
            )

        return "I checked your payment history and did not find a duplicate charge."

    if tool_result and tool_result.tool_name == "create_support_ticket":
        ticket_id = tool_result.data.get("ticket_id")

        return (
            f"A support ticket has been created successfully. "
            f"Your ticket ID is {ticket_id}."
        )

    return "The request was completed and validated successfully."


def run_agent_workflow(user_query: str) -> AgentState:
    state = AgentState(user_query=user_query)

    state.decision = make_decision(user_query)

    state.tool_result = execute_tool(
        decision=state.decision,
        user_query=user_query
    )

    state = validate_tool_result(state)

    state.final_response = generate_final_response(state)

    return state


if __name__ == "__main__":
    sample_queries = [
        "I want a refund of $50",
        "I want a refund of $700",
        "Where is my refund?",
        "I was charged twice this month",
        "Cancel my monthly plan",
        "How do I reset my password?"
    ]

    for query in sample_queries:
        result = run_agent_workflow(query)

        print("\nUser Query:", result.user_query)
        print("Intent:", result.decision.intent)
        print("Tool:", result.decision.tool_name)
        print("Validation:", result.validation_result.status)
        print("Final Response:", result.final_response)