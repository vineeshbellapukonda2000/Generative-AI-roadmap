from app.models import AgentState, ValidationResult


def validate_tool_result(state: AgentState) -> AgentState:
    if state.tool_result is None:
        message = "Tool result is missing."

        state.validation_result = ValidationResult(
            status="invalid",
            is_valid=False,
            message=message
        )

        state.errors.append(message)
        return state

    tool_result = state.tool_result

    if tool_result.status == "not_called":
        state.validation_result = ValidationResult(
            status="not_required",
            is_valid=True,
            message="No tool was required for this request."
        )
        return state

    if tool_result.status == "failed":
        message = tool_result.error_message or "Tool execution failed."

        state.validation_result = ValidationResult(
            status="tool_failed",
            is_valid=False,
            message=message
        )

        state.errors.append(message)
        return state

    data = tool_result.data or {}

    if tool_result.tool_name == "check_refund_eligibility":
        if data.get("requires_human_approval") is True:
            state.approval_status = "required"

            state.validation_result = ValidationResult(
                status="needs_human_approval",
                is_valid=True,
                message="Refund is valid for review, but human approval is required before processing.",
                requires_human_action=True
            )
            return state

        if data.get("eligible_for_auto_refund") is True:
            state.approval_status = "not_required"

            state.validation_result = ValidationResult(
                status="valid",
                is_valid=True,
                message="Refund is eligible for automatic processing."
            )
            return state

        message = "Refund eligibility could not be confirmed."

        state.validation_result = ValidationResult(
            status="invalid",
            is_valid=False,
            message=message
        )

        state.errors.append(message)
        return state

    if tool_result.tool_name == "cancel_subscription":
        if data.get("cancellation_status") == "confirmation_required":
            state.validation_result = ValidationResult(
                status="needs_user_confirmation",
                is_valid=True,
                message="User confirmation is required before cancelling the subscription.",
                requires_user_confirmation=True
            )
            return state

    state.validation_result = ValidationResult(
        status="valid",
        is_valid=True,
        message="Tool result validated successfully."
    )

    return state