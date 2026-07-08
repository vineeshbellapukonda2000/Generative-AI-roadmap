## Production Agent Workflow Foundation

This project implements the core foundation of a production-style agentic AI workflow.

Instead of sending a user query directly to an LLM, the system follows a controlled workflow:

1. Stores the user request in AgentState
2. Uses a Decision Agent to classify intent and decide whether a tool is needed
3. Executes only approved backend tools through a Tool Executor
4. Stores tool output as ToolResult
5. Validates the tool result using a Validation Layer
6. Stores validation output as ValidationResult
7. Generates a final response based on validated state

This design improves reliability, traceability, and safety compared to a normal chatbot.



## Architecture

- AgentState: Stores the full workflow context
- AgentDecision: Stores intent, tool requirement, selected tool, priority, and reason
- Decision Agent: Decides what should happen next
- Tool Executor: Executes approved tools safely
- TOOL_REGISTRY: Maintains allowed backend tools
- ToolResult: Stores raw output from tool execution
- Validation Layer: Checks tool result and decides next safe action
- ValidationResult: Stores validation status and message
- Final Response Generator: Creates the final user-facing answer


## Example Flow

User query:

"I want a refund of $700"

Workflow:

1. Decision Agent identifies intent as refund_request
2. AgentDecision selects check_refund_eligibility
3. Tool Executor checks TOOL_REGISTRY
4. Approved refund eligibility tool is executed
5. ToolResult returns refund_amount = 700 and requires_human_approval = True
6. Validation Layer marks the request as needs_human_approval
7. Final response tells the user that human approval is required before processing