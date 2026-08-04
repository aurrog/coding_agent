from agent.llm import LLMClient
from agent.prompts import build_initial_messages
from config import AgentContext, AgentSettings
from core.models import AgentResult
from tools.registry import ToolRegistry


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        settings: AgentSettings,
        context: AgentContext,
    ):
        self._llm = llm
        self._tools = tools
        self._settings = settings
        self._context = context

    def run(self, user_request: str) -> AgentResult:
        if not user_request.strip():
            return AgentResult.failed(
                iterations=0,
                tool_calls_count=0,
                error="User request must not be empty",
            )

        messages = build_initial_messages(
            user_request,
            self._context.workspace_root,
            self._context.permission_mode,
        )
        total_tool_calls = 0

        for iteration in range(self._settings.max_iterations):
            if self._context_size(messages) > (
                self._settings.max_context_characters
            ):
                return AgentResult.incomplete(
                    iterations=iteration,
                    tool_calls_count=total_tool_calls,
                    reason="CONTEXT_LIMIT_REACHED",
                )

            try:
                response = self._llm.complete(
                    messages=messages,
                    tools=self._tools.schemas(),
                )
            except Exception as exc:
                return AgentResult.failed(
                    iterations=iteration + 1,
                    tool_calls_count=total_tool_calls,
                    error=f"LLM_REQUEST_FAILED: {type(exc).__name__}",
                )

            messages.append(response.to_assistant_message())

            if response.tool_calls:
                for call in response.tool_calls:
                    if total_tool_calls >= (
                        self._settings.max_tool_calls
                    ):
                        return AgentResult.incomplete(
                            iterations=iteration + 1,
                            tool_calls_count=total_tool_calls,
                            reason="TOOL_CALL_LIMIT_REACHED",
                        )

                    result = self._tools.execute(call)
                    messages.append(result.to_message())
                    total_tool_calls += 1
                continue

            if response.content and response.content.strip():
                return AgentResult.completed(
                    final_text=response.content,
                    iterations=iteration + 1,
                    tool_calls_count=total_tool_calls,
                )

            return AgentResult.failed(
                iterations=iteration + 1,
                tool_calls_count=total_tool_calls,
                error="MODEL_RETURNED_EMPTY_RESPONSE",
            )

        return AgentResult.incomplete(
            iterations=self._settings.max_iterations,
            tool_calls_count=total_tool_calls,
            reason="ITERATION_LIMIT_REACHED",
        )

    @staticmethod
    def _context_size(messages: list[dict]) -> int:
        return sum(len(str(message)) for message in messages)
