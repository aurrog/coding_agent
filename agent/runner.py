import json
import logging
from time import perf_counter
from uuid import uuid4

from agent.approval import ApprovalProvider, DenyAllApprovalProvider
from agent.llm import LLMClient
from agent.observability import (
    NullProgressReporter,
    ProgressReporter,
)
from agent.prompts import build_initial_messages
from config import AgentContext, AgentSettings
from core.models import (
    AgentResult,
    ApprovalDecision,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from tools.registry import ToolRegistry


logger = logging.getLogger("coding_agent.runner")


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        settings: AgentSettings,
        context: AgentContext,
        progress: ProgressReporter | None = None,
        approval: ApprovalProvider | None = None,
    ):
        self._llm = llm
        self._tools = tools
        self._settings = settings
        self._context = context
        self._progress = progress or NullProgressReporter()
        self._approval = approval or DenyAllApprovalProvider()

    def run(self, user_request: str) -> AgentResult:
        run_id = uuid4().hex[:12]
        total_usage = TokenUsage()
        if not user_request.strip():
            logger.warning("run=%s rejected_empty_request", run_id)
            return AgentResult.failed(
                iterations=0,
                tool_calls_count=0,
                error="User request must not be empty",
                usage=total_usage,
            )

        logger.info(
            "run=%s started permission_mode=%s",
            run_id,
            self._context.permission_mode,
        )

        messages = build_initial_messages(
            user_request,
            self._context.workspace_root,
            self._context.permission_mode,
        )
        total_tool_calls = 0
        denied_fingerprints: set[str] = set()

        for iteration in range(self._settings.max_iterations):
            if self._context_size(messages) > (
                self._settings.max_context_characters
            ):
                logger.warning(
                    "run=%s stopped reason=CONTEXT_LIMIT_REACHED "
                    "iterations=%d tool_calls=%d total_tokens=%d",
                    run_id,
                    iteration,
                    total_tool_calls,
                    total_usage.total_tokens,
                )
                return AgentResult.incomplete(
                    iterations=iteration,
                    tool_calls_count=total_tool_calls,
                    reason="CONTEXT_LIMIT_REACHED",
                    usage=total_usage,
                )

            current_iteration = iteration + 1
            self._progress.model_request(
                current_iteration,
                self._settings.max_iterations,
            )
            request_started = perf_counter()
            try:
                response = self._llm.complete(
                    messages=messages,
                    tools=self._tools.schemas(),
                )
            except Exception as exc:
                logger.exception(
                    "run=%s iteration=%d llm_request_failed duration=%.3f",
                    run_id,
                    current_iteration,
                    perf_counter() - request_started,
                )
                return AgentResult.failed(
                    iterations=current_iteration,
                    tool_calls_count=total_tool_calls,
                    error=f"LLM_REQUEST_FAILED: {type(exc).__name__}",
                    usage=total_usage,
                )

            request_duration = perf_counter() - request_started
            total_usage = total_usage + response.usage
            self._progress.model_response(response.usage, total_usage)
            logger.info(
                "run=%s iteration=%d llm_response duration=%.3f "
                "finish_reason=%s tool_calls=%d prompt_tokens=%d "
                "completion_tokens=%d "
                "total_tokens=%d usage_reported=%s",
                run_id,
                current_iteration,
                request_duration,
                response.finish_reason or "unknown",
                len(response.tool_calls),
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                response.usage.reported,
            )
            messages.append(
                response.to_assistant_message(
                    compact_tool_arguments=True,
                )
            )

            if response.tool_calls:
                for call in response.tool_calls:
                    if total_tool_calls >= (
                        self._settings.max_tool_calls
                    ):
                        logger.warning(
                            "run=%s stopped reason=TOOL_CALL_LIMIT_REACHED "
                            "iterations=%d tool_calls=%d total_tokens=%d",
                            run_id,
                            current_iteration,
                            total_tool_calls,
                            total_usage.total_tokens,
                        )
                        return AgentResult.incomplete(
                            iterations=current_iteration,
                            tool_calls_count=total_tool_calls,
                            reason="TOOL_CALL_LIMIT_REACHED",
                            usage=total_usage,
                        )

                    target = self._tool_target(call)
                    result, tool_duration = self._execute_tool_call(
                        call,
                        target=target,
                        denied_fingerprints=denied_fingerprints,
                        run_id=run_id,
                    )
                    error_code = (
                        result.error.code
                        if result.error is not None
                        else None
                    )
                    self._progress.tool_finished(
                        call.name,
                        ok=result.ok,
                        error_code=error_code,
                        elapsed_seconds=tool_duration,
                    )
                    logger.info(
                        "run=%s tool=%s call_id=%s ok=%s "
                        "error_code=%s duration=%.3f",
                        run_id,
                        call.name,
                        call.id,
                        result.ok,
                        error_code or "none",
                        tool_duration,
                    )
                    messages.append(result.to_message())
                    if (
                        result.ok
                        and result.data is not None
                        and call.name in {
                            "create_file",
                            "edit_file",
                            "write_file",
                        }
                    ):
                        compacted_reads = self._compact_stale_reads(
                            messages,
                            result.data.get("path"),
                        )
                        if compacted_reads:
                            logger.debug(
                                "run=%s tool=%s compacted_stale_reads=%d",
                                run_id,
                                call.name,
                                compacted_reads,
                            )
                    total_tool_calls += 1
                continue

            if response.content and response.content.strip():
                logger.info(
                    "run=%s completed iterations=%d tool_calls=%d "
                    "total_tokens=%d",
                    run_id,
                    current_iteration,
                    total_tool_calls,
                    total_usage.total_tokens,
                )
                return AgentResult.completed(
                    final_text=response.content,
                    iterations=current_iteration,
                    tool_calls_count=total_tool_calls,
                    usage=total_usage,
                )

            logger.error(
                "run=%s stopped reason=MODEL_RETURNED_EMPTY_RESPONSE "
                "iterations=%d tool_calls=%d total_tokens=%d",
                run_id,
                current_iteration,
                total_tool_calls,
                total_usage.total_tokens,
            )
            return AgentResult.failed(
                iterations=current_iteration,
                tool_calls_count=total_tool_calls,
                error="MODEL_RETURNED_EMPTY_RESPONSE",
                usage=total_usage,
            )

        return self._finalize_after_iteration_limit(
            messages=messages,
            run_id=run_id,
            total_tool_calls=total_tool_calls,
            total_usage=total_usage,
        )

    def _execute_tool_call(
        self,
        call: ToolCall,
        *,
        target: str | None,
        denied_fingerprints: set[str],
        run_id: str,
    ) -> tuple[ToolResult, float]:
        prepared = self._tools.prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared, 0.0

        approved: bool | None = None
        if prepared.requires_approval:
            if prepared.fingerprint in denied_fingerprints:
                logger.info(
                    "run=%s tool=%s approval_reused_denial "
                    "fingerprint=%s",
                    run_id,
                    prepared.tool_name,
                    prepared.fingerprint,
                )
                approved = False
            else:
                request = prepared.approval_request
                if request is None:
                    logger.error(
                        "run=%s tool=%s approval_request_missing",
                        run_id,
                        prepared.tool_name,
                    )
                    return ToolResult.failure(
                        tool_call_id=prepared.tool_call_id,
                        code="TOOL_PREPARATION_ERROR",
                        message="Tool approval request is missing",
                    ), 0.0

                logger.info(
                    "run=%s tool=%s approval_requested fingerprint=%s",
                    run_id,
                    prepared.tool_name,
                    prepared.fingerprint,
                )
                try:
                    decision = self._approval.request(request)
                except Exception:
                    logger.exception(
                        "run=%s tool=%s approval_provider_failed",
                        run_id,
                        prepared.tool_name,
                    )
                    decision = ApprovalDecision.DENIED
                approved = decision == ApprovalDecision.APPROVED
                logger.info(
                    "run=%s tool=%s approval_decision=%s fingerprint=%s",
                    run_id,
                    prepared.tool_name,
                    "approved" if approved else "denied",
                    prepared.fingerprint,
                )
                if not approved:
                    denied_fingerprints.add(prepared.fingerprint)

        if not prepared.requires_approval or approved:
            self._progress.tool_started(call.name, target)
        tool_started = perf_counter()
        result = self._tools.execute_prepared(
            prepared,
            approved=approved,
        )
        return result, perf_counter() - tool_started

    @staticmethod
    def _context_size(messages: list[dict]) -> int:
        return sum(len(str(message)) for message in messages)

    @staticmethod
    def _tool_target(call: ToolCall) -> str | None:
        if not isinstance(call.arguments, dict):
            return None
        path = call.arguments.get("path")
        if not isinstance(path, str):
            return None
        return path[:200]

    @staticmethod
    def _compact_stale_reads(
        messages: list[dict],
        changed_path: object,
    ) -> int:
        if not isinstance(changed_path, str):
            return 0

        compacted = 0
        for message in messages:
            if message.get("role") != "tool":
                continue
            raw_content = message.get("content")
            if not isinstance(raw_content, str):
                continue
            try:
                payload = json.loads(raw_content)
            except json.JSONDecodeError:
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("path") != changed_path or "content" not in data:
                continue
            content = data.get("content")
            if not isinstance(content, str) or content.startswith("<compacted"):
                continue
            data["content"] = (
                "<compacted stale file content after successful change>"
            )
            message["content"] = json.dumps(payload, ensure_ascii=False)
            compacted += 1
        return compacted

    def _finalize_after_iteration_limit(
        self,
        *,
        messages: list[dict],
        run_id: str,
        total_tool_calls: int,
        total_usage: TokenUsage,
    ) -> AgentResult:
        final_iteration = self._settings.max_iterations + 1
        messages.append(
            {
                "role": "system",
                "content": (
                    "The tool-use iteration budget is exhausted. Do not "
                    "request more tools. Return a concise final answer now. "
                    "State clearly if any requested work remains incomplete."
                ),
            }
        )
        self._progress.model_request(
            final_iteration,
            self._settings.max_iterations,
            final=True,
        )
        request_started = perf_counter()
        try:
            response = self._llm.complete(messages=messages, tools=[])
        except Exception as exc:
            logger.exception(
                "run=%s finalization_failed duration=%.3f",
                run_id,
                perf_counter() - request_started,
            )
            return AgentResult.failed(
                iterations=final_iteration,
                tool_calls_count=total_tool_calls,
                error=(
                    "FINALIZATION_LLM_REQUEST_FAILED: "
                    f"{type(exc).__name__}"
                ),
                usage=total_usage,
            )

        total_usage = total_usage + response.usage
        self._progress.model_response(response.usage, total_usage)
        logger.info(
            "run=%s finalization_response duration=%.3f "
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d",
            run_id,
            perf_counter() - request_started,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )

        if response.tool_calls:
            return AgentResult.incomplete(
                iterations=final_iteration,
                tool_calls_count=total_tool_calls,
                reason="MODEL_REQUESTED_TOOL_DURING_FINALIZATION",
                final_text=response.content,
                usage=total_usage,
            )
        if not response.content or not response.content.strip():
            return AgentResult.incomplete(
                iterations=final_iteration,
                tool_calls_count=total_tool_calls,
                reason="MODEL_RETURNED_EMPTY_FINAL_RESPONSE",
                usage=total_usage,
            )

        logger.info(
            "run=%s completed_after_iteration_limit iterations=%d "
            "tool_calls=%d total_tokens=%d",
            run_id,
            final_iteration,
            total_tool_calls,
            total_usage.total_tokens,
        )
        return AgentResult.completed(
            final_text=response.content,
            iterations=final_iteration,
            tool_calls_count=total_tool_calls,
            usage=total_usage,
        )
