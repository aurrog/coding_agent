from agent.llm import OpenAICompatibleLLM
from agent.runner import AgentRunner
from config import AgentContext, Settings
from core.models import AgentStatus
from security.policy import ToolPolicy
from security.workspace import Workspace
from tools.files import (
    CreateFileTool,
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
)
from tools.registry import ToolRegistry


def build_runner(settings: Settings) -> AgentRunner:
    workspace = Workspace(settings.workspace_root)
    policy = ToolPolicy(settings.permission_mode)
    registry = ToolRegistry(policy)
    registry.register(ListFilesTool(workspace))
    registry.register(ReadFileTool(workspace))
    registry.register(SearchTextTool(workspace))
    registry.register(CreateFileTool(workspace))
    registry.register(EditFileTool(workspace))

    return AgentRunner(
        llm=OpenAICompatibleLLM(settings.llm),
        tools=registry,
        settings=settings.agent,
        context=AgentContext(
            workspace_root=settings.workspace_root,
            permission_mode=policy.mode.value,
        ),
    )


def main() -> int:
    print("Введите путь к рабочей директории")
    workspace_root = input(">>> ").strip()
    if not workspace_root:
        print("Рабочая директория не указана")
        return 2

    try:
        settings = Settings.from_env(workspace_root)
        runner = build_runner(settings)
    except (OSError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    print("\nВведите запрос на анализ кода")
    user_request = input(">>> ").strip()
    if not user_request:
        print("Запрос не указан")
        return 2

    result = runner.run(user_request)
    if result.final_text:
        print(result.final_text)
    if result.status != AgentStatus.COMPLETED:
        print(f"Agent stopped: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
