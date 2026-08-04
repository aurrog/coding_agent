from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str | None
    model: str
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class AgentSettings:
    max_iterations: int
    max_tool_calls: int
    max_context_characters: int


@dataclass(frozen=True)
class AgentContext:
    workspace_root: Path
    permission_mode: str


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    permission_mode: str
    llm: LLMSettings
    agent: AgentSettings

    @classmethod
    def from_env(
        cls,
        workspace_root: str | Path,
        env_path: str | Path = ".env",
    ) -> "Settings":
        load_dotenv(env_path)

        api_key = _required_env("API_KEY")
        model = _required_env("MODEL")
        base_url = os.getenv("BASE_URL") or None

        timeout_seconds = _positive_float(
            os.getenv("TIMEOUT_SECONDS")
            or os.getenv("TIMEOUT_SETTINGS")
            or "60",
            "TIMEOUT_SECONDS",
        )
        max_retries = _non_negative_int(
            os.getenv("MAX_RETRIES", "2"),
            "MAX_RETRIES",
        )
        max_iterations = _positive_int(
            os.getenv("MAX_ITERATIONS", "8"),
            "MAX_ITERATIONS",
        )
        max_tool_calls = _positive_int(
            os.getenv("MAX_TOOL_CALLS", "20"),
            "MAX_TOOL_CALLS",
        )
        max_context_characters = _positive_int(
            os.getenv("MAX_CONTEXT_CHARACTERS", "200000"),
            "MAX_CONTEXT_CHARACTERS",
        )

        permission_mode = os.getenv(
            "PERMISSION_MODE",
            "read_only",
        )

        return cls(
            workspace_root=Path(workspace_root).expanduser().resolve(),
            permission_mode=permission_mode,
            llm=LLMSettings(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            ),
            agent=AgentSettings(
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                max_context_characters=max_context_characters,
            ),
        )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


def _positive_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_int(raw_value: str, name: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _positive_float(raw_value: str, name: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
