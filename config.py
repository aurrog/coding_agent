from dataclasses import dataclass, field
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
class ObservabilitySettings:
    log_level: str = "INFO"
    log_file: Path = Path("agent.log")
    show_progress: bool = True


@dataclass(frozen=True)
class VerificationSettings:
    test_command: tuple[str, ...] = (
        "uv",
        "run",
        "python",
        "-m",
        "unittest",
        "discover",
        "-v",
    )
    lint_command: tuple[str, ...] = (
        "ruff",
        "check",
        ".",
    )
    timeout_seconds: float = 120.0
    max_output_characters: int = 12_000


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    permission_mode: str
    llm: LLMSettings
    agent: AgentSettings
    observability: ObservabilitySettings = field(
        default_factory=ObservabilitySettings
    )
    verification: VerificationSettings = field(
        default_factory=VerificationSettings
    )

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
        log_level = _log_level(os.getenv("LOG_LEVEL", "INFO"))
        log_file = Path(
            os.getenv("LOG_FILE", "agent.log") or "agent.log"
        ).expanduser().resolve()
        show_progress = _boolean(
            os.getenv("SHOW_PROGRESS", "true"),
            "SHOW_PROGRESS",
        )
        verification_timeout_seconds = _bounded_positive_float(
            os.getenv("VERIFICATION_TIMEOUT_SECONDS", "120"),
            "VERIFICATION_TIMEOUT_SECONDS",
            maximum=300,
        )
        max_command_output_characters = _bounded_positive_int(
            os.getenv("MAX_COMMAND_OUTPUT_CHARACTERS", "12000"),
            "MAX_COMMAND_OUTPUT_CHARACTERS",
            maximum=100_000,
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
            observability=ObservabilitySettings(
                log_level=log_level,
                log_file=log_file,
                show_progress=show_progress,
            ),
            verification=VerificationSettings(
                timeout_seconds=verification_timeout_seconds,
                max_output_characters=max_command_output_characters,
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


def _bounded_positive_int(
    raw_value: str,
    name: str,
    *,
    maximum: int,
) -> int:
    value = _positive_int(raw_value, name)
    if value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _bounded_positive_float(
    raw_value: str,
    name: str,
    *,
    maximum: float,
) -> float:
    value = _positive_float(raw_value, name)
    if value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _log_level(raw_value: str) -> str:
    value = raw_value.upper()
    if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("LOG_LEVEL must be a standard logging level")
    return value


def _boolean(raw_value: str, name: str) -> bool:
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")
