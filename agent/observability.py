from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Protocol, TextIO

from core.models import TokenUsage


SYSTEM_LOGGER_NAME = "coding_agent"
_system_logger = logging.getLogger(SYSTEM_LOGGER_NAME)
_system_logger.addHandler(logging.NullHandler())
_system_logger.propagate = False


class ProgressReporter(Protocol):
    def model_request(
        self,
        iteration: int,
        max_iterations: int,
        *,
        final: bool = False,
    ) -> None:
        ...

    def model_response(
        self,
        current: TokenUsage,
        total: TokenUsage,
    ) -> None:
        ...

    def tool_started(self, name: str, target: str | None) -> None:
        ...

    def tool_finished(
        self,
        name: str,
        *,
        ok: bool,
        error_code: str | None,
        elapsed_seconds: float,
    ) -> None:
        ...


class NullProgressReporter:
    def model_request(
        self,
        iteration: int,
        max_iterations: int,
        *,
        final: bool = False,
    ) -> None:
        pass

    def model_response(
        self,
        current: TokenUsage,
        total: TokenUsage,
    ) -> None:
        pass

    def tool_started(self, name: str, target: str | None) -> None:
        pass

    def tool_finished(
        self,
        name: str,
        *,
        ok: bool,
        error_code: str | None,
        elapsed_seconds: float,
    ) -> None:
        pass


class ConsoleProgressReporter:
    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stderr
        self._usage_unavailable_reported = False

    def model_request(
        self,
        iteration: int,
        max_iterations: int,
        *,
        final: bool = False,
    ) -> None:
        if final:
            self._write(
                "[agent] Формирую финальный ответ без tools"
            )
            return
        self._write(
            f"[agent] Запрос к модели: итерация "
            f"{iteration}/{max_iterations}"
        )

    def model_response(
        self,
        current: TokenUsage,
        total: TokenUsage,
    ) -> None:
        if current.reported:
            self._write(
                "[tokens] "
                f"+{current.prompt_tokens} вход / "
                f"+{current.completion_tokens} выход; "
                f"всего {total.total_tokens}"
            )
        elif not self._usage_unavailable_reported:
            self._write(
                "[tokens] Провайдер не вернул статистику "
                "использования"
            )
            self._usage_unavailable_reported = True

    def tool_started(self, name: str, target: str | None) -> None:
        target_suffix = f" ({target})" if target else ""
        self._write(f"[tool] {name}{target_suffix}: запуск")

    def tool_finished(
        self,
        name: str,
        *,
        ok: bool,
        error_code: str | None,
        elapsed_seconds: float,
    ) -> None:
        if ok:
            outcome = "готово"
        else:
            outcome = f"ошибка {error_code or 'UNKNOWN'}"
        self._write(
            f"[tool] {name}: {outcome} ({elapsed_seconds:.3f} с)"
        )

    def _write(self, message: str) -> None:
        print(message, file=self._stream, flush=True)


def configure_system_logging(
    *,
    level: str,
    log_file: Path,
) -> None:
    logger = logging.getLogger(SYSTEM_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    )
    logger.addHandler(handler)


def format_usage(usage: TokenUsage) -> str:
    if not usage.reported:
        return "данные о токенах недоступны"

    details = (
        f"вход: {usage.prompt_tokens}, "
        f"выход: {usage.completion_tokens}, "
        f"всего: {usage.total_tokens}"
    )
    optional: list[str] = []
    if usage.cached_tokens:
        optional.append(f"cached: {usage.cached_tokens}")
    if usage.reasoning_tokens:
        optional.append(f"reasoning: {usage.reasoning_tokens}")
    if optional:
        details += f" ({', '.join(optional)})"
    return details
