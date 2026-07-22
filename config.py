from dataclasses import dataclass

@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int
    max_retries: int


@dataclass(frozen=True)
class AgentSettings:
    max_iterations: int
    max_tool_calls: int
    max_context_characters: int



class Settings:
    def __init__(self):
        pass