from dataclasses import dataclass

from dotenv import load_dotenv
from pathlib import Path
import os


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


@dataclass(frozen=True)
class Settings:
    workspace_root:Path
    permission_mode: str
    llm: LLMSettings
    agent: AgentSettings

    @classmethod
    def from_env(cls, workspace_root: str, env_path='.env'):
        load_dotenv(env_path)

        try:
            api_key=os.environ['API_KEY']
            base_url=os.environ['BASE_URL']
            model=os.environ['MODEL']

            timeout_settings=os.environ['TIMEOUT_SETTINGS']
            max_retries=os.environ['MAX_RETRIES']

            max_iterations=os.environ['MAX_ITERATIONS']
            max_tool_calls=os.environ['MAX_TOOL_CALLS']
            max_context_characters=os.environ['MAX_CONTEXT_CHARACTERS']

        except KeyError:
            raise KeyError('secret keys are not load')

        llm=LLMSettings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_settings,
            max_retries=max_retries
        )

        agent=AgentSettings(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_context_characters=max_context_characters
        )

        return Settings(
            workspace_root=workspace_root,
            permission_mode='',
            llm=llm,
            agent=agent
        )


        
