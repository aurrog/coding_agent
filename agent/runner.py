from config import Settings
from agent.llm import LLMClient


class AgentRunner:
    def __init__(self, llm: LLMClient, tools, settings: Settings):
        self._llm = llm
        self._tools = tools
        self._setting = settings


    def run(self, user_request: str):
        pass