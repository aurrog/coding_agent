from core.models import ToolCall
from tools.base import Tool

class ToolRegistry:
    def __init__(self, policy):
        self._tools={}
        self._policy={}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f'Tool already registred: {tool.name}')

        self._tools[tool.name]=tool

    def execute(self, call: ToolCall):
        tool=self._tools.get(call.name)

        if tool is None:
            pass
