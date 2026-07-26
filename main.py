# from agent import loop


# user_query='Remove all from working_directory/main.py and write a FastAPI /health endpoint'


# r = loop.run_agent(user_request=user_query, max_iterations=5)
# print('-'*100)

# print(r)

from config import Settings
from agent.llm import OpenAICompatibleLLM


settings=Settings.from_env('working_directory')


llm=OpenAICompatibleLLM(settings.llm)



