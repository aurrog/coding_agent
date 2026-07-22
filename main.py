from agent import loop


user_query='Write a /health endpoint in working_directory'


r = loop.run_agent(user_request=user_query, max_iterations=5)
print(r)