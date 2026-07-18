from agent.llm import chat


messages = [
    {'role': 'system', 'content': 'Ты - ИИ помощник программист. Пиши кратко и только то, что просят.'}
]

while True:
    user_input=input('>>> ')

    if user_input=='quit':
        break

    messages.append({'role':'user', 'content':user_input})

    model_output=chat(messages)
    messages.append({'role':'assistant', 'content': model_output})
    print('\n',model_output)
    

