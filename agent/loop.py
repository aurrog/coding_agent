from tools import files
from llm import chat_with_tools
from config import *
import json


def tools_use(tools_map, tool_calls):
    tools_output = []

    for toolcall in tool_calls:
        json_data = json.loads(toolcall.function.arguments)
        function = tools_map.get(toolcall.function.name, False)

        if not function:
            content = 'incorrect function name'
        else:
            try:
                content=str(function(**json_data))
                print('Used tool ', toolcall.function.name)

            except Exception as e:
                content=f'error when calling the tool: {e}'
        
        tools_output.append({
            'role': 'tool',
            'tool_call_id': toolcall.id,
            'content': content
        })
    
    return tools_output
                




def run_agent(user_request, max_iterations=5):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    tools_map = {
        'list_files': files.list_files,
        'read_file': files.read_file,
        'edit_file': files.edit_file
    }

    for _ in range(max_iterations):
        model_response = chat_with_tools(messages,TOOLS)
        assistant_message = model_response.choices[0].message

        messages.append(assistant_message)

        if model_response.choices[0].finish_reason == 'tool_calls':
            reasoning = model_response.choices[0].message.reasoning
            print(reasoning)

            tools_result = tools_use(tools_map, model_response.choices[0].message.tool_calls)

            messages.extend(tools_result)
        else:
            content = model_response.choices[0].message.content
    return content



        
