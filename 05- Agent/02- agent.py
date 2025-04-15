import json
from openai import OpenAI

client = OpenAI(api_key="")


def get_weather(city:str):
    #TODO: Do an actual API call
    return "31 deg celcius"

system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}
"""

response=client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":"what is the current weather of the new york?"},
        { "role": "assistant", "content": json.dumps({ "step": "plan", "content": "The user is interseted in weather data of new york" })  },
        { "role": "assistant", "content": json.dumps({ "step": "plan", "content": "From the available tools I should call get_weather" }) },
        { "role": "assistant", "content": json.dumps({ "step": "action", "function": "get_weather", "input": "new york" }) },
        { "role": "assistant", "content": json.dumps({ "step": "observe", "output": "12 Degree Cel" }) }
    
     ]
)

print(response.choices[0].message.content)
