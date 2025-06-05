from openai import OpenAI

client = OpenAI(api_key="your_API_key")

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role":"system", "content":"You are an AI assitent and your name is Aksay Kumar"}, #System prompt -> to set the initial background
        {"role":"user", "content":"what is your name"} 
    ]
)

#OR
#System prompt
# system_prompt="You are an AI assitent and your name is Aksay Kumar"

# result = client.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role":"system", "content":system_prompt}, 
#         {"role":"user", "content":"what is your name"} 
#     ]
# )


print(result.choices[0].message.content)
