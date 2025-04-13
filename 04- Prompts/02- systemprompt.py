from openai import OpenAI

client = OpenAI(api_key="your_API_key")


result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role":"system", "content":"You are an AI assitent and your name is Aksay Kumar"}, #System prompt -> to set the initial background
        {"role":"user", "content":"what is your name"} 
    ]
)

print(result.choices[0].message.content)
