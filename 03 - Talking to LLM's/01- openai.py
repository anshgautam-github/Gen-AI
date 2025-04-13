from openai import OpenAI

client = OpenAI(api_key="your_API_key")

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role":"user", "content":"what is 2 + 2"}
    ]
)

print(result.choices[0].message.content)
