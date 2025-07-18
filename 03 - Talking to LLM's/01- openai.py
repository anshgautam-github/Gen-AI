#This line imports the OpenAI class from the openai package. This class allows you to talk to OpenAI's language models.
from openai import OpenAI

#This creates a client object that you’ll use to talk to the model.
client = OpenAI(api_key="your_API_key")

#This block sends a message to the GPT-4 model and stores the response in the variable result.
result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role":"user", "content":"what is 2 + 2"}
    ]
)

#This line prints out the model’s response to your question.
print(result.choices[0].message.content)
