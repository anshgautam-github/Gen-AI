from google import genai
from google.genai import types

# Only run this block for Gemini Developer API
client = genai.Client(api_key='your_API_key')

response = client.models.generate_content( #here in the content,we do not need to specify the start,end, etc, this particualr sdk will do on it's own
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
    
)
print(response.text)
