from openai import OpenAI

client = OpenAI(api_key="your_key")

response=client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"user","content":"what is the current weather of the chennai?"}
     ]
)

print(response.choices[0].message.content)

#Why it didn't work? Cut-off knowkedge (kind of you're right and wrong)
# WEather data changes every second, even not every second, it updates in hours every 1 or 2
# LLm's work on pretrained model-> Uske andar we don;t have any real time thing ,so it is not of our use
# What we have to do is ->> we have to make our LLM an agent -> ki bhai tere pass dimag h , 
# hath , pao mere use kr-> tu ja na , user ne pucha h usko smartly anaylze kr -> usko NLP kr, 
# usko understand kr -> , aur google pe ja -> search kar weather -> and respnse de muje

#we will use a system prompt -> sari cheez yahi a agyi  
# TOOL mtlb ek function 
