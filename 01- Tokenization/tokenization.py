import tiktoken   
#TikTOken is a tokenizer library developed by OpenAI. 
#It's used to break down text into smaller pieces called tokens


#how to use it, we need an encoder and then select a model
encoder= tiktoken.encoding_for_model('gpt-4o')

print("Vocab size", encoder.n_vocab) #2,00,019 (200k)
#here we can only know the vocab size of OpenAI bcoz tiktoken is the library used by OpenAI for tokenization

text="The cat sat on the mat"
tokens=encoder.encode(text)

print("Tokens",tokens)  #[976, 9059, 10139, 402, 290, 2450] , same tokens we get if we get mutiple times

#Decoding the TOKENS
my_tokens=[976, 9059, 10139, 402, 290, 2450]
decoded=encoder.decode(my_tokens)
print("Decoded Text-> ",decoded) #The cat sat on the mat
