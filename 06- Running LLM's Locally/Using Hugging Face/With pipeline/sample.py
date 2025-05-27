#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#📦 Step 1: Install & Set Up 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Installs the Hugging Face transformers library which gives us access to pre-trained language models.
!pip install transformers

#This sets your Hugging Face token (so you can access restricted models like Gemma). You can get this token from your Hugging Face account.
import os
os.environ["HF_TOKEN"] = ""

#Specifies which model you're going to use. "google/gemma-3-1b-it" is a Google-trained instruction-tuned version of their Gemma model.
model_name = "google/gemma-3-1b-it"
#So, now how do we run a model? first we do tokenisation-> user input ko tokenize, then put in transformer , then de-tokenize it.


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#🤖 Step 2: Load the Model & Tokenizer
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

from transformers import AutoTokenizer
#Loads a tokenizer that converts text like "Hello" into numbers (tokens) that the model understand
#Differnt models have diff tokenizers, here, autoTokenizer automatically takes care which tokenizer to make use of as per the model 
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Shows how the tokenizer breaks the input sentence into token IDs + attention mask.
print(tokenizer("Hello, how are you?"))
print(tokenizer.get_vocab())
input_tokens = tokenizer("Hello, how are you?")["input_ids"]  #yeh input_id -> when u print tokenizer thing, u get output, in that we get token as well as attention mask-> usmei se we only need input token , so we mentined there specifically
print(input_tokens)
#Now tokenization process done -> we now have to load the model 


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#🧠 Step 4: Load the Model
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Loads the actual Gemma model in bfloat16 format (a lighter precision that saves memory, often used on Colab with GPUs).
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)
#now we have both tokenizer and the model(tranformer), now we just have to start the transformer 


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#✨ Step 5: Generate a Response
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Create a Text Generation Pipeline
from transformers import pipeline

#Transformers have a function named pipeline, so we can create a pipeline for text generation, and we say to that pipeline, that i need a textgeneration piepline,
#and i pass my model and tokenizer
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
#pipleine crated 
#So i said to my pipeline func -> that this is my tokenizer and model, so jo bhi user tumko input dega-> usko iss tokenizer se tokenize and detokenize krna h and
#usi model ya transformer ke andar use krna h.

#Generate Text
gen_pipeline("Hey there", max_new_tokens=25)
#now i just calleld my pipeline with the input and internally, it did create token, de tokentize..etc

#The pipeline in the transformers library provides a high-level abstraction for working with transformer-based models. 
#It simplifies the process of running common tasks like text generation, text classification, sentiment analysis, and more, without requiring you to write a lot of code for each task.
