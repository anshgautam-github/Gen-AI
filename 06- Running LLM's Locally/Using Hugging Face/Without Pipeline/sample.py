#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#📦 Step 1: Install & Set Up
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Installs the Hugging Face transformers library which gives us access to pre-trained language models.
!pip install transformers

#This sets your Hugging Face token (so you can access restricted models like Gemma). You can get this token from your Hugging Face account.
#This sets your Hugging Face token (so you can access restricted models like Gemma). You can get this token from your Hugging Face account.
import os
os.environ["HF_TOKEN"] = "hf_..."  

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#🤖 Step 2: Load the Model & Tokenizer
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Specifies which model you're going to use. "google/gemma-3-1b-it" is a Google-trained instruction-tuned version of their Gemma model.
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "google/gemma-3-1b-it"

#Loads a tokenizer that converts text like "Hello" into numbers (tokens) that the model understands.
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Shows how the tokenizer breaks the input sentence into token IDs + attention mask.
print(tokenizer("hello, hi how are u"))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#📝 Step 3: Tokenize Your Prompt
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

##Converts your prompt into a tensor format (return_tensors="pt" means PyTorch) so the model can process it.
input_prompt = ["The capital of India is"]
tokenized = tokenizer(input_prompt, return_tensors="pt")
tokenized["input_ids"]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#🧠 Step 4: Load the Model
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Loads the actual Gemma model in bfloat16 format (a lighter precision that saves memory, often used on Colab with GPUs).
import torch
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#✨ Step 5: Generate a Response
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Tells the model: "Here's the prompt. Now generate up to 25 new words (tokens) that follow."
gen_result = model.generate(tokenized["input_ids"], max_new_tokens=25)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#📤 Step 6: Decode the Output
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Converts the token IDs back into human-readable text.
ouput = tokenizer.batch_decode(gen_result)



#✅ Final Output
#You’ll get something like:

#["The capital of India is New Delhi. It is one of the largest..."]

