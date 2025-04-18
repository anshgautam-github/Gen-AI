!pip install transformers
import os
os.environ["HF_TOKEN"] = ""
model_name = "google/gemma-3-1b-it"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer("Hello, how are you?"))
print(tokenizer.get_vocab())
input_tokens = tokenizer("Hello, how are you?")["input_ids"]
print(input_tokens)


from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)


#Create a Text Generation Pipeline
from transformers import pipeline
gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


#Generate Text
gen_pipeline("Hey there", max_new_tokens=25)


#The pipeline in the transformers library provides a high-level abstraction for working with transformer-based models. 
#It simplifies the process of running common tasks like text generation, text classification, sentiment analysis, and more, without requiring you to write a lot of code for each task.
