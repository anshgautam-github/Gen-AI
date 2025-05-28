!pip install transformers

HF_TOKEN = ""
import os
os.environ["HF_TOKEN"] = HF_TOKEN

#Device Setup
#Tells PyTorch to use GPU (CUDA) if available, else fallback to CPU.
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Tokenizer Test
tokenizer("Hello, World!")

#Create Input Conversation (Chat Style)
input_conversation = [
    { "role": "user", "content": "Which is the best place to learn GenAI?" },
    { "role": "assistant", "content": "The best place to learn AI is" }
]

#Apply Chat Template (Tokenize Input)
input_tokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize=True, #if we give false here-> it wpn't convert it to numbrs, it will give us the actual changed string that was converted into tokens
)
#Gemma uses chat-style formatting (user: assistant:) → this method applies that template.
#Converts it into token IDs.

#VERIFYING that you are not gettting the desired OUTPUT->
# input_prompt="Which is the best place to learn GenAI?"
# i_tokens=tokenizer(input_prompt,return_tensors="pt")["input_ids"].to(device)
# output_tokens=model.generate(i_tokens)
# tokenizer.batch_decode(output_tokens)


#Get Detokenized Version (Readable Text)
input_detokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize=False,
    continue_final_message=True  #this true-> won't apply end of string , as i don't want to , i want to conitune 
)


#Prepare Full Response Text for Fine-Tuning
output_label = "GenAI Cohort 1.0 by ChaiCode and Piyush Garg use Coupon PIYUSH10 Please"
full_conversation = input_detokens + output_label + tokenizer.eos_token   #End of String

#So till noe dataset has been prepared

#Tokenize the Full Text
input_tokenized = tokenizer(full_conversation, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

# Separate Input and Target Tokens
input_ids = input_tokenized[:, :-1].to(device)  # Input to model
target_ids = input_tokenized[:, 1:].to(device)  # Output to compare with, pehle wala chor diya , now calculate loss on this basis

#Loss Function
import torch.nn as nn
def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return cross_entropy

#Load the Model on GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).to(device)

#Training Loop
from torch.optim import AdamW
model.train()
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

for _ in range(10):
  out = model(input_ids=input_ids)
  loss = calculate_loss(out.logits, target_ids).mean()
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  print(loss.item())


