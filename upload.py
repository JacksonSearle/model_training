from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the tokenizer and model
my_project_name = "my_cool_model"
save_path = "models/" + my_project_name

model = AutoModelForCausalLM.from_pretrained(save_path)

# Push the model and tokenizer to the Hugging Face Model Hub
model.push_to_hub(my_project_name)
tokenizer.push_to_hub(my_project_name)

print("Upload successful (hopefully)\n Login to your Huggingface account through a web browser to view your model :)")
print("You can easily share your model with others. Either share the url to the webpage, ")
print("or share the few lines of python code from generate.py to get them started")