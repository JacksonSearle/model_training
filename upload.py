from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the tokenizer and model
model_name = "my_cool_model"
save_path = "models/" + model_name

model = AutoModelForCausalLM.from_pretrained(save_path)

# Push the model and tokenizer to the Hugging Face Model Hub
huggingface_model_name = 'my_cool_model'
model.push_to_hub(huggingface_model_name)
tokenizer.push_to_hub(huggingface_model_name)
