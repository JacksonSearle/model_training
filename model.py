from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'  # You can replace this with any other model you are using
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)