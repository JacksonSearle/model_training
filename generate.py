from transformers import AutoModelForCausalLM, AutoTokenizer

model_tokenizer_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# You can generate from your trained model, or the vanilla version of anything on Huggingface

model_name = "models/my_cool_model" # or "gpt2-medium" for example

model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate answers using the trained model
input_prompt = "21 And it came to pass that the people of Nephi did"
input_prompt_encoded = tokenizer.encode(input_prompt, return_tensors="pt")
output_answer = model.generate(input_prompt_encoded, max_new_tokens=50)
decoded_answer = tokenizer.decode(output_answer[0])
print("\nGenerated Answer:\n\n", decoded_answer)