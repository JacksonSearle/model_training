from transformers import AutoModelForCausalLM, AutoTokenizer

model_tokenizer_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# You can generate from your trained model, or the vanilla version of anything on Huggingface

model_name = "models/my_cool_model" # or "gpt2-medium" for example

model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate answers using the trained model
input_question = "But soft, what light through yonder window breaks?\n"
input_question_encoded = tokenizer.encode(input_question, return_tensors="pt")
output_answer = model.generate(input_question_encoded, max_new_tokens=200)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)