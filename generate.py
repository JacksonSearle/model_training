from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

checkpoint = "models/my_cool_model"

# Load the trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Generate answers using the trained model
input_question = "But soft, what light through yonder window breaks?\n"
input_question_encoded = tokenizer.encode(input_question, return_tensors="pt")
output_answer = model.generate(input_question_encoded, max_new_tokens=200)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)