from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM

def scale(parameter):
    return int(parameter * 1/20)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

configuration = LlamaConfig(vocab_size = len(tokenizer), 
                            hidden_size = scale(4096), 
                            intermediate_size = scale(11008), 
                            num_hidden_layers = scale(32), 
                            num_attention_heads = scale(32), 
                            
                            max_position_embeddings = 2048)

model = LlamaForCausalLM(configuration)

million_parameters = int(model.num_parameters()/1000000)
print(f'{million_parameters} Million Parameters') 