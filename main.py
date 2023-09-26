import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def load_datasets(data_files):
    dataset = load_dataset('text', data_files=data_files)
    return dataset


def tokenize_dataset(dataset):
    def tokenize_function(examples):
        return tokenizer(examples['text'])
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def train_model(train_dataset, data_collator, training_args, save_path):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(save_path)


def main():
    # Grab the right 
    model_name = 'gpt-2'  # You can replace this with any other model you are using
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset_path = "datasets/Romeo_and_Juliet/book.txt"
    save_path = "models/my_cool_model"
    tokenized_path = "tokenized"
    
    if not os.path.exists(tokenized_path):
        os.makedirs(tokenized_path)
        
    train_dataset = load_datasets({'train': dataset_path})
    train_dataset = tokenize_dataset(train_dataset['train'])
    train_dataset.save_to_disk(tokenized_path)
    
    training_args = TrainingArguments(
        learning_rate=1e-3,
        output_dir="./checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        warmup_steps=500,
        logging_dir="./logs"
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_model(train_dataset, data_collator, training_args, save_path)

    prompt = "Once upon a time "
    encoded = tokenizer(prompt, return_tensors="pt")

    output_answer = model.generate(encoded['input_ids'], max_length=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)


if __name__ == '__main__':
    main()