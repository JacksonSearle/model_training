import os
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from model import model, tokenizer

def load_datasets(data_files):
    dataset = load_dataset('text', data_files=data_files)
    return dataset


def tokenize_dataset(dataset):
    def tokenize_function(examples):
        output = tokenizer(
            examples['text'],
            max_length=512, #
            truncation=True #
            )
        # output["labels"] = output["input_ids"].clone()
        return output
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
    # input_ids_column = tokenized_datasets['input_ids']
    # tokenized_datasets.add_column(name='labels', column=input_ids_column)
    # tokenized_datasets = tokenized_datasets.rename_column('input_ids', 'labels')
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
    dataset_path = "datasets/Romeo_and_Juliet/book.txt"
    save_path = "models/my_cool_model/final_model"
    tokenized_path = "tokenized"

    train_dataset = load_datasets({'train': dataset_path})
    train_dataset = tokenize_dataset(train_dataset['train'])
    train_dataset.save_to_disk(tokenized_path)
    
    training_args = TrainingArguments(
        output_dir="models/my_cool_model",
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        warmup_steps=500,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=1024)
    
    train_model(train_dataset, data_collator, training_args, save_path)

    prompt = "Once upon a time "
    encoded = tokenizer(prompt, return_tensors="pt")

    output_answer = model.generate(encoded['input_ids'], max_length=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)


if __name__ == '__main__':
    main()