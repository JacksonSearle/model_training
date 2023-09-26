import argparse
from model import model, tokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def load_datasets(data_files):
    pass

def tokenize_dataset(dataset):
    return


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
    model_name = 'my_cool_model'
    print(f'Model Name: {model_name}')
    save_path = "models/" + model_name

    dataset_path = "datasets/Romeo_and_Juliet.txt"

    train_dataset = load_datasets(dataset_path)

    # load the text file
    # Split the text file into train and test
    # Tokenize the dataset if it doesn't exist
    # Save the tokenized file to a folder called "tokenized". Create that folder if it doesn't exist
    # 

    training_args = TrainingArguments(
        learning_rate=1e-3,
        output_dir="./checkpoints",
        num_train_epochs=.1,
        per_device_train_batch_size=4,
        warmup_steps=500,
        logging_dir="./logs"
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(save_path)

    prompt = "Once upon a time "
    encoded = tokenizer.encode(prompt, return_tensors="pt")

    output_answer = model.generate(encoded, max_new_tokens=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)

if __name__ == '__main__':
    main()