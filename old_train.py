import argparse
from old_model import model, tokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import datasets
import os

# Function to tokenize dataset
def tokenize_dataset(dataset):
    # Inner function to apply tokenization to the text examples
    def tokenize_function(examples):
        output = tokenizer(examples['text'])
        # output['labels'] = output['input_ids']
        return output
    
    # Map applies the given function to all items in the provided dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Function to get or create a tokenized dataset
def get_dataset(dataset_path, text_file_path):
    tokenized_path = f'{dataset_path}/tokenized'
    
    # If tokenized dataset doesn't exist, create it
    if not os.path.exists(tokenized_path):
        data_files = {'train': text_file_path}
        # Loading the dataset and sampling it by paragraph
        train_dataset = load_dataset('text', data_files=data_files, sample_by="paragraph")
        # Tokenizing the dataset
        train_dataset = tokenize_dataset(train_dataset['train'])
        # Save the tokenized dataset to disk
        train_dataset.save_to_disk(tokenized_path)
    else:
        # Load the tokenized dataset from disk
        train_dataset = datasets.load_from_disk(tokenized_path)
    
    return train_dataset

def main():
    # Paths to dataset and model
    book_name = 'The_Book_of_Mormon'
    dataset_path = f'datasets/{book_name}'
    text_file_path = f'{dataset_path}/book.txt'
    my_project_name = 'my_cool_model'
    save_path = f'models/{my_project_name}'

    # Get the tokenized dataset
    train_dataset = get_dataset(dataset_path, text_file_path)

    training_args = TrainingArguments(
        learning_rate=1e-3,
        output_dir="./checkpoints",
        num_train_epochs=.1,
        per_device_train_batch_size=1,
        warmup_steps=500,
        logging_dir="./logs"
    )

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(save_path)

    prompt = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>You will be given a definition of a task first, then some input of the task.\nThis task is about using the specified sentence and converting the sentence to Resource Description Framework (RDF) triplets of the form (subject, predicate object). The RDF triplets generated must be such that the triplets accurately capture the structure and semantics of the input sentence. The input is a sentence and the output is a list of triplets of the form [subject, predicate, object] that capture the relationships present in the sentence. When a sentence has more than 1 RDF triplet possible, the output must contain all of them.\n\nAFC Ajax (amateurs)'s ground is Sportpark De Toekomst where Ajax Youth Academy also play.\nOutput:<SEP>"
    encoded = tokenizer.encode(prompt, return_tensors="pt")

    # I think this might not work?
    model.to('cpu')

    output_answer = model.generate(encoded, max_new_tokens=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)

if __name__ == '__main__':
    main()