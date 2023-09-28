# Importing required libraries and modules
import os  # Provides functions for interacting with the operating system
import torch
import datasets  # Library providing a collection of datasets and metric for Natural Language Processing
from datasets import load_dataset  # Function to load a dataset
from transformers import AutoModelForCausalLM, AutoTokenizer  # Importing transformers models and tokenizers
from transformers import DataCollatorForLanguageModeling  # Provides a function to collate data for language modeling
from transformers import TrainingArguments, Trainer  # Provides training utility functions

# Uncomment this next line to use LLama
from llama_model import model, tokenizer


# Function to tokenize dataset
def tokenize_dataset(dataset):
    # Inner function to apply tokenization to the text examples
    def tokenize_function(examples):
        output = tokenizer(examples['text'])
        return output
    
    # Map applies the given function to all items in the provided dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Function to get or create a tokenized dataset
def get_dataset(dataset_path, text_file_path):
    tokenized_path = f'{dataset_path}/tokenized'
    
    data_files = {'train': text_file_path}
    # Loading the dataset and sampling it by paragraph
    train_dataset = load_dataset('text', data_files=data_files, sample_by="paragraph")
    # Tokenizing the dataset
    train_dataset = tokenize_dataset(train_dataset['train'])
    
    return train_dataset

# Main function to execute the model training and text generation
def main(model, tokenizer):
    # Paths to dataset and model
    book_name = 'The_Book_of_Mormon'
    dataset_path = f'datasets/{book_name}'
    text_file_path = f'{dataset_path}/book.txt'
    my_project_name = 'my_cool_model'
    save_path = f'models/{my_project_name}'
    
    # Get the tokenized dataset
    train_dataset = get_dataset(dataset_path, text_file_path)
    
    # Define the training arguments
    training_args = TrainingArguments(
        save_total_limit=1,  # Maximum number of checkpoints to keep
        output_dir=save_path,  # Output directory for model and checkpoints
        learning_rate=2e-5,  # Learning rate for training
        num_train_epochs=1,  # Number of training epochs
        per_device_train_batch_size=1,  # Batch size per device
        warmup_steps=500,  # Number of warm-up steps
    )
    
    # Define the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    # Save the trained model
    trainer.save_model(save_path)
    
    # Prompt for text generation
    prompt = "21 And it came to pass that the people of Nephi did"
    encoded = tokenizer(prompt, return_tensors="pt").to('cuda')

    # Generate text based on prompt
    output_answer = model.generate(encoded['input_ids'], max_length=50)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)

# Execution point for the script
if __name__ == '__main__':
    directory_name = "datasets"
    # Check if the datasets directory exists
    if not os.path.exists(directory_name):
        print('Run "python setup.py" to download datasets and models from the login node. Then "sbatch job.sh".\n Look at readme.md for more setup info.')
    
    # Define the model and tokenizer
    # model_name = 'gpt2'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Call the main function
    main(model, tokenizer)
