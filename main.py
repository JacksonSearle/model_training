import os
import datasets
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from model import model, tokenizer

def tokenize_dataset(dataset):
    def tokenize_function(examples):
        output = tokenizer(
            examples['text']
            )
        return output
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

def get_dataset(dataset_path, text_file_path):
    tokenized_path = f'{dataset_path}/tokenized'

    if not os.path.exists(tokenized_path):
        data_files = {'train': text_file_path}
        train_dataset = load_dataset('text', data_files=data_files, sample_by="paragraph")
        train_dataset = tokenize_dataset(train_dataset['train'])
        train_dataset.save_to_disk(tokenized_path)
    else:
        train_dataset = datasets.load_from_disk(tokenized_path)
    return train_dataset


def main():
    book_name = 'Romeo_and_Juliet'
    dataset_path = f'datasets/{book_name}'
    text_file_path = f'{dataset_path}/book.txt'

    model_name = 'my_cool_model'
    save_path = f'models/{model_name}'

    train_dataset = get_dataset(dataset_path, text_file_path)
    
    training_args = TrainingArguments(
        output_dir="models/my_cool_model",
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        warmup_steps=500,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(save_path)

    prompt = "But soft, what light through yonder window breaks?\n"
    encoded = tokenizer(prompt, return_tensors="pt").to('cuda')

    output_answer = model.generate(encoded['input_ids'], max_length=200)
    decoded_answer = tokenizer.decode(output_answer[0])
    print("Generated Answer:", decoded_answer)


if __name__ == '__main__':
    main()