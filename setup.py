import os
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_models():
    model_names = ['gpt2', 'gpt2-medium', "meta-llama/Llama-2-7b-hf"]
    for model_name in model_names:
        print(f'Downloading {model_name} model...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(model_name)

def download_dataset():
    print(f'\n\nDownloading datasets')
    directory_name = "datasets"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    books = {
        "The_Book_of_Mormon": "https://www.gutenberg.org/files/17/17-0.txt",
        "The_Letters_of_Jane_Austen": "https://www.gutenberg.org/cache/epub/42078/pg42078.txt",
        "My_Bondage_and_My_Freedom": "https://www.gutenberg.org/cache/epub/202/pg202.txt",
        "Anne_of_Green_Gables": "https://www.gutenberg.org/cache/epub/45/pg45.txt",
        "Romeo_and_Juliet": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
    }

    for name, link in books.items():
        if not os.path.exists(f'{directory_name}/{name}'):
            os.makedirs(f'{directory_name}/{name}')
        name = f'{directory_name}/{name}/book.txt'
        print(name)
        if not os.path.exists(name):
            try:
                subprocess.check_call([
                    "curl",
                    "-L",
                    "-o",
                    name,
                    link
                ])
            except subprocess.CalledProcessError as e:
                print(f"Error downloading file: {e}")
                exit()

    print("Datesets downloaded successfully!")

def complete_setup():
    download_models()
    download_dataset()

if __name__ == "__main__":
    complete_setup()
