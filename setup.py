import os
import subprocess
from transformers import AutoTokenizer

# def download_llama():
#     print(f'\n\nDownloading Llama\r')
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     print("\033[1;32m Downloaded Llama-2! \033[1;30m\n\n")

def download_dataset():
    #TODO: Download books to dataset folder
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

if __name__ == "__main__":
    # download_llama()
    download_dataset()
