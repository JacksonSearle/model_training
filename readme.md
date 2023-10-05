# Model Training Guide

Welcome to the Model Training Guide! This repository serves as a comprehensive guide for setting up, training, generating, and uploading text models utilizing Hugging Face's gpt2 and Llama 2 platforms. The project offers structured environment, concise file structures, and clear instructions to facilitate smooth user experience throughout the model training process.

## Project Structure Overview
This project comprises several directories and files, each serving a specific purpose in the model training process:
- **datasets**: Holds all the datasets.
- **models**: Contains all the models.
- **setup.py**: Responsible for downloading the required datasets and models for training.
- **train.py**: Initiates the model training process.
- **generate.py**: Generates text from a trained model.
- **upload.py**: Allows uploading of models to Hugging Face's hub.

## Initial Setup and Environment Configuration

To begin, adhere to the following steps to ensure your environment is correctly set up and ready for model training:

1. **Create a Hugging Face Account**: Navigate to [Hugging Face](https://huggingface.co/login) and sign up if you haven’t already.
2. **Request Llama 2 Access**: Apply for access at [Llama 2](https://ai.meta.com/llama/) using the same email you used for Hugging Face.
3. **Obtain Access to the Llama-2-7b-hf Model**: Seek access to the [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model through Hugging Face, access is usually granted within 0-2 days.
4. **Clone the Repository and Set Up the Environment**:
   ```
   git clone https://github.com/JacksonSearle/model_training.git
   mamba create --name model_training
   mamba activate model_training
   mamba install python
   pip install -r requirements.txt
   ```
5. **Configure Access Token**:
   Follow the [token setup guide](https://huggingface.co/docs/hub/security-tokens) on Hugging Face. Then run ```huggingface-cli login``` in the command line to login to huggingface
6. **Run Setup and Submit Job**:
   ```
   python setup.py
   sbatch job.sh
   ```
7. **Upload to Hugging Face**:
   Once your job has successfully completed, run `python upload.py` in your command line to upload your model to Hugging Face's online hub. Login to your Hugging Face account to view your model or search for it [here](https://huggingface.co/models).

## Model Training and Generation
- Use `sbatch job.sh` to start the training process, and use `generate.py` to create text from your trained model.
- Adjust hyperparameters in `train.py` and `setup.py` as needed; refer to the **Hyperparameter Adjustments** section for more details.
- Explore different models, datasets, and make adjustments to hyperparameters as you wish.

### Hyperparameter Adjustments
- **num_train_epochs** (in train.py): Determines how many times the model will go through the entire dataset.
- **model_name** (in train.py): Change this to any gpt2 model name you see in setup.py. Add any new models you want to setup.py and re-run setup.py in the login node to download the model for training.
- **books** (in setup.py) and **book_name** (in train.py): In setup.py, include a book name and its URL from [Project Gutenberg](https://www.gutenberg.org/). Ensure the URL ends with ".txt". Re-run setup.py in the login node to download the new dataset for training. Change **book_name** in train.py to select that new book.
- **Wikipedia Dataset** (in train.py): This is a collection of Wikipedia articles. Uncomment the line in train.py to select it. Comment out the line that loads a book as a dataset.
- **max_new_tokens** (in generate.py and train.py): Change this to a larger value to generate more sample text from your model.
- **Train From Scratch** (in train.py): Comment out the block of code that defines the model and tokenizer. Uncomment the line towards the top of the page where it imports llama_model. Change the fraction in llama_model.py to change the size of the model. If your model is too big for the gpu you will get some errors during training. You may want to choose the Wikipedia dataset in train.py to give your from-scratch model enough data to learn English.

## GPU Training Recommendations
- For gpt2 (124M parameters) training try `#SBATCH --gpus=1 -C kepler` with a batch size of 1
- For gpt-medium (355M parameters) training try `#SBATCH --gpus=1 -C pascal` with a batch size of 1
- For models that don't fit on pascal, as the TAs for how to access ampere gpus
- If your job takes forever to run and you want to cancel it, type `scancel job_id` into the command line where the job_id is a large number corresponding to your job's id. Try `sbatch job.sh` again without the `-C gpu_name`. This will put your job on whatever GPU is available instead of selecting a specific one.

### Hardware and Resource Information
For insights on available hardware resources, refer to [BYU Resources](https://rc.byu.edu/documentation/resources).

## Conclusion
This comprehensive guide is designed to ease your journey through model training, generation, and uploading. We hope you explore various models and datasets and have a fulfilling coding experience. Happy training!
