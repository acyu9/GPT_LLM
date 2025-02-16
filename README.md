# GPT Large Language Model

This project implements two key components:

1. **GPT-1 Model**: Following the tutorial [*Create a Large Language Model from Scratch with Python*](https://www.youtube.com/watch?v=UU1WVnMk4E8&t=696s&ab_channel=freeCodeCamp.org) by FreeCodeCamp, this part of the project builds a language model from scratch using PyTorch and CUDA. It covers tokenization, model architecture, and training with a custom implementation of a transformer-based architecture in Jupyter Notebook.

2. **GPT-2 Model**: Using the Hugging Face `transformers` library, generated responses from pre-trained GPT-2 model, GPT-2 model fine-tuned with *Wizard of Oz* text, and pipeline model are compared.

## Features  
- Implements a custom LLM based on the GPT-1 architecture from scratch using PyTorch and CUDA.  
- Fine-tunes a pre-trained GPT-2 model to generate *Wizard of Oz*-like text.  
- Covers key tasks like tokenization, model architecture, and training for both models.  
- Demonstrates the process of training from scratch and fine-tuning a pre-trained model for text generation. 

## Libraries  
This project uses the following Python libraries:  
- transformer
- torch  
- datasets
- Jupyter Notebook  