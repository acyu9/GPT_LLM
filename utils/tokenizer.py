from transformers import GPT2Tokenizer
from utils.device import get_device
from torch import Tensor

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Simplies model & speed up training - eos indicates the end of sequence
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_text(tokenizer, input: str) -> dict[str, Tensor]:
    """
    Tokenize the input str to 
    {
        'input_ids': tensor([[ 7454,  2402,   257,   640,   287,   262,  1956,   286, 18024]],
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], 
         device='cuda:0')
    }
    """
    device = get_device()

    # Max token length for GPT2
    tokenized_input = tokenizer(input, return_tensors="pt", max_length=1024, padding=True, truncation=True).to(device)
    return tokenized_input