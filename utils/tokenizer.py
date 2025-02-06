from transformers import GPT2Tokenizer

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Simplies model & speed up training - eos indicates the end of sequence
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer