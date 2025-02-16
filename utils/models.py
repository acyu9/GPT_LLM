from transformers import GPT2LMHeadModel
from utils.device import get_device

def load_pretrained_model():
    device = get_device()
    return GPT2LMHeadModel.from_pretrained("gpt2").to(device)
