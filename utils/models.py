from transformers import GPT2LMHeadModel
from utils.device import get_device

def load_pretrained_model():
    device = get_device()
    return GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# def load_fine_tuned_model(model_path="models/fine_tuned_gpt2/"):
#     return GPT2LMHeadModel.from_pretrained(model_path)
