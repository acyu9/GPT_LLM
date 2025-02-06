from utils.models import load_pretrained_model
from utils.tokenizer import get_tokenizer
from utils.device import get_device
from scripts.generate import generate_text

PROMPT = "Once upon a time in the land of Oz"

def pretrained_gpt():
    tokenizer = get_tokenizer()
    model = load_pretrained_model()
    device = get_device()

    inputs = tokenizer(PROMPT, return_tensors="pt", max_length=1024, truncation=True).to(device)

    generated_text = generate_text(tokenizer, inputs, model)

    print("Pre-Trained Generated Text:\n", generated_text)

def fine_tuned_gpt():
    # Tokenize "Wizard of Oz" text
    with open("wizard_of_oz.txt", "r") as file:
        text = file.read()

    # Max token length for GPT2
    # inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

if __name__ == "__main__":
    pretrained_gpt()
