from utils.models import load_pretrained_model
from utils.tokenizer import get_tokenizer, tokenize_text
from model.train import fine_tune
from model.generate import generate_text

PROMPT = "Once upon a time in the land of Oz"

def pretrained_gpt():
    """
    Generate text for the PROMPT with pretrained GPT2
    """
    tokenizer = get_tokenizer()
    model = load_pretrained_model()

    tokenized_input = tokenize_text(tokenizer, PROMPT)
    generated_text = generate_text(tokenizer, tokenized_input, model)

    print("Pre-Trained Generated Text:\n", generated_text)

def fine_tuned_gpt():
    """
    Fine-tune pretrained GPT2 with 'Wizard of Oz' text then generate text for PROMPT
    
    Steps:
    1. Tokenize/convert input text into tokens (ex. 'Once' -> 625)
    2. pt = PyTorch tensor = multi-dimensional array (like numpy) but also for cuda
    3. Computes probability scores of next tokens and selects with some randomness
    4. Decode generated token back into readable text
    """
    with open("data\wizard_of_oz.txt", "r") as file:
        text = file.read()

    # print(text[:100])

    tokenizer = get_tokenizer()
    model = load_pretrained_model()
    
    # Fine tune GPT2 model with 'Wizard of Oz' text
    tokenized_input = tokenize_text(tokenizer, text)
    fine_tuned_model = fine_tune(tokenized_input, model)

    # Generate text for PROMPT
    tokenized_prompt_input = tokenize_text(tokenizer, PROMPT)
    generated_text = generate_text(tokenizer, tokenized_prompt_input, fine_tuned_model)

    print("Fine-Tuned Generated Text:\n", generated_text)


if __name__ == "__main__":
    pretrained_gpt()
    fine_tuned_gpt()

    # Pre-Trained Generated Text:
    # Once upon a time in the land of Oz, he was brought into the palace of the Great Gaius, 
    # but he was soon turned aside by the Great Gaius, who asked him to return to his own city, 
    # but he refused,

    # Fine-Tuned Generated Text:
    # Once upon a time in the land of Oz, I have seen your house.

    # I am very curious to see if I can help. I found out that if I tell you, 
    # I will receive a gift of my own. I want to