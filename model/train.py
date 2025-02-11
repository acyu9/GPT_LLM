from transformers import TrainingArguments, Trainer
from datasets import Dataset

def token_to_dataset(inputs):
    # Convert tokenized input into Dataset format
    dataset = Dataset.from_dict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": inputs["input_ids"]
    })
    return dataset

def fine_tune(inputs, model):
    dataset = token_to_dataset(inputs)

    training_args = TrainingArguments(
        output_dir="data/fine_tuned_gpt2/",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        # logging_dir="logs/",
        save_total_limit=2,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    
    return model