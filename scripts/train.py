from transformers import TrainingArguments, Trainer

def fine_tune(model, data):
    training_args = TrainingArguments(
        # output_dir="models/fine_tuned_gpt2/",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        # logging_dir="logs/",
        save_total_limit=2,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data
    )

    trainer.train()
    
    return model