from transformers import TrainingArguments

MODEL_NAME = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"

TRAIN_ARGS = TrainingArguments(
    output_dir="results/training_logs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="results/training_logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
