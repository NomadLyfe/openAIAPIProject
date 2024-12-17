import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load data
dataset = load_dataset('text', data_files={'train': 'training_dataset.txt'})

# Filter out empty text entries
dataset = dataset.filter(lambda example: example["text"] is not None and example["text"].strip() != "")

# Initialize tokenizer and add padding token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset and add labels
def tokenize_data(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Set labels equal to input_ids
    return tokenized

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Split into training and evaluation sets
train_dataset, eval_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1).values()

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Use the new eval_strategy argument
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=200,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add the evaluation dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

# Test the model with a sample text
input_text = "Dear Amber,"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# Move inputs to the same device as the model
device = model.device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate output
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,  # Controls randomness
    top_k=50,         # Filters to top-k tokens
    top_p=0.9,        # Nucleus sampling
    do_sample=True,   # Enable sampling
    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=True))

# # Move model to the appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Perplexity calculation
# def calculate_perplexity(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)  # Move to model's device
#     outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     return math.exp(loss.item())

# # Test with a sample
# print(calculate_perplexity("Sample text to evaluate"))
