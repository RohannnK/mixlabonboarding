from datasets import load_dataset

# Load GSM8K Dataset with 'main' configuration
dataset = load_dataset("gsm8k", "main")

# Save to JSON files
dataset["train"].to_json("data/gsm_train.json")
dataset["validation"].to_json("data/gsm_val.json")
dataset["test"].to_json("data/gsm_test.json")
from datasets import load_dataset

# Load GSM8K Dataset
dataset = load_dataset("gsm8k")

# Save to text files
dataset["train"].to_json("data/gsm_train.json")
dataset["validation"].to_json("data/gsm_val.json")
dataset["test"].to_json("data/gsm_test.json")

