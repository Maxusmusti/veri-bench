from datasets import load_dataset

dataset = load_dataset("json", data_files="/new_data/data_vault/reasoning/limo_cleaned.jsonl", split="train")
# select a random subset of 50k samples
dataset = dataset.shuffle(seed=42).select(range(30))


