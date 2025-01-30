import json

# Path to save the subset
subset_path = "./results/prosqa_subset.json"

# Example subset data (use your actual subset if available)
data_subset = [
    {
        "question": "Every shumpus is a rempus. Is Tom a lempus or scrompus?",
        "answer": "Tom is a lempus."
    },
    {
        "question": "Sally is a zhorpus. Is Sally a fompus or worpus?",
        "answer": "Sally is a worpus."
    }
]

# Save the subset to a JSON file
with open(subset_path, "w") as f:
    json.dump(data_subset, f, indent=4)

print(f"Subset saved to {subset_path}")

