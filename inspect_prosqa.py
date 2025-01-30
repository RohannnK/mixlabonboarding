import json

# Load the validation dataset
with open('./data/prosqa_valid.json', 'r') as f:
    data = json.load(f)

# Extract a subset of the data
subset = data[:10]
print("Subset of Data:", subset)
# Extract the first 10 examples
subset = data[:10]

# Separate questions and answers
questions = [item["question"] for item in subset]
answers = [item["answer"] for item in subset]

print("Questions:")
print(questions)

print("\nAnswers:")
print(answers)
# Save the subset to a new file
with open('./data/prosqa_subset.json', 'w') as outfile:
    json.dump(subset, outfile, indent=4)

