import os
from transformers import AutoTokenizer
import json

# Download the Punkt tokenizer for Nepali language
tokenizer = AutoTokenizer.from_pretrained("Sakonii/distilbert-base-nepali")

folder_path = "../archive/Desh/Desh"  # Replace with the path to your folder containing Nepali txt files
output_dict = {}

# Function to tokenize and handle long sequences
def tokenize_and_chunk(text, max_length):
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
    return [text[start:end] for start, end in zip(tokens[0::2], tokens[1::2])]

# Iterate through all files in the folder
for filename in os.listdir(folder_path)[:20]:
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # Read the content of each text file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # Tokenize Nepali text and handle long sequences
        tokenized_chunks = []
        for chunk in tokenize_and_chunk(file_content, max_length=384):
            tokenized_chunks.extend(tokenize_and_chunk(chunk, max_length=384))

        # Store the content in a dictionary with the filename as the key
        output_dict[filename] = tokenized_chunks

# Save the output_dict to a JSON file
output_json_path = "output_data.json"
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_dict, json_file, ensure_ascii=False, indent=2)

print(f"Data has been stored in {output_json_path}")
