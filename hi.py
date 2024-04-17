import os.path
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from constants import MODEL_NAME

# Provide the path to your text file
text_file_path = "../archive/Desh/Desh/1.txt"

model = SentenceTransformer(MODEL_NAME, device='cpu')

# Read data from the text file
with open(text_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create a DataFrame from the lines in the text file
df = pd.DataFrame({"text": lines})

# Encode vectors using Sentence Transformer
vectors = model.encode(df["text"].tolist(), show_progress_bar=True)

# Save the vectors to a NumPy file
np.save("nepali_vectors.npy", vectors, allow_pickle=False)
