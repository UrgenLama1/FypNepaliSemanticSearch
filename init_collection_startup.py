import json
import os.path

from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance

from tqdm import tqdm

from constants import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, TEXT_FIELD_NAME, MODEL_NAME, MODEL_W
import numpy as np

qdrant_client = QdrantClient("http://localhost:6333")

qdrant_client.recreate_collection(
    collection_name="nepali",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

text_file_path = "../archive/Desh/Desh/1.txt"

# Read the text file and load it into payload as a list of dictionaries
with open(text_file_path, 'r', encoding='utf-8') as file:
    payload = [{"text": line.strip()} for line in file]

# Load all vectors into memory (replace with your method for loading vectors)
vectors = np.load("./nepali_vectors.npy")

# Upload to Qdrant collection
qdrant_client.upload_collection(
    collection_name="nepali",
    vectors=vectors,
    payload=payload,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256,  # How many vectors will be uploaded in a single request?
)
