MODEL_NAME = "sankalpakc/NepaliKD-SentenceTransformers-paraphrase-multilingual-MiniLM-L12-v2"

import os

MODEL_W= "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

COLLECTION_NAME =  "STARTUPS_DEMO"


TEXT_FIELD_NAME = "document"