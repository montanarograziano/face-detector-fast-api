import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
QDRANT_URL: str = os.environ.get("QDRANT_URL")
QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY")
MODEL_NAME: str = "facebook/dino-vits16"
DATA_DIR: Path = Path("../data/photos")
SAMPLE_SIZE: int = 200
COLLECTION_NAME: str = "images_collection"
