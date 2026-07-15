import os
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./database/chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "endoscopy_frames")
TARGET_FPS = int(os.getenv("TARGET_FPS", 5))
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", 0.96))