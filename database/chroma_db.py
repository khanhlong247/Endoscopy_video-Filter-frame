import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

class ChromaDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_frame(self, frame_id, embedding, metadata):
        self.collection.add(
            ids=[frame_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def get_all_frames_for_video(self, video_run_id):
        results = self.collection.get(
            where={"video_run_id": video_run_id},
            include=["embeddings", "metadatas"]
        )
        return results

db_manager = ChromaDBManager()