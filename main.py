import time
from config import TARGET_FPS, COSINE_THRESHOLD
from utils.video_utils import extract_frames
from utils.filter_utils import save_processed_frames
from service.embedding_service import embed_and_store_frame
from database.chroma_db import db_manager
from service.filter_service import run_global_vectorized_filtering
from celery import group

def process_video_pipeline(video_path: str):
    print(f"--- START PIPELINE: {video_path} ---")
    
    print("[1] Extracting frames...")
    run_id, frames_metadata = extract_frames(video_path, TARGET_FPS)
    print(f"Successfully extracted {len(frames_metadata)} frames. Run ID: {run_id}")

    print("[2] Pushing embedding tasks to Celery queue...")
    job = group(embed_and_store_frame.s(meta) for meta in frames_metadata)
    result = job.apply_async()
    
    result.join()
    print("[2] Done. All vectors have been stored in ChromaDB.")

    print("[3] Fetching vectors from DB and running global filtering algorithm...")
    db_results = db_manager.get_all_frames_for_video(run_id)
    
    kept_frames = run_global_vectorized_filtering(db_results, COSINE_THRESHOLD)
    print(f"[3] Done filtering: Kept {len(kept_frames)}/{len(frames_metadata)} unique frames.")

    save_processed_frames(run_id, kept_frames)
    print("--- PIPELINE COMPLETED ---")

if __name__ == "__main__":
    video_input = "data/video/video2.mp4"
    process_video_pipeline(video_input)