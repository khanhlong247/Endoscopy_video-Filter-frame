import json
import os

def save_processed_frames(run_id: str, keep_frames: list):
    output_dir = os.path.join("data", "processed_frame")
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{run_id}_filtered.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(keep_frames, f, indent=4)
    print(f"[SUCCESS] Saved {len(keep_frames)} frames to {file_path}")