import cv2
import os
from datetime import datetime

def extract_frames(video_path: str, target_fps: int):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = datetime.now().strftime("%Y%md_%H%M%S")
    run_id = f"{video_name}_{timestamp_str}"
    
    output_dir = os.path.join("data", "frame", run_id)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps_goc = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps_goc / target_fps)
    
    frames_metadata = []
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            timestamp = saved_count / target_fps
            frames_metadata.append({
                "frame_path": frame_path,
                "timestamp": timestamp,
                "frame_name": frame_filename,
                "video_run_id": run_id
            })
            saved_count += 1
        count += 1

    cap.release()
    return run_id, frames_metadata