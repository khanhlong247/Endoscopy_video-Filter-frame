from utils.celery_app import app
from service.model_service import DINOv2Singleton
from database.chroma_db import db_manager
import torch
import cv2
from PIL import Image

@app.task(name="embed_and_store_frame")
def embed_and_store_frame(frame_meta: dict):
    processor, model, device = DINOv2Singleton.get_model()
    
    frame_path = frame_meta["frame_path"]
    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

    embedding_list = embedding.cpu().numpy().tolist()

    frame_id = f"{frame_meta['video_run_id']}_{frame_meta['timestamp']}"
    db_manager.add_frame(frame_id, embedding_list, frame_meta)
    
    return f"Embedded and stored frame: {frame_meta['frame_name']}"