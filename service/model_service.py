import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv2Singleton:
    _processor = None
    _model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_model(cls):
        if cls._model is None:
            print(f"[INIT] DINOv2 has been initialized on {cls._device}...")
            cls._processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
            cls._model = AutoModel.from_pretrained('facebook/dinov2-small')
            cls._model.to(cls._device)
            cls._model.eval()
            print("[INIT] Model loaded successfully. Ready for processing.")
        return cls._processor, cls._model, cls._device