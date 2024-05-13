import torch
from transformers import ViTModel

from face_detector import logger

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def get_model(model_name: str) -> ViTModel:
    model = ViTModel.from_pretrained(model_name).to(DEVICE)
    logger.info(f"Device used: {DEVICE}")
    logger.info(f"Model loaded: {model_name}")
    return model
