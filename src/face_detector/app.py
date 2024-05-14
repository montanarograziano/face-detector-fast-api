import polars as pl
import torch
from constants import DATA_DIR, MODEL_NAME, SAMPLE_SIZE
from db import (
    create_client,
    get_embeddings,
    get_payload,
    process_image,
    save_embeddings,
    search_image,
    visualize_images,
)
from model import get_model
from p_tqdm import p_umap
from PIL import Image
from pathlib import Path

client = create_client()

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model = get_model(MODEL_NAME)

# Tune sample size accordingly to your need
images: list[Path] = list(DATA_DIR.rglob("*.jpg"))[:SAMPLE_SIZE]
image_names, image_files = zip(*p_umap(process_image, images))
images_df = pl.DataFrame({"Image": image_files, "Name": image_names})
embeddings = get_embeddings(images, images_df)
payload = get_payload(images_df)

save_embeddings(client, SAMPLE_SIZE, embeddings, payload)

# Sample Image
img = Image.open(images[0])
results = search_image(img, model, client)
visualize_images(results, images)
