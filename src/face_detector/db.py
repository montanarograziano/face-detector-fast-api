from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from constants import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL
from IPython.core.display_functions import display
from model import DEVICE
from p_tqdm import t_map
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import ScoredPoint
from transformers import ViTModel

from face_detector import logger


def create_client() -> QdrantClient:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    if client.get_collection(collection_name=COLLECTION_NAME):
        logger.info("Collection already exists. Skip creating.")
    else:
        logger.info(f"Creating collection {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
        )
    return client


def process_image(filename: Path) -> tuple[str, Image.Image]:
    img_file = Image.open(filename)
    img_copy = deepcopy(img_file)  # Workaround for OSError
    img_file.close()
    return str(filename).split(".")[0], img_copy


def get_embeddings_from_image(model: ViTModel, item) -> np.ndarray:
    inputs = model(images=item, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return outputs


def get_embeddings(model, images_df: pl.DataFrame) -> list[np.ndarray]:
    logger.info("Loading images")
    helper = partial(get_embeddings_from_image, model)
    embeddings = t_map(helper, images_df["Image"])

    return embeddings


def get_payload(images_df: pl.DataFrame) -> list[dict[str, int | tuple[int, Any]]]:
    payload = []
    for i, name in enumerate(images_df["Name"]):
        payload.append({"image_id": i, "name": name})
    return payload


def save_embeddings(
    client: QdrantClient,
    sample_size: int,
    embeddings: list[np.ndarray],
    payload: list[dict[str, int | tuple[int, Any]]],
) -> None:
    logger.info("Saving embeddings")
    for i in range(sample_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(ids=[i], vectors=embeddings[i], payloads=[payload[i]]),
        )

    # check if the update is successful
    client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )
    # To visually inspect the collection we just created,
    # we can scroll through our vectors with the client.scroll() method.
    client.scroll(collection_name=COLLECTION_NAME, limit=10)


def visualize_images(results, images, top_k=2):
    for i in range(top_k):
        image_id = results[i].payload["image_id"]
        name = results[i].payload["name"]
        score = results[i].score
        image = Image.open(images[image_id])

        logger.info(
            f"Result #{i + 1}: {name} was diagnosed with {score * 100} confidence"
        )
        logger.info(f"This image score was {score}")
        display(image)


def search_image(
    img: Image.Image, model: ViTModel, client: QdrantClient
) -> list[ScoredPoint]:
    image_embeddings = model(images=img, return_tensors="pt")
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=image_embeddings.mean(dim=1)[0].tolist(),
        limit=5,
        with_payload=True,
    )
    return results
