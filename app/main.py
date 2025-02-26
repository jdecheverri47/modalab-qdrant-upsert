import logging
import torch
import requests
from io import BytesIO
from PIL import Image
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from sqlalchemy import text
from app.database import SessionLocal
from app.qdrant_client import get_qdrant_client, get_existing_ids
from qdrant_client import models  # To use PointStruct
from app.config import settings

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)

# Load the CLIP model and its processor once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def generate_image_embedding(image_url: str, processor: CLIPProcessor, model: CLIPModel):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    # Convert the image to RGB
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs_image = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_image)
    # Normalize the embedding
    return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

def process_products():
    logging.info("Starting the process of indexing new products...")
    db = SessionLocal()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    try:
        # Query the entire products table
        query = text("SELECT * FROM products")
        result = db.execute(query)
        products = result.fetchall()

        if not products:
            logging.info("No products found in the DB.")
            return

        # Get the IDs already indexed in Qdrant
        existing_ids = get_existing_ids(client, collection_name)
        logging.info("Existing IDs in Qdrant: %s", existing_ids)

        for row in products:
            # Convert the row to a dictionary to access fields by name
            product_data = dict(row._mapping)
            product_id = product_data.get("id")
            main_image_url = product_data.get("main_image")

            if not main_image_url:
                logging.warning("Product %s has no main image", product_id)
                continue

            if product_id in existing_ids:
                continue  # The product is already indexed

            try:
                # Generate the embedding from the "main_image" field
                embedding_tensor = generate_image_embedding(main_image_url, clip_processor, clip_model)
                # Convert the tensor to a 1D array (list of floats)
                embedding = np.array(embedding_tensor.detach().numpy()).flatten().tolist()

                # Create the PointStruct object for Qdrant
                point = models.PointStruct(
                    id=product_id,
                    vector=embedding,
                    payload=product_data  # You can filter or transform the payload as needed
                )

                # Perform the upsert for this product in Qdrant
                client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                logging.info("Product %s indexed in Qdrant", product_id)
            except Exception as e:
                logging.error("Could not index the image of product %s: %s", product_id, e)

    except Exception as e:
        logging.error("Error in the indexing process: %s", e)
    finally:
        client.close()
        db.close()

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(process_products, 'interval', hours=24)
    scheduler.start()
    logging.info("Scheduler started.")
    yield
    scheduler.shutdown()
    logging.info("Scheduler stopped.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "API running. The indexing job runs every 12 hours."}
