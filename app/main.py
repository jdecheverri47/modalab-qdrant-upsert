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
from qdrant_client import models  # Para utilizar PointStruct
from app.config import settings

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)

# Cargar el modelo CLIP y su processor una sola vez
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def generate_image_embedding(image_url: str, processor: CLIPProcessor, model: CLIPModel):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    # Convertir la imagen a RGB
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs_image = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs_image)
    # Normaliza el embedding
    return image_features / image_features.norm(p=2, dim=-1, keepdim=True)

def process_products():
    logging.info("Iniciando proceso de indexación de productos nuevos...")
    db = SessionLocal()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    try:
        # Consulta toda la tabla de productos
        query = text("SELECT * FROM products")
        result = db.execute(query)
        products = result.fetchall()

        if not products:
            logging.info("No se encontraron productos en la DB.")
            return

        # Obtiene los IDs ya indexados en Qdrant
        existing_ids = get_existing_ids(client, collection_name)
        logging.info("IDs existentes en Qdrant: %s", existing_ids)

        for row in products:
            # Convierte la fila en un diccionario para acceder a los campos por nombre
            product_data = dict(row._mapping)
            product_id = product_data.get("id")
            main_image_url = product_data.get("main_image")

            if not main_image_url:
                logging.warning("Producto %s no tiene imagen principal", product_id)
                continue

            if product_id in existing_ids:
                continue  # El producto ya está indexado

            try:
                # Genera el embedding a partir del campo "main_image"
                embedding_tensor = generate_image_embedding(main_image_url, clip_processor, clip_model)
                # Convierte el tensor a un array 1D (lista de floats)
                embedding = np.array(embedding_tensor.detach().numpy()).flatten().tolist()

                # Crea el objeto PointStruct para Qdrant
                point = models.PointStruct(
                    id=product_id,
                    vector=embedding,
                    payload=product_data  # Puedes filtrar o transformar el payload según necesites
                )

                # Realiza el upsert para este producto en Qdrant
                client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                logging.info("Producto %s indexado en Qdrant", product_id)
            except Exception as e:
                logging.error("No se pudo indexar la imagen del producto %s: %s", product_id, e)

    except Exception as e:
        logging.error("Error en el proceso de indexación: %s", e)
    finally:
        client.close()
        db.close()

# Scheduler para ejecutar el job periódicamente (cada 10 minutos)
scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(process_products, 'interval', minutes=10)
    scheduler.start()
    logging.info("Scheduler iniciado.")
    yield
    scheduler.shutdown()
    logging.info("Scheduler detenido.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "API en ejecución. El job de indexación se ejecuta cada 10 minutos."}
