import os
from dotenv import load_dotenv

load_dotenv()  

class Settings:
    MODALAB_DB_URL: str = os.getenv("MODALAB_DB_URL", "")
    QDRANT_ENDPOINT: str = os.getenv("QDRANT_ENDPOINT", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "modalab_products")

settings = Settings()
