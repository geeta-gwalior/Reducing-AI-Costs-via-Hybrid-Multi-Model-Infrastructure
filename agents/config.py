import os
from dotenv import load_dotenv

load_dotenv()

# Configuration Load
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
GEMMA_ENDPOINT_ID = os.getenv("GEMMA_ENDPOINT_ID")

# Routing Logic Keywords
SIMPLE_KEYWORDS = ['hi', 'hello', 'hey', 'weather', 'time', 'joke', 'name']