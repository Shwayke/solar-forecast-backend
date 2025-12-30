import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('EXTERNAL_API_URL')
API_KEY = os.getenv('EXTERNAL_API_KEY')