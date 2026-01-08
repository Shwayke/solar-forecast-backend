import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('IMS_API_URL')
API_KEY = os.getenv('IMS_API_KEY')