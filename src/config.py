import os
from dotenv import load_dotenv

# Load .env file if it exists (for local development)
load_dotenv()

def get_api_url():
    """Get API URL from environment"""
    url = os.environ.get('IMS_API_URL')
    if not url:
        raise ValueError("IMS_API_URL environment variable is not set!")
    return url

def get_api_key():
    """Get API key from environment"""
    key = os.environ.get('IMS_API_KEY')
    if not key:
        raise ValueError("IMS_API_KEY environment variable is not set!")
    return key