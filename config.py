# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Expose environment variables as module-level constants or a dictionary
config = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "STRIPE_API_KEY": os.getenv("STRIPE_API_KEY"),
    "MYSQL_HOST": os.getenv("MYSQL_HOST"),
    "MYSQL_USER": os.getenv("MYSQL_USER"),
    "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD"),
    "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE"),
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN"),
    "MYSQL_URI": os.getenv("MYSQL_URI")
}
