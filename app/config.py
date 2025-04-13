# app/config.py

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT = "rag-chatbot"


settings = Settings()

# Set environment variables for Langsmith
os.environ["LANGCHAIN_TRACING"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
