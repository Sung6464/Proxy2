import logging
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config as Config
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
embeddings = AzureOpenAIEmbeddings(
            api_key=Config.AZURE_OPENAI_EMBEDDING_API_KEY,
            model=Config.AZURE_OPENAI_EMBEDDING_MODEL,
            use_azure=True,
            azure_endpoint=Config.AZURE_OPENAI_EMBEDDING_API_BASE,
            azure_deployment=Config.AZURE_OPENAI_EMBEDDING_MODEL,
            api_version=Config.AZURE_OPENAI_EMBEDDING_API_VERSION
            )
            