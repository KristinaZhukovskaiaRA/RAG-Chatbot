# app/services/document_manager.py

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from utils.logger import logger


class DocumentManager:
    def __init__(
        self,
        upload_folder: str = "data/documents",
        url_file: str = "data/urls/urls.txt",
    ):
        logger.info(
            f"Initializing DocumentManager with upload_folder: {upload_folder}, url_file: {url_file}"
        )
        self.upload_folder = upload_folder
        self.url_file = url_file

        if not os.path.exists(self.upload_folder):
            logger.info(f"Creating upload folder: {self.upload_folder}")
            os.makedirs(self.upload_folder)

        if not os.path.exists(os.path.dirname(self.url_file)):
            logger.info(
                f"Creating URL file directory: {os.path.dirname(self.url_file)}"
            )
            os.makedirs(os.path.dirname(self.url_file))

        if not os.path.exists(self.url_file):
            logger.info(f"Creating empty URL file: {self.url_file}")
            with open(self.url_file, "w", encoding="utf-8") as f:
                f.write("")
        logger.info("DocumentManager initialized successfully")

    def extract_documents_from_file(self, filepath: str) -> List[Document]:
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        logger.info(f"Extracting documents from file: {filepath}")

        try:
            if ext == ".pdf":
                logger.debug("Using PyPDFLoader for PDF file")
                loader = PyPDFLoader(filepath)
            elif ext == ".txt":
                logger.debug("Using TextLoader for text file")
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filepath}")
                return []

            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(filepath)
            logger.info(
                f"Successfully extracted {len(documents)} documents from {filepath}"
            )
            return documents

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []

    def read_uploaded_documents(self) -> List[Document]:
        logger.info(f"Reading documents from upload folder: {self.upload_folder}")
        all_documents = []
        for filename in os.listdir(self.upload_folder):
            filepath = os.path.join(self.upload_folder, filename)
            docs = self.extract_documents_from_file(filepath)
            all_documents.extend(docs)
        logger.info(f"Total documents read from upload folder: {len(all_documents)}")
        return all_documents

    def read_url_documents(self) -> List[Document]:
        logger.info("Reading documents from URLs...")
        all_documents = []
        try:
            with open(self.url_file, "r", encoding="utf-8") as f:
                urls = [url.strip() for url in f.readlines() if url.strip()]
            logger.info(f"Found {len(urls)} URLs to process")

            for url in urls:
                logger.info(f"Fetching documents from URL: {url}")
                url_docs = self.fetch_documents_from_url(url)
                all_documents.extend(url_docs)
        except Exception as e:
            logger.error(f"Error loading URLs: {e}")
        logger.info(f"Total documents read from URLs: {len(all_documents)}")
        return all_documents

    def read_all_documents(self) -> List[Document]:
        logger.info("Reading all documents (both uploaded and from URLs)...")
        documents = self.read_uploaded_documents()
        url_documents = self.read_url_documents()
        documents.extend(url_documents)
        logger.info(f"Total documents read: {len(documents)}")
        return documents

    def fetch_documents_from_url(self, url: str) -> List[Document]:
        logger.info(f"Fetching documents from URL: {url}")
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_url"] = url
            logger.info(f"Successfully fetched {len(docs)} documents from URL")
            return docs
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return []

    def save_url_list(self, url_list: List[str]):
        logger.info(f"Saving {len(url_list)} URLs to file")
        try:
            with open(self.url_file, "w", encoding="utf-8") as f:
                f.write("\n".join(url_list))
            logger.info("URL list saved successfully")
        except Exception as e:
            logger.error(f"Error saving URLs: {e}")

    def delete_file(self, file_path: str) -> bool:
        logger.info(f"Attempting to delete file: {file_path}")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def delete_url(self, url: str, url_list: List[str]) -> bool:
        logger.info(f"Attempting to delete URL: {url}")
        if url in url_list:
            url_list.remove(url)
            try:
                with open(self.url_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(url_list))
                logger.info(f"Successfully deleted URL: {url} from {self.url_file}")
                return True
            except Exception as e:
                logger.error(f"Error updating URL file: {e}")
                return False
        else:
            logger.warning(f"URL not found: {url}")
            return False
