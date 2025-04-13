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
        self.upload_folder = upload_folder
        self.url_file = url_file

        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

        if not os.path.exists(os.path.dirname(self.url_file)):
            os.makedirs(os.path.dirname(self.url_file))

        if not os.path.exists(self.url_file):
            with open(self.url_file, "w", encoding="utf-8") as f:
                f.write("")

    def extract_documents_from_file(self, filepath: str) -> List[Document]:
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(filepath)
            elif ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                logger.warning(f"Unsupported file type: {ext} for file {filepath}")
                return []

            documents = loader.load()
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(filepath)

            return documents

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []

    def read_uploaded_documents(self) -> List[Document]:
        all_documents = []
        for filename in os.listdir(self.upload_folder):
            filepath = os.path.join(self.upload_folder, filename)
            docs = self.extract_documents_from_file(filepath)
            all_documents.extend(docs)
        return all_documents

    def read_url_documents(self) -> List[Document]:
        all_documents = []
        try:
            with open(self.url_file, "r", encoding="utf-8") as f:
                urls = [url.strip() for url in f.readlines() if url.strip()]

            for url in urls:
                url_docs = self.fetch_documents_from_url(url)
                all_documents.extend(url_docs)
        except Exception as e:
            logger.error(f"Error loading URLs: {e}")
        return all_documents

    def read_all_documents(self) -> List[Document]:
        documents = self.read_uploaded_documents()
        documents.extend(self.read_url_documents())
        return documents

    def fetch_documents_from_url(self, url: str) -> List[Document]:
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_url"] = url
            return docs
        except Exception as e:
            logger.error(f"Error loading URL {url}: {e}")
            return []

    def save_url_list(self, url_list: List[str]):
        try:
            with open(self.url_file, "w", encoding="utf-8") as f:
                f.write("\n".join(url_list))
        except Exception as e:
            logger.error(f"Error saving URLs: {e}")

    def delete_file(self, file_path: str) -> bool:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def delete_url(self, url: str, url_list: List[str]) -> bool:
        if url in url_list:
            url_list.remove(url)
            try:
                with open(self.url_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(url_list))
                logger.info(f"Deleted URL: {url} from {self.url_file}")
                return True
            except Exception as e:
                logger.error(f"Error updating URL file: {e}")
                return False
        else:
            logger.warning(f"URL not found: {url}")
            return False
