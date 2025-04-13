import os
from typing import Dict, List

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from services.document_manager import DocumentManager
from services.vector_db_manager import VectorDBManager


class AIService:
    def __init__(self):
        print("🔄 Loading documents...")

        self.document_manager = DocumentManager()
        documents = self.document_manager.read_uploaded_documents()

        if not documents:
            raise ValueError(
                "No documents were loaded. Please upload files to proceed."
            )

        print("🔎 Embedding documents...")

        self.vector_db = VectorDBManager()
        embeddings, metadata = self.vector_db.compute_embeddings(documents)
        self.vector_db.metadata = metadata
        self.index = self.vector_db.build_faiss_index(embeddings)

        # Create FAISS-compatible VectorStore for LangChain retriever
        self.faiss_vectorstore = FAISS.from_texts(
            texts=[doc.page_content for doc in documents],
            embedding=HuggingFaceBgeEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            ),
            metadatas=[doc.metadata for doc in documents],
        )

        self.retriever = self.faiss_vectorstore.as_retriever(
            search_type="similarity", k=4
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=1024,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
                      You are a helpful, knowledgeable, and kind assistant. Your role is to answer user questions using the context retrieved from uploaded files and provided links.

                      Please follow these guidelines carefully:

                      1. **Always begin your response by listing the sources used** to answer the user's question. Use a clear and readable format (e.g., filenames, URLs, page numbers, etc.) before giving the actual answer.
                      2. Only use the provided context to answer the user's question. If the answer is not available in the context, respond politely and **do not** guess.
                      3. If the question requires real-time data or falls outside the scope of the context, use your available tools (e.g., web search) if permitted.
                      4. Be thorough yet concise — like a friendly, thoughtful teacher who wants the user to understand deeply.
                      5. If the user asks for clarification (e.g., "Explain this" or "What does this mean?"), provide an insightful and easy-to-understand explanation.
                      6. Always be respectful. Never respond with anything inappropriate, offensive, or irrelevant.
                  """.strip()
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    "Answer this question using the context: {question}\n\nContext:\n{context}\n\nAdditionally, include any relevant general background knowledge related to the topic in your response.",
                ),
            ]
        )

        self.chain: Runnable = self.prompt | self.llm

        self.chat_history: List[Dict[str, str]] = []

        print("✅ ChatBot is ready!")

    def chat(self, query: str):
        print("💬 Retrieving context...")

        if isinstance(self.retriever, VectorStoreRetriever):
            results_with_scores = (
                self.retriever.vectorstore.similarity_search_with_score(query, k=3)
            )
            filtered_results = [(doc, score) for doc, score in results_with_scores]

            if not filtered_results:
                print("⚠️ No results passed the similarity threshold (<= 1.0).")

            else:
                for doc, score in filtered_results:
                    print(
                        f"[{score:.4f}] {doc.metadata.get('source_file') or doc.metadata.get('source_url')}"
                    )
                    print(doc.page_content[:200], "...\n")

            retrieved_docs = [doc for doc, _ in filtered_results]

        else:
            retrieved_docs = self.retriever.get_relevant_documents(query)

        print(retrieved_docs)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(context)

        print("🧠 Generating response...")

        for chunk in self.chain.stream(
            {"question": query, "context": context, "chat_history": self.chat_history}
        ):
            print(chunk.content, end="", flush=True)

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=chunk.content))
