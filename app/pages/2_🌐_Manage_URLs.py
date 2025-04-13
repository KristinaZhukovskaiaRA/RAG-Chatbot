# app/pages/2_🌐_Manage_URLs.py

import os

import streamlit as st
from services.document_manager import DocumentManager
from services.vector_db_manager import VectorDBManager

st.title("🌐 Manage URLs")

document_manager = DocumentManager()
vector_db = VectorDBManager()

st.header("Add URL")
url = st.text_input("Enter URL to add", placeholder="https://example.com")

if url and st.button("Add URL"):
    with st.spinner("Processing URL..."):
        new_docs = document_manager.fetch_documents_from_url(url)

        if new_docs:
            if os.path.exists(document_manager.url_file):
                with open(document_manager.url_file, "r", encoding="utf-8") as f:
                    existing_urls = f.read().splitlines()
            else:
                existing_urls = []

            if url not in existing_urls:
                existing_urls.append(url)
                document_manager.save_url_list(existing_urls)

            documents = document_manager.read_uploaded_documents()
            documents.extend(new_docs)

            if documents:
                embeddings, metadata = vector_db.compute_embeddings(documents)
                vector_db.metadata = metadata
                vector_db.build_faiss_index(embeddings)
                st.success("URL processed successfully!")
            else:
                st.error("No valid documents found from the URL.")
        else:
            st.error("Could not fetch content from the URL.")

st.header("Current URLs")
if os.path.exists(document_manager.url_file):
    with open(document_manager.url_file, "r", encoding="utf-8") as f:
        urls = f.read().splitlines()

    if urls:
        for url in urls:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(url)
            with col2:
                if st.button("Delete", key=f"delete_{url}"):
                    if document_manager.delete_url(url, urls):
                        st.success(f"Deleted: {url}")
                        st.rerun()
    else:
        st.info("No URLs added yet.")
else:
    st.info("No URLs added yet.")
