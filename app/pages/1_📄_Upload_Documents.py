# app/pages/1_📄_Upload_Documents.py

import os

import streamlit as st
from services.document_manager import DocumentManager
from services.vector_db_manager import VectorDBManager

st.title("📄 Upload Documents")

document_manager = DocumentManager()
vector_db = VectorDBManager()

st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files to upload", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("Upload"):
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(
                    document_manager.upload_folder, uploaded_file.name
                )
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"Uploaded: {uploaded_file.name}")

            documents = document_manager.read_uploaded_documents()
            if documents:
                embeddings, metadata = vector_db.compute_embeddings(documents)
                vector_db.metadata = metadata
                vector_db.build_faiss_index(embeddings)
                st.success("Documents processed successfully!")
            else:
                st.error("No valid documents found in the uploaded files.")

st.header("Current Documents")
if os.path.exists(document_manager.upload_folder):
    files = os.listdir(document_manager.upload_folder)
    if files:
        for file in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Delete", key=f"delete_{file}"):
                    file_path = os.path.join(document_manager.upload_folder, file)
                    if document_manager.delete_file(file_path):
                        st.success(f"Deleted: {file}")
                        st.rerun()
    else:
        st.info("No documents uploaded yet.")
else:
    st.info("No documents uploaded yet.")
