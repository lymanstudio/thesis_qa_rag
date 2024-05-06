import os
from typing import List
from langchain_core.vectorstores import VectorStore #Vector Store들의 베이스 클래스
from langchain_core.documents import Document
from langchain_community.vectorstores import chroma, faiss

def get_vector_store(vector_store_model:VectorStore, documents:List[Document], embedding_model):
    return vector_store_model.from_documents(embedding = embedding_model, documents = documents)

def create_store(docs, embedding_model, vdb = 'faiss', save_store = False, save_path = None):
    if vdb == 'faiss':
        vector_store_model = faiss.FAISS
    else:
        vector_store_model = chroma
    vector_store = get_vector_store(
        vector_store_model = vector_store_model, 
        documents = docs, 
        embedding_model = embedding_model
    )

    if save_store:
        if save_path is None:
            save_path = os.path.join("./", "vector_stores/rag_faiss_index")
        vector_store.save_local(save_path)
    return vector_store

def load_store(embedding_model, load_path):
    vs = faiss.FAISS.load_local(
        folder_path= load_path,
        embeddings = embedding_model,
        allow_dangerous_deserialization = True
    )

    return vs