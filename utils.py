import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def load_keys():
    return load_dotenv(dotenv_path=".env")

def check_docs(docs, show_len = 100, show_docs = 5, meta_show_only_keys = False):
    print(f"▶︎ No. of Documents: {len(docs)} \n\n▶︎ Contents")
    for idx, doc in enumerate(docs):
        if idx < show_docs or idx == len(docs) - 1:
            show_contet = doc.page_content[:show_len] + "..." if len(doc.page_content) > show_len else doc.page_content
                
            if idx == len(docs) - 1 and len(docs) != 1:
                print("...\n")
            print(f"* Doc {str(idx)}: {show_contet}\n※ Metadata: {doc.metadata if meta_show_only_keys == False else str([k for k in doc.metadata.keys()])}\n")
        else:
            continue

def check_docs_str(docs, show_len = 100, show_docs = 5, meta_show_only_keys = False):
    to_return = f"▶︎ No. of Documents: {len(docs)} \n\n▶︎ Contents"
    for idx, doc in enumerate(docs):
        if idx < show_docs or idx == len(docs) - 1:
            show_contet = doc.page_content[:show_len] + "..." if len(doc.page_content) > show_len else doc.page_content
                
            if idx == len(docs) - 1 and len(docs) != 1:
                to_return += "\n...\n"
            to_return += f"* Doc {str(idx)}: {show_contet}\n※ Metadata: {doc.metadata if meta_show_only_keys == False else str([k for k in doc.metadata.keys()])}\n"
        else:
            continue
    return to_return

def load_pdf(file_name):
    pdf_loader = PyMuPDFLoader(
        file_path = file_name
    )

    return pdf_loader.load()

def load_pdf_local(file_name, data_dir = "./data"):
    pdf_loader = PyMuPDFLoader(
        file_path = os.path.join(data_dir, file_name)
    )

    return pdf_loader.load()

def clean_paper(docs, chain):
    result_docs = []
    result_concat = ""
    prv_page = ""
    for i, doc in enumerate(docs):
        result = chain.invoke({
            "prv_page" : prv_page,
            "content" : doc.page_content
        })

        result_docs.append(Document(page_content = result, metadata = doc.metadata))
        result_concat += ("\n" + result)
        prv_page = result
        print(f"\t>> Page No. {i+ 1} ({i + 1}/{len(docs)}) cleaning is completed.")    

    return result_docs, result_concat

def clean_a_paper(doc, prv_page, chain):
    return chain.invoke({
            "prv_page" : prv_page,
            "content" : doc.page_content
        })


def chunk_paper(docs):
    semanticChunker = SemanticChunker(
        embeddings= OpenAIEmbeddings(),
        buffer_size = 2,
        breakpoint_threshold_type='percentile',
        breakpoint_threshold_amount=80, # 백분위수를 습관적으로 소수점으로 적는 버릇이 있는데 백분위수 그대로 적어줘야한다. 만약 90% 이상의 값에서만 나누고 싶을 경우 .9가 아닌 90으로 넣어줘야 한다.
    ) 

    chunked_docs = semanticChunker.split_documents(
        documents= docs,
    )

    return chunked_docs
