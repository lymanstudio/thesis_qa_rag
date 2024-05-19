import os
import time

from utils import *
from vectorstore import *
from rag_chains import *
from langchain_openai import OpenAIEmbeddings

def query(q_chain, reference, q: str, params : dict = None):
    question = q_chain.invoke(
        {
            "context": reference,
            "question": q  
        }
    )
    
    if params is None:
        params = {
            "title": "title",
            "abstract": "subject",
            "add_info": ['title', 'subject'],
        } 
    print(question)
    return {
        "title": params["title"],
        "abstract": params["abstract"],
        "add_info": params["add_info"],
        "context": question["processed_query"],
        "question": question["processed_query"],
        "language": question["language"]
    }

def main():

    

    return True

if __name__ == '__main__':
    key_status = load_keys()
    print("▶ API Key status : {}".format("Good to go." if key_status else "Invaild or missing API key."))
    if key_status:
        main()
    else:
        print("exit")



file_name = "framework_for_indoor_elements_classification_via_inductive_learning_on_floor_plan_graphs.pdf"
loaded_pdf = load_pdf_local(file_name = file_name)
thesis_name = file_name.split('.')[0]
vectorstore_path = os.path.join("./", "model/vectorstore/", thesis_name + "_index")

if os.path.exists(vectorstore_path) == False:
    print("▶ Constructing a new vector store for the uploaded paper.")
    print("\t* Cleaning the paper...")

    cleaned_paper, cleaned_paper_concat = clean_paper(loaded_pdf, chain = paper_clean_chain())
    # check_docs(cleaned_paper)
    
    print("\t* Chunking pages into sets of relevant sentences...")
    chunked_docs = chunk_paper(cleaned_paper)
    # check_docs(chunked_docs)

    print("\t* Creating a new vector store...")
    vs = create_store(
        chunked_docs, 
        embedding_model= OpenAIEmbeddings(), 
        vdb= 'faiss', 
        save_store = True, 
        save_path = vectorstore_path
    )
    

else:
    print("▶ Pre-constructed vector store exitsts! Load it from the local directory.")
    vs = load_store(
        embedding_model=OpenAIEmbeddings(),
        load_path= vectorstore_path
    )

print("▶ Setting up QA bot...")
# Set Up retriever out of vector store
retriever = vs.as_retriever(search_type = "mmr", search_kwargs = {"k": 10})

# Make a QA Chains
## Q chain
meta_data_dict = "\n".join(f"{k} : {v}" for k, v in next(iter(vs.docstore._dict.values())).metadata.items())
query_chain = q_chain(llm = ChatOpenAI(model = 'gpt-3.5-turbo'))

## A chain
paper_qa_chain = a_chain(
    vector_store= vs,
    retriever=retriever,
    llm = ChatOpenAI(model = 'gpt-3.5-turbo')
)



paper_qa_chain.invoke(
    query(
        query_chain,
        reference=meta_data_dict,
        q="既存の論文との違いは何？"
    )
)