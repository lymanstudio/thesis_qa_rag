import streamlit as st
import openai
import math
import time
from utils import *
from vectorstore import *
from rag_chains import *
from langchain_openai import OpenAIEmbeddings



def is_api_key_valid(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.embeddings.create(input = ["Hello"], model="text-embedding-3-small")
    except:
        return False
    else:
        return True
    
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
    status_ok = False

    st.set_page_config(
        page_title = "Paper RAG test",
        page_icon = ":notes:"
    )

    st.title("Paper RAG test")
    st.write("Upload your thesis paper and ask anything about it!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    with st.sidebar:
        setup_section, upload_section = st.tabs(["Settings", "Upload PDF File"])
        
        with setup_section:
            llm_type = st.radio(
                label = 'Select LLM Type',
                options = ("ChatGPT", "Other")
            )
            api_key = st.text_input("API Key", key = "chatbot_api_key", type = "password")

            if not api_key:
                st.info(f"Please input your {"OpenAI" if llm_type == "ChatGPT" else ""} API key to continue.")
                st.stop()

            if llm_type == 'ChatGPT':
                if is_api_key_valid(api_key)== True:
                    st.success("▶ API Key Status : {}".format("Good to go."))
                    add_k_select =  st.number_input(
                        label = "Select the No. of Documents",
                        min_value = 5,
                        max_value =  20,
                        step = 1,
                        key = 'k'
                    )
                    os.environ['OPENAI_API_KEY'] = api_key
                else:
                    st.error(f"▶ API Key Status: Invaild {"OpenAI" if llm_type == "ChatGPT" else ""} API key.")
                    st.stop()

            

        with upload_section:
            upload_file = st.file_uploader(
                label = "Choose a PDF file",
                accept_multiple_files=False
            )

            if upload_file is not None:
                with open(upload_file.name , mode = 'wb') as w:
                    w.write(upload_file.getvalue())

                loaded_pdf = load_pdf(file_name = upload_file.name)
                if loaded_pdf is not None:
                    st.info("PDF upload completed.")
                    st.write(f"filename : {upload_file.name}")

                    with st.expander("Show sample of uploaded documents"):
                        st.write(check_docs_str(
                                docs = loaded_pdf,
                                show_len = 100,
                                show_docs = 3,
                                meta_show_only_keys=True
                            )
                        )
                    status_ok = True
                else:
                    st.error("Uploading error. Please try again.")
                    status_ok = False
                    st.stop()

    if status_ok:
        thesis_name = upload_file.name.split('.')[0]
        vectorstore_path = os.path.join("./", "model/vectorstore/", thesis_name + "_index")

        if os.path.exists(vectorstore_path) == False:
            with st.status("Constructing a new vector store for the uploaded paper..."):
                progress_text = f"Cleaning the paper...(page no. {1} [{1}/{len(loaded_pdf)}])"
                progress_bar = st.progress(0, text = progress_text)
                
                clean_chain = paper_clean_chain()
                cleaned_paper = []
                cleaned_paper_concat = ""
                prv_page = ""
                delta = math.ceil(100/len(loaded_pdf))
                progress = 0
                for i, doc in enumerate(loaded_pdf):
                    cur_result = clean_a_paper(
                        doc = doc, 
                        prv_page = prv_page,
                        chain = clean_chain
                    )
                    progress = progress + delta if (progress + delta) <= 100 else 100
                    if i < len(loaded_pdf):
                        progress_text = f"Cleaning the documents...(page no. {i + 2} [{i + 2}/{len(loaded_pdf)}])"
                    else:
                        progress_text = f"Cleaning the documents... Done!"

                    progress_bar.progress(progress, text = progress_text)
                    cleaned_paper.append(Document(page_content = cur_result, metadata = doc.metadata))
                    cleaned_paper_concat += ("\n" + cur_result)
                
                st.write("\tChunking pages into sets of relevant sentences...")
                chunked_docs = chunk_paper(cleaned_paper)

                st.write("\tCreating a new vector store...")
                vs = create_store(
                    chunked_docs, 
                    embedding_model= OpenAIEmbeddings(), 
                    vdb= 'faiss', 
                    save_store = True, 
                    save_path = vectorstore_path
                )
            

        else:
            with st.status("Pre-constructed vector store exitsts! Load it from the local directory."):
                st.write("Setting up QA bot...")
                time.sleep(2)
                vs = load_store(
                    embedding_model=OpenAIEmbeddings(),
                    load_path= vectorstore_path
                )
                st.write("Done!")
        
        # Set Up retriever out of vector store
        retriever = vs.as_retriever(search_type = "mmr", search_kwargs = {"k": 10})

        # Make a QA Chains
        ## Q chain
        meta_data_dict = "\n".join(f"{k} : {v}" for k, v in next(iter(vs.docstore._dict.values())).metadata.items())
        query_chain = q_chain(llm = ChatOpenAI(model = 'gpt-3.5-turbo'))

        ## A chain
        answer_chain = a_chain(
            vector_store= vs,
            retriever=retriever,
            llm = ChatOpenAI(model = 'gpt-3.5-turbo')
        )
        if user_query := st.chat_input("Ask anything about your paper."):

            st.write(f"Q: {user_query}")
            with st.spinner('Wait for it...'):
                answer = answer_chain.invoke(
                    query(
                        query_chain,
                        reference=meta_data_dict,
                        q=user_query
                    )
                )
            st.write(f"A: {answer}")
            # if st.button("Quit"):
            #     stop_sign == True


if __name__ == '__main__':
    main()