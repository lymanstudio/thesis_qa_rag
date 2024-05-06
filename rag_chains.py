from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from typing import List
from langchain_core.documents import Document

def paper_clean_chain():
    prompt = PromptTemplate.from_template("""
    You are an editor who is an expert on editing thesis papers into rich and redundant-erased writings. Your job is to edit PAPER.
    If the client gives you PAPER(a part of the thesis paper) with PRV_PAGE(the summary of the previous page).
    To make an edited version of PAPER, you have to keep the following rules.
    1. Erase all the additional information that is not directly related to the idea and content of the paper, such as the name of a journal, page numbers, and so on.
    In most cases, the additional information is located in the first or the last part of PAPER. 
    2. Erase all the reference/citation marks of numbers in the middle of PAPER.
    3. Edit PAPER in a rich manner and should contain all the ideas and content. Do not discard any content. 
    4. It has to be related and successive to the content of PRV_PAGE. But should not repeatedly have the PRV_PAGE content.
    5. Note that successive pages are waiting to be edited, so the result should not end with the feeling that it is the last document.
    6. Do not conclude at the end of the current editing, unless PAPER only contains references(imply that current PAPER is the end of the thesis). 

    ## PRV_PAGE: {prv_page}

    ## PAPER: {content} 
    """
    )

    model = ChatOpenAI(model = 'gpt-3.5-turbo')

    return prompt | model | StrOutputParser()

def q_chain(llm) -> str:
    class response(BaseModel):
        processed_query: str = Field(description="Processed version of user input query")
        language: str = Field(description="The language that the user spoken")
    
    structured_parser = JsonOutputParser(pydantic_object=response)
    
    processing_prompt = PromptTemplate.from_template("""
    Your job is translating and converting a user input query into an easy-to-understand LLM model input query regarding CONTEXT.
    The CONTEXT is a set of metadata for a thesis paper consisting of the title, abstract, and other additional information. Its structure is a sequence of pairs of keys and values.
    
    Here is the sequence of your job:
    1. If needed(using different languages), translate the user input QUERY into the language that CONTEXT is written.
    2. Depending of the CONTEXT values, convert the user input QUERY into a question that a QA LLM model for the CONTEXT paper would be easy to comprehend
    3. OUTPUT is json format object that holds converted output and language that user speaks in QUERY.
        The converted output should go to "processed_query" key and the language go to "language" key.
    # CONTEXT:
    {context}

    # QUERY:
    {question}

    # OUTPUT:          
    """)

    return (
        {
            "context": itemgetter('context') | RunnablePassthrough(),
            "question": itemgetter('question') | RunnablePassthrough()
        }
        | processing_prompt
        | llm
        | structured_parser
    )

def a_chain(vector_store, retriever, llm):

    prompt = PromptTemplate.from_template("""

    당신의 임무는 논문에 대한 정보를 활용해 사용자가 던지는 질문에 대해 답변을 해주는 것입니다. 주어진 정보는 논문의 제목(TITLE), 논문의 초록(ABSTRACT), 질문에 대한 세부 정보를 담은 컨텍스트(CONTEXT), 그리고 논문에 대한 기타 정보(ADDITIONAL_INFO)입니다.
    답변은 CONTEXT를 기반으로 작성하되 논문의 제목과 초록을 참고하고 사용자가 이해하기 쉽게 설명해야합니다. 주어진 CONTEXT를 기반으로 답변을 찾을 수 없는 경우 "답변을 찾을 수 없습니다."라고 답변해 주세요.
    답변의 언어는 {language}로 해주세요.                          
    # TITLE:
    {title}

    # ABSTRACT:
    {abstract}
                                        
    # ADDITIONAL_INFO:
    {add_info}                

    # CONTEXT:
    {context}

    # 질문:
    {question}

    # 답변:
    """
    )

    def get_metadata(key:str) -> str: # 벡터 스토어의 첫 Document의 metadata 딕셔너리에서 key에 맞는 value를 뱉어주는 함수
        return next(iter(vector_store.docstore._dict.values())).metadata[key]

    def get_metadata_otherthen(exclude_keys:List[str]) -> str: # 벡터 스토어의 첫 Document의 metadata 딕셔너리에서 인자로 받은 key들을 제외한 다른 key들과 value 쌍을 스트링으로 뱉어주는 함수
        return "\n".join(f"{k} : {v}" for k, v in next(iter(vector_store.docstore._dict.values())).metadata.items() if k not in (exclude_keys))

    def concat_docs(docs:List[Document]) -> str: # retriever가 반환한 모든 Document들의 page_content를 하나의 단일 string으로 붙여주는 함수
        return "".join(doc.page_content for doc in docs)

    return (
        {
            "title": itemgetter('title') | RunnableLambda(get_metadata), # 입력 받은 데이터 중 title을 get_metadata함수의 인자로 넣고 반환 받은 value 값을 title로 prompt에 넣어줌
            "abstract": itemgetter('abstract') | RunnableLambda(get_metadata),
            "add_info": itemgetter('add_info') | RunnableLambda(get_metadata_otherthen),
            "context": itemgetter('question') | retriever | concat_docs, # 입력 받은 데이터 중 question을 retriever에 전달, 반환 받은 k개의 Document들을 concat_docs 함수에 전달, 내용들이 concat된 하나의 스트링을 context로 prompt에 넣어줌
            "question": itemgetter('question') | RunnablePassthrough(), # 입력 받은 데이터 중 question을 그냥 받은 형태 그대로 전달, question으로 prompt에 넣어줌
            "language": itemgetter('language') | RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )