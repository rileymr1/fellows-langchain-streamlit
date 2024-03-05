import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# import params
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

# Access secrets using streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MONGODB_CONN_STRING = st.secrets["MONGODB_CONN_STRING"]
DB_NAME = st.secrets["DB_NAME"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = st.secrets["INDEX_NAME"]

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# prompt and first part of chain inspired by https://python.langchain.com/docs/modules/data_connection/retrievers/#using-retrievers-in-lcel
def rag_chain(retriever):
    """
    Multi-modal RAG chain,

    :param retriever: A function that retrieves the necessary context for the model.
    :return: A chain of functions representing the multi-modal RAG process.
    """

    template = """You like to give lots of detailed output in your answers. Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
 
    # Initialize the multi-modal Large Language Model with specific parameters
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024, openai_api_key=OPENAI_API_KEY)


    # Define the RAG pipeline
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

# initialize MongoDB python client
client = MongoClient(MONGODB_CONN_STRING)

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Load MongoDBAtlas
vectorstore_mmembd = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_CONN_STRING,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Make retriever
retriever_mmembd = vectorstore_mmembd.as_retriever()

# Create RAG chain
chain = rag_chain(retriever_mmembd) 

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)
