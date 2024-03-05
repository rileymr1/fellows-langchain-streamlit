import base64
import io
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, dotenv_values

import getpass
import os
import params
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

# Delete previous os environment variables that stuck around
del os.environ['OPENAI_API_KEY']
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_ATLAS_CLUSTER_URI = os.getenv('MONGODB_CONN_STRING')

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
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("INDEX_NAME")

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Load MongoDBAtlas
vectorstore_mmembd = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_ATLAS_CLUSTER_URI,
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
