# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:34:18 2025

@author: user
"""

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import sys
from langchain_experimental.agents.agent_toolkits import create_csv_agent


class ChatBot():

    # load the dataset
    csv_file_path = "sub_warehouse_data.csv"
    loader = CSVLoader(file_path=csv_file_path,
                       encoding="utf-8", csv_args={'delimiter': ','})

    # load the api
    groq_api = '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'
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    # Download Sentence Transformers Embedding From Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    # Converting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Answer the user's questions based on the column names: 
        #,	From, To, Category,	Code, Item Name, Client, Total Quantity, Total Quality, Unit. Original question: {question}, ask a followup question"""
    )

    retriever = MultiQueryRetriever.from_llm(
        docsearch.as_retriever(),
        llm,
        prompt=QUERY_PROMPT)

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}


    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


