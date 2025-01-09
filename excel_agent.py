# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:24:00 2025

@author: user
"""

import pandas as pd 
import os
from langchain_groq import ChatGroq
import warnings
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import sys



class data_agent():
    os.chdir('C:\\Users\\user\\Desktop\\Chat_Bot_Mindsets')


    groq_api = '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'
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)


    excel_path="Warehouse_Data_Summary.xlsx"

    df = pd.read_excel(excel_path)
    del df['Code']

    df['Total Quantity']=abs(df['Total Quantity'])
    df['Total Quality']=abs(df['Total Quality'])
    df.columns=['#', 'From', 'To', 'Category', 'Item Name', 'Client', 'Total Quantity Sold','Total Sales (JOR)', 'Unit']

    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
    
x=data_agent()
c=x.agent.invoke('What is the total number of rows?')
    

#def query_data(query):

   #response = agent.invoke(query)
 #   return response['output']

#query_data('What is the total number of rows?')





