# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:06:37 2025

@author: user
"""

import streamlit as st
#from chat_bot import ChatBot
from excel_agent import data_agent

#bot = ChatBot()

ex_agent=data_agent()

st.set_page_config(page_title="Excel Bot Helper")


with st.sidebar:
    st.title('Excel Helper Data')
    
    
    
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, ask me about your data :)"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Exploring your data"):
            response= ex_agent.agent.invoke(input)
            st.write(response['output']) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)