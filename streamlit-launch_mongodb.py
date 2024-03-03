import streamlit as st
from chain_mongodb import chain as rag_chroma_multi_modal_chain

st.title('🦜🔗 FellowsGPT')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

def generate_response(input_text):
    llm = rag_chroma_multi_modal_chain
    st.info(llm.invoke(input_text))

# For debugging without launching to streamlit
def generate_response_terminal(input_text):
    llm = rag_chroma_multi_modal_chain
    print(llm.invoke(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What is the rubric for leads on a slide? Be detailed in your response.')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)

generate_response_terminal("What is the rubric for leads on a slide?")

import os
print(os.environ("OPENAI_API_KEY"))