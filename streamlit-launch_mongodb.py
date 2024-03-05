import streamlit as st
# import os
# from dotenv import load_dotenv, dotenv_values
from chain_mongodb import chain as rag_chroma_multi_modal_chain

st.title('🦜🔗 FellowsGPT')

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')

# load_dotenv()
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

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
    if not OPENAI_API_KEY.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and OPENAI_API_KEY.startswith('sk-'):
        generate_response(text)

# generate_response_terminal("What is the rubric for leads on a slide?")