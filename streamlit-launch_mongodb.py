import streamlit as st
from chain_mongodb import chain as rag_chroma_multi_modal_chain
from getStreamlitIP import external_ip as ip

st.title('🏥 FellowsGPT')

OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')

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
