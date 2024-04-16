import base64

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_txt_files, get_document_text
from remote_loader import get_google_doc

st.set_page_config(page_title="אני הבוט של חברת אלעל")
#st.title("אני הבוט של חברת אלעל")

font_path = "./static/fonts/PingHL-Medium.otf"
with open(font_path, "rb") as font_file:
    base64_font = base64.b64encode(font_file.read()).decode('utf-8')

# Define the custom font along with your styles
font_css = f"""
<style>
@font-face {{
    font-family: 'PingFang HL';
    src: url(data:font/opentype;base64,{base64_font}) format('opentype');
}}

html, body, .stApp {{
    direction: rtl !important;
    text-align: right !important;
    color: #1b358f !important;
    font-family: 'PingFang HL' !important;
}}

h1, h2, h3 {{
    color: #1b358f !important;
    font-family: 'PingFang HL' !important;
}}
</style>
"""

st.markdown(font_css, unsafe_allow_html=True)

with open("./logo-he-desktop.svg", "r") as file:
    svg = file.read()

# Embed SVG using Markdown
#st.markdown(f'<h1>{svg} אני הבוט של חברת אלעל</h1>', unsafe_allow_html=True)
st.markdown(f'<h1>{svg} אני הבוט של חברת אלעל</h1>', unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name,"r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#local_css("style.css")  # Assuming you have a CSS file named 'style.css'



def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("הנה התשובה:"):
                stream = ask_question(qa, prompt)
                response = st.write_stream(stream)
                #st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response})
        #message = {"role": "assistant", "content": response.content}
        #st.session_state.messages.append(message)


@st.cache_resource
def get_retriever(openai_api_key=None):
    example_pdf_path = "pdf/elal_info.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    #docs = load_txt_files()

    document_url = "https://docs.google.com/document/d/1cmxtchfuuySNTMKAs8GffdHlhpRHz1UlzkM6_ONW7yU/edit?usp=sharing"
    document_text = get_google_doc(document_url)
    #print("google document text " + document_text)
    docs = []
    doc = Document(page_content=document_text, metadata={'title': '', 'page': 1})

    docs.append(doc)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)


def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
    chain = create_full_chain(ensemble_retriever,
                              openai_api_key=openai_api_key,
                              chat_memory=StreamlitChatMessageHistory(key="langchain_messages"))
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key, huggingfacehub_api_token=huggingfacehub_api_token)
        st.subheader("שאל אותי ככל העולה על רוחך בנושא אלעל")
        show_ui(chain, "מה תרצה לדעת על חברת אלעל?")
    else:
        st.stop()


run()
