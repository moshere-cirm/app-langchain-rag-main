import os
import os
import shutil
import base64

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from gdrive_download import download_file
from local_loader import get_document_text

st.set_page_config(page_title="זכרון שלוניקי")


font_path = "./static/fonts/YiddishkeitAlefAlefAlef-Bold.otf"
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

with open("./zichron_s.png", "rb") as file:
    png = file.read()

# Embed SVG using custom CSS class for positioning
#st.markdown(f'<div class="logo">{svg}</div>', unsafe_allow_html=True)
# Modify the SVG positioning
# Encode the binary data to base64
encoded_png = base64.b64encode(png).decode('utf-8')

# Create the HTML string with the base64-encoded image
# Create the HTML string with the base64-encoded image
html_string = f'''
<div>
    <img src="data:image/png;base64,{encoded_png}" style="display: block; width: 100%; max-width: 300px;">
</div>
'''

# Display the image
st.markdown(html_string, unsafe_allow_html=True)

# Add a title or heading
st.markdown('''
    <title>הבוט של זכרון שלוניקי</title>
    <meta name="description" content="הבוט של זכרון שלוניקי" />
    <meta property="og:title" content="זכרון שלוניקי" />

<meta property="og:description" content="הבוט של זכרון שלוניקי" />
<meta property="og:type" content="article" />
<meta property="og:locale" content="he-il" />
    <h3 style="margin-top: 20px;">הבוט של ״זכרון שלוניקי״</h3>
    <h3>  מבוסס על הספר - בעריכת דוד א׳ רקנטי ז״ל  </h3>
    <h3>   <a href="https://drive.google.com/file/d/1IAqCbHVdex2vwVho5rMzCgJ0B0jCvnA7/view?usp=drive_link" target="_blank">זכרון שלוניקי חלק א׳</a> </h3>
      

''', unsafe_allow_html=True)
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
            with st.spinner("בעבודה:"):
                stream = ask_question(qa, prompt)
                response = st.write_stream(stream)
                #st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response})
        #message = {"role": "assistant", "content": response.content}
        #st.session_state.messages.append(message)


@st.cache_resource
def get_retriever(openai_api_key=None, google_api_token=None):
    dir_name = "pdf"

    # Create directory
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' was created successfully.")
    else:
        print(f"Directory '{dir_name}' already exists.")
    example_pdf_path_a = ("pdf/zichron_saloniki_a.pdf")

    if not os.path.exists(example_pdf_path_a):
        #download file
        download_file('1IAqCbHVdex2vwVho5rMzCgJ0B0jCvnA7', example_pdf_path_a, google_api_token)

    docs_a = get_document_text(open(example_pdf_path_a, "rb"))
    example_pdf_path_b = ("pdf/zichron_saloniki_b.pdf")
    if not os.path.exists(example_pdf_path_b):
        #download file
        download_file('1Y54RvtIAt-MhVJIy_99wW-cv1T1zmSrz', example_pdf_path_b, google_api_token)


    docs_b = get_document_text(open(example_pdf_path_b, "rb"))

    combined_docs = docs_a + docs_b


    return ensemble_retriever_from_docs(combined_docs, embeddings=None)


def get_chain(openai_api_key=None, huggingfacehub_api_token=None, google_api_token=None):
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key, google_api_token=google_api_token)
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
    google_api_token = st.session_state.get("GOOGLE_CREDENTIALS_JSON")



    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input('OPENAI_API_KEY', "OpenAI API key",
                                                 info_link="https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

        if not google_api_token:
            google_api_token = get_secret_or_input('GOOGLE_CREDENTIALS_JSON', "Google API Token",
                                                           info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if not google_api_token:
        st.warning("Missing GOOGLE_CREDENTIALS_JSON")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key, huggingfacehub_api_token=huggingfacehub_api_token, google_api_token=google_api_token)
        #st.subheader("שאל אותי ככל העולה על רוחך בנושא אלעל")
        show_ui(chain, "שלום, כיצד אוכל לעזור?")
    else:
        st.stop()


run()
