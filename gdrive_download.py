import json
import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import streamlit as st

# Scopes required to access Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Path to the OAuth 2.0 credentials JSON file

# Function to authenticate and return the service
def authenticate(session_state):
    creds_json = get_secret_or_input('GOOGLE_CREDENTIALS_JSON', "GDriver API key",
                                                 info_link="https://platform.openai.com/account/api-keys")

    if creds_json:
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_authorized_user_info(creds_dict, SCOPES)

        return build('drive', 'v3', credentials=creds)


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
# Function to download a file from Google Drive
def download_file(file_id, output_path, session_state):
    service = authenticate(session_state)
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(output_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f'Download {int(status.progress() * 100)}%.')
    print(f'File downloaded to {output_path}')

if __name__ == '__main__':
    # Replace 'your-file-id' with the actual file ID of the file you want to download
    FILE_ID = '1Y54RvtIAt-MhVJIy_99wW-cv1T1zmSrz'
    OUTPUT_PATH = 'zicrhon_drive_a.pdf'
    download_file(FILE_ID, OUTPUT_PATH)
