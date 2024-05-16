import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Scopes required to access Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Path to the OAuth 2.0 credentials JSON file
CREDENTIALS_FILE = 'client_secret_588659940013-eu1qgqoh0opo1t0ko3jbd01unogopif0.apps.googleusercontent.com.json'

# Function to authenticate and return the service
def authenticate():
    creds = None
    # Check if token.json file exists
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

# Function to download a file from Google Drive
def download_file(file_id, output_path):
    service = authenticate()
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
