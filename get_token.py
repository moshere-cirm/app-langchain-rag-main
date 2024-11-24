from google_auth_oauthlib.flow import InstalledAppFlow
import json

# Path to your credentials file (the JSON you provided)
CREDENTIALS_FILE = "client_secrets.json"

# Scopes for Google Drive read-only access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Initialize OAuth flow
flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
creds = flow.run_local_server(port=0)

# Generate credentials JSON with token and refresh token
new_credentials = {
    "token": creds.token,
    "refresh_token": creds.refresh_token,
    "token_uri": creds.token_uri,
    "client_id": creds.client_id,
    "client_secret": creds.client_secret,
    "scopes": creds.scopes,
    "expiry": creds.expiry.isoformat() if creds.expiry else None
}

# Save new credentials to a file
with open("new_credentials.json", "w") as f:
    json.dump(new_credentials, f)

print("New credentials saved to new_credentials.json")
