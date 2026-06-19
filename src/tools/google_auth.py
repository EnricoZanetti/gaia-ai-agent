"""
Shared Google OAuth2 credential manager.

All Google API integrations (Calendar, Gmail) use this single module so that
one token.json covers all required scopes.  On first run the user is directed
to a browser for consent; subsequent runs refresh the cached token silently.

Setup
-----
1. Create a project at https://console.cloud.google.com
2. Enable the Google Calendar API and Gmail API
3. Create an OAuth 2.0 Client ID (Desktop app)
4. Download the JSON file and save it as credentials.json in the project root
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# All scopes requested upfront - requesting them separately would force
# the user to re-authorize every time a new scope is added.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# Both files live in the project root (not committed - see .gitignore)
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = Path("token.json")


def get_credentials() -> Credentials:
    """
    Return valid Google OAuth2 credentials.

    Flow:
    1. Load cached token from token.json if it exists.
    2. Refresh it if expired (uses the stored refresh token - no browser).
    3. Run the full browser-based OAuth flow only if no valid token exists.
    4. Persist the resulting credentials back to token.json.
    """
    creds: Credentials | None = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "credentials.json not found. "
                    "Download it from the Google Cloud Console and place it in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())

    return creds
