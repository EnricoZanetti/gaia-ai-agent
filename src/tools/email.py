"""
Gmail integration for reading recent emails and sending the morning report.

Reading  - fetches metadata (sender, subject, snippet) for unread emails
           received in the last 24 hours, without downloading full bodies.
Sending  - composes and sends an HTML email via the Gmail send API.
"""

import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from googleapiclient.discovery import build

from src.tools.google_auth import get_credentials


def get_recent_emails(max_results: int = 10) -> list[dict]:
    """
    Return recent unread emails from the last 24 hours.

    Fetches only metadata (no full body download) to stay fast and
    respect privacy - the snippet field gives enough context for a briefing.

    Each item is a dict with keys:
        sender      (str)
        subject     (str)
        snippet     (str)  - short preview of the message body
        received_at (str)  - RFC 2822 date string from the Date header
    """
    service = build("gmail", "v1", credentials=get_credentials())

    result = (
        service.users()
        .messages()
        .list(userId="me", q="is:unread newer_than:1d", maxResults=max_results)
        .execute()
    )

    emails = []
    for msg in result.get("messages", []):
        detail = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msg["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"],
            )
            .execute()
        )

        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        emails.append(
            {
                "sender": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(No subject)"),
                "snippet": detail.get("snippet", ""),
                "received_at": headers.get("Date", ""),
            }
        )

    return emails


def send_email(to: str, subject: str, body_html: str) -> bool:
    """
    Send an HTML email from the authenticated Gmail account.

    Args:
        to:        Recipient email address.
        subject:   Email subject line.
        body_html: HTML body content (inner content only, no <html>/<body> tags needed).

    Returns
    -------
        True if the message was accepted by the API.
    """
    service = build("gmail", "v1", credentials=get_credentials())

    message = MIMEMultipart("alternative")
    message["To"] = to
    message["Subject"] = subject
    # Wrap the inner HTML content in a minimal but well-formed structure
    full_html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; max-width: 700px; margin: auto; padding: 20px;">
        {body_html}
      </body>
    </html>
    """
    message.attach(MIMEText(full_html, "html"))

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()

    return True
