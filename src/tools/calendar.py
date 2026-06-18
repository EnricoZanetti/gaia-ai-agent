"""
Google Calendar integration.

Fetches all events scheduled for today from the user's primary calendar,
ordered by start time.
"""

from datetime import datetime, timezone

from googleapiclient.discovery import build

from src.tools.google_auth import get_credentials


def get_todays_events() -> list[dict]:
    """
    Return today's calendar events from Google Calendar.

    Each event is a dict with keys:
        title       (str)  — event summary
        start       (str)  — ISO-8601 start datetime or date
        end         (str)  — ISO-8601 end datetime or date
        location    (str | None)
        description (str | None)
    """
    service = build("calendar", "v3", credentials=get_credentials())

    now = datetime.now(timezone.utc)
    # Query window: midnight → 23:59:59 today (UTC)
    time_min = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    time_max = now.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,   # expand recurring events into individual instances
            orderBy="startTime",
        )
        .execute()
    )

    events = []
    for item in result.get("items", []):
        # All-day events use "date"; timed events use "dateTime"
        start = item["start"].get("dateTime", item["start"].get("date", ""))
        end = item["end"].get("dateTime", item["end"].get("date", ""))
        events.append(
            {
                "title": item.get("summary", "(No title)"),
                "start": start,
                "end": end,
                "location": item.get("location"),
                "description": item.get("description"),
            }
        )

    return events
