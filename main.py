"""
Morning Report Agent - entry point.

Usage
-----
    python main.py

On first run a browser window will open for Google OAuth authorization.
All subsequent runs refresh the cached token silently.

Configuration is read from a .env file in the project root.
See .env.example for the full list of required and optional variables.
"""

import sys

from src.agent.graph import build_graph
from src.config import UserConfig


def main() -> None:
    # Load and validate configuration before touching any API
    config = UserConfig.from_env()
    try:
        config.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    print(f"🌅  Generating morning report for {config.user_name}…")
    print(f"    Topics : {', '.join(config.news_topics)}")
    print(f"    Region : {config.location.upper()}")
    print()

    graph = build_graph()

    final_state = graph.invoke(
        {
            "user_name": config.user_name,
            "recipient_email": config.recipient_email,
            "news_topics": config.news_topics,
            "location": config.location,
            # List fields start empty; each fetch node appends to them
            "calendar_events": [],
            "email_items": [],
            "news_items": [],
            "errors": [],
            # Scalar output fields start with sentinel values
            "report_markdown": "",
            "email_sent": False,
        }
    )

    # Surface any non-fatal warnings
    if final_state.get("errors"):
        print("⚠️  Warnings:")
        for err in final_state["errors"]:
            print(f"   • {err}")
        print()

    if final_state.get("email_sent"):
        print(f"✅  Morning report sent to {config.recipient_email}")
    else:
        print("❌  Report was not sent. See warnings above.")
        # Still show the report in the terminal so the run isn't wasted
        if final_state.get("report_markdown"):
            print("\n--- Report Preview (not sent) ---")
            print(final_state["report_markdown"])


if __name__ == "__main__":
    main()
