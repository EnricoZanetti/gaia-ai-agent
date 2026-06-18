"""
User configuration for the morning report agent.

All settings are loaded from environment variables (via a .env file).
See .env.example for the full list of supported variables.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class UserConfig:
    """Runtime configuration derived from environment variables."""

    # --- Identity ---
    user_name: str
    recipient_email: str

    # --- News preferences ---
    news_topics: list[str]
    location: str  # ISO 3166-1 alpha-2 country code, e.g. "us", "gb"

    @classmethod
    def from_env(cls) -> "UserConfig":
        """Construct a UserConfig from the current environment."""
        raw_topics = os.getenv("NEWS_TOPICS", "technology,AI")
        return cls(
            user_name=os.getenv("USER_NAME", "Friend"),
            recipient_email=os.getenv("RECIPIENT_EMAIL", ""),
            news_topics=[t.strip() for t in raw_topics.split(",") if t.strip()],
            location=os.getenv("NEWS_LOCATION", "us").lower(),
        )

    def validate(self) -> None:
        """Raise ValueError if any required configuration is missing."""
        missing = []
        if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            missing.append("ANTHROPIC_API_KEY or OPENAI_API_KEY")
        if not self.recipient_email:
            missing.append("RECIPIENT_EMAIL")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
