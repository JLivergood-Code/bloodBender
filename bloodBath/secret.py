"""Compatibility secrets module loaded from environment and local .env."""

from pathlib import Path
import os

from dotenv import load_dotenv


def _load_from_cwd_env() -> None:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)


_load_from_cwd_env()

TIMEZONE_NAME = os.getenv("TIMEZONE_NAME") or os.getenv("BLOODBATH_TIMEZONE") or "UTC"
TCONNECT_EMAIL = os.getenv("TCONNECT_EMAIL") or os.getenv("BLOODBATH_EMAIL")
TCONNECT_PASSWORD = os.getenv("TCONNECT_PASSWORD") or os.getenv("BLOODBATH_PASSWORD")
TCONNECT_REGION = os.getenv("TCONNECT_REGION") or os.getenv("BLOODBATH_REGION") or "US"
CACHE_CREDENTIALS = (
    (os.getenv("CACHE_CREDENTIALS") or os.getenv("BLOODBATH_CACHE_CREDENTIALS") or "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)