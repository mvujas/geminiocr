import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_SYSTEM_INSTRUCTION = (
    "Extract all text and data from the provided images. "
    "Return ONLY valid JSON matching the requested schema."
)


@dataclass
class Settings:
    """Configuration for a geminiocr session.

    Resolution order (highest wins): explicit kwargs > env vars > defaults.
    """

    api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    )
    cache_ttl: str | None = field(
        default_factory=lambda: os.environ.get("GEMINI_CACHE_TTL", "3600s") or None
    )
    system_instruction: str = ""
    response_schema: dict[str, Any] | None = None
    max_retries: int = 3
    retry_delay: float = 2.0
    concurrency: int = 5

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it as an environment variable, "
                "put it in a .env file, or pass api_key= to Settings()."
            )
        if not self.system_instruction:
            self.system_instruction = _DEFAULT_SYSTEM_INSTRUCTION
