import logging
import time

from google import genai
from google.genai import types

from geminiocr.config import Settings

logger = logging.getLogger("geminiocr")


class InstructionCache:
    """Manages a single Gemini context cache for the session.

    The cache is created lazily on first access and auto-refreshes
    before TTL expiry so long-running batches don't fail mid-run.
    """

    def __init__(self, client: genai.Client, settings: Settings):
        self._client = client
        self._settings = settings
        self._cache = None
        self._created_at: float = 0

    @property
    def name(self) -> str:
        """Return the cache resource name, creating/refreshing as needed."""
        if self._needs_refresh():
            self._create()
        return self._cache.name

    def _ttl_seconds(self) -> int:
        return int(self._settings.cache_ttl.rstrip("s"))

    def _needs_refresh(self) -> bool:
        if self._cache is None:
            return True
        elapsed = time.monotonic() - self._created_at
        return elapsed >= (self._ttl_seconds() - 60)

    def _create(self):
        logger.info(
            "Creating instruction cache (model=%s, ttl=%s)",
            self._settings.model,
            self._settings.cache_ttl,
        )
        self._cache = self._client.caches.create(
            model=self._settings.model,
            config=types.CreateCachedContentConfig(
                display_name="geminiocr_instructions",
                system_instruction=self._settings.system_instruction,
                ttl=self._settings.cache_ttl,
            ),
        )
        self._created_at = time.monotonic()
