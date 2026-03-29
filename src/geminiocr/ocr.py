import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from tqdm import tqdm

from geminiocr.cache import InstructionCache
from geminiocr.config import Settings

logger = logging.getLogger("geminiocr")

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class OCRSession:
    """A session that holds the Gemini client and processes image groups.

    For a single group, calls the API directly (no cache overhead).
    For batch processing (multiple groups), creates a context cache so the
    system instruction is sent once and reused across all requests.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._client = genai.Client(api_key=self.settings.api_key)
        self._cache: InstructionCache | None = None

    def _ensure_cache(self) -> InstructionCache:
        """Lazily create the instruction cache for batch use."""
        if self._cache is None:
            self._cache = InstructionCache(self._client, self.settings)
        return self._cache

    # -- Public API --

    def process_group(
        self,
        group_id: str,
        image_paths: list[Union[str, Path]],
        use_cache: bool = False,
    ) -> dict:
        """OCR one group of images and return parsed JSON dict.

        Args:
            group_id: Identifier for this group (for logging).
            image_paths: Paths to images that belong together.
            use_cache: If True, use the shared instruction cache.
                       Automatically set to True during batch processing.
        """
        parts = _build_image_parts(image_paths)
        parts.append(types.Part(text="Extract data from these images."))
        result = self._call_with_retry(parts, use_cache=use_cache)
        logger.info("Processed group '%s': %d images", group_id, len(image_paths))
        return result

    def process_batch(
        self,
        groups: dict[str, list[Union[str, Path]]],
    ) -> dict[str, dict | Exception]:
        """Process all groups concurrently with a shared instruction cache.

        Returns a dict mapping group_id -> parsed JSON dict.
        On per-group failure, the value is the Exception instead.
        """
        if not groups:
            return {}

        # Single group: skip cache overhead
        if len(groups) == 1:
            gid, paths = next(iter(groups.items()))
            try:
                return {gid: self.process_group(gid, paths, use_cache=False)}
            except Exception as exc:
                return {gid: exc}

        # Multiple groups: use cache to save cost (unless disabled)
        use_cache = self.settings.cache_ttl is not None
        if use_cache:
            self._ensure_cache()
        total_images = sum(len(v) for v in groups.values())
        logger.info(
            "Starting batch: %d groups, %d total images (cache=%s)",
            len(groups), total_images, use_cache,
        )

        results: dict[str, dict | Exception] = {}
        try:
            asyncio.run(self._process_batch_async(groups, results, use_cache=use_cache))
        except (KeyboardInterrupt, Exception) as exc:
            succeeded = sum(1 for v in results.values() if not isinstance(v, Exception))
            logger.warning(
                "Batch interrupted: %d/%d groups completed. "
                "Returning partial results. (%s)",
                succeeded,
                len(groups),
                exc,
            )
        return results

    # -- Internals --

    async def _process_batch_async(self, groups, results, use_cache: bool = True):
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        progress = tqdm(total=len(groups), desc="Processing", unit="group")

        async def _do_one(gid, paths):
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(
                        None, self.process_group, gid, paths, use_cache
                    )
                    results[gid] = result
                except Exception as exc:
                    logger.error("Failed for group '%s': %s", gid, exc)
                    results[gid] = exc
                finally:
                    progress.update(1)

        tasks = [_do_one(gid, paths) for gid, paths in groups.items()]
        await asyncio.gather(*tasks)
        progress.close()

    def _call_with_retry(self, parts: list[types.Part], use_cache: bool) -> dict:
        """Call Gemini with exponential backoff on transient errors."""
        config_kwargs: dict = {}

        if use_cache:
            config_kwargs["cached_content"] = self._ensure_cache().name
        else:
            config_kwargs["system_instruction"] = self.settings.system_instruction

        if self.settings.response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = self.settings.response_schema

        last_exc = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.settings.model,
                    contents=[types.Content(parts=parts)],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return json.loads(response.text)
            except Exception as exc:
                last_exc = exc
                if attempt < self.settings.max_retries:
                    delay = self.settings.retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d failed (%s), retrying in %.1fs...",
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise last_exc


def _build_image_parts(image_paths: list[Union[str, Path]]) -> list[types.Part]:
    """Encode local images as inline Gemini Parts."""
    parts = []
    for path in image_paths:
        path = Path(path)
        mime = _MIME_TYPES.get(path.suffix.lower(), "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        parts.append(
            types.Part(inline_data=types.Blob(mime_type=mime, data=data))
        )
    return parts
