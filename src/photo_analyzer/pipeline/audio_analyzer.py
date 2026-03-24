"""Audio analysis pipeline.

Strategy:
  1. Extract technical metadata and ID3/tag fields via mutagen.
  2. If album art is embedded, analyse it with the Ollama vision LLM
     (same path as photo analysis) for scene/mood tags.
  3. Send a text-only prompt to the Ollama LLM with all available
     metadata fields so it can suggest additional organisational tags
     and a meaningful filename — even for audio files with no artwork.
  4. Merge results into a single structured record.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.config import get_config
from ..core.logger import get_logger
from ..analyzer.llm_client import OllamaClient
from ..utils.audio import AudioProcessor

logger = get_logger(__name__)


class AudioAnalyzer:
    """Analyse audio files using metadata extraction and an Ollama LLM."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_client = OllamaClient(self.config.llm)
        self.audio_processor = AudioProcessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_audio(
        self,
        file_path: Union[str, Path],
        session=None,
    ) -> Dict[str, Any]:
        """Analyse a single audio file and return structured results."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info(f"Analysing audio: {file_path}")

        metadata = self.audio_processor.get_audio_metadata(file_path)
        art_path: Optional[Path] = None

        try:
            art_path = self.audio_processor.save_album_art_to_temp(file_path)

            # Run vision analysis on album art and text-based tag suggestion concurrently
            art_task = (
                self.llm_client.analyze_image(art_path)
                if art_path
                else asyncio.sleep(0, result=None)
            )
            text_task = self._analyse_via_text(metadata, file_path)

            art_result, text_result = await asyncio.gather(
                art_task, text_task, return_exceptions=True
            )

            return self._merge_results(
                file_path, metadata, art_result, text_result
            )

        except Exception as e:
            logger.error(f"Failed to analyse audio {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'media_type': 'audio',
                'success': False,
                'error': str(e),
                'metadata': metadata,
            }
        finally:
            if art_path and art_path.exists():
                try:
                    art_path.unlink()
                except OSError:
                    pass

    async def analyze_batch(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int = 3,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Analyse multiple audio files with bounded concurrency."""
        semaphore = asyncio.Semaphore(batch_size)
        completed = 0

        async def _analyse_one(path):
            nonlocal completed
            async with semaphore:
                result = await self.analyze_audio(path)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(file_paths))
                return result

        tasks = [_analyse_one(p) for p in file_paths]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[Dict[str, Any]] = []
        for i, r in enumerate(raw):
            if isinstance(r, Exception):
                processed.append({
                    'file_path': str(file_paths[i]),
                    'media_type': 'audio',
                    'success': False,
                    'error': str(r),
                })
            else:
                processed.append(r)
        return processed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _analyse_via_text(
        self, metadata: Dict[str, Any], file_path: Path
    ) -> Dict[str, Any]:
        """Ask the LLM to suggest tags and a filename based on metadata text."""
        title = metadata.get('title') or file_path.stem
        artist = metadata.get('artist', '')
        album = metadata.get('album', '')
        genre = metadata.get('genre', '')
        year = metadata.get('year', '')
        duration = metadata.get('duration_seconds', 0)

        prompt = (
            "You are helping organise a music/audio library. "
            "Based only on the following audio file metadata, suggest:\n"
            "1. A JSON array of up to 10 relevant organisational tags "
            "(genre, mood, era, activity, instrument, etc.).\n"
            "2. A short descriptive filename stem (no extension, no spaces, "
            "use underscores, max 60 chars).\n\n"
            f"Title: {title}\n"
            f"Artist: {artist}\n"
            f"Album: {album}\n"
            f"Genre: {genre}\n"
            f"Year: {year}\n"
            f"Duration: {int(duration)}s\n\n"
            'Respond in JSON: {"tags": [...], "suggested_filename": "..."}'
        )

        try:
            # Use the /api/generate endpoint with no image – text-only LLM call
            import httpx, json as _json

            request_data = {
                "model": self.config.llm.primary_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300,
                },
            }

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.llm.timeout)
            ) as client:
                resp = await client.post(
                    f"{self.config.llm.ollama_url.rstrip('/')}/api/generate",
                    json=request_data,
                )
                resp.raise_for_status()
                raw_text = resp.json().get('response', '').strip()

            # Parse JSON from response
            import re
            m = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if m:
                data = _json.loads(m.group())
                return {
                    'tags': data.get('tags', []),
                    'suggested_filename': data.get('suggested_filename', ''),
                    'success': True,
                }
        except Exception as e:
            logger.warning(f"Text-based LLM tag suggestion failed: {e}")

        # Fallback: derive tags from metadata
        tags = [t for t in [genre, artist, album, str(year)] if t]
        return {
            'tags': tags[:10],
            'suggested_filename': self._make_filename(metadata, file_path),
            'success': False,
        }

    def _merge_results(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
        art_result,
        text_result,
    ) -> Dict[str, Any]:
        tags: List[str] = []
        description = ''
        suggested_filename = ''
        model_used = 'none'

        # Tags and filename from text LLM
        if isinstance(text_result, dict) and not isinstance(text_result, Exception):
            tags.extend(text_result.get('tags', []))
            suggested_filename = text_result.get('suggested_filename', '')
            if text_result.get('success'):
                model_used = self.config.llm.primary_model

        # Description and extra tags from album art vision analysis
        if isinstance(art_result, dict) and art_result and not isinstance(art_result, Exception):
            description = art_result.get('response', art_result.get('description', ''))
            # Parse tags from the vision response if present
            import json as _json, re
            m = re.search(r'"tags"\s*:\s*(\[.*?\])', description, re.DOTALL)
            if m:
                try:
                    art_tags = _json.loads(m.group(1))
                    tags.extend(art_tags)
                except Exception:
                    pass
            model_used = art_result.get('model', model_used)

        # Deduplicate while preserving order
        unique_tags = list(dict.fromkeys(t.lower() for t in tags if t))

        if not suggested_filename:
            suggested_filename = self._make_filename(metadata, file_path)

        return {
            'file_path': str(file_path),
            'media_type': 'audio',
            'success': True,
            'description': description,
            'tags': unique_tags[:20],
            'confidence': 0.7 if model_used != 'none' else 0.3,
            'suggested_filename': suggested_filename,
            'model_used': model_used,
            'metadata': metadata,
        }

    def _make_filename(self, metadata: Dict[str, Any], file_path: Path) -> str:
        parts = []
        if metadata.get('artist'):
            parts.append(metadata['artist'].lower().replace(' ', '_')[:20])
        if metadata.get('title'):
            parts.append(metadata['title'].lower().replace(' ', '_')[:30])
        if not parts:
            parts.append(file_path.stem[:40])
        raw = '_'.join(parts)
        # Keep only safe filename chars
        import re
        return re.sub(r'[^\w\-]', '_', raw).strip('_') or 'audio'
