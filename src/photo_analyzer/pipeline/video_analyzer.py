"""Video analysis pipeline.

Extracts representative key frames from a video file and uses the
existing Ollama vision LLM to analyse each frame.  Results from all
frames are then aggregated into a single, video-level description.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_config
from ..core.logger import get_logger
from ..analyzer.llm_client import OllamaClient
from ..utils.video import VideoProcessor

logger = get_logger(__name__)


class VideoAnalyzer:
    """Analyse a video by sampling key frames and running them through the vision LLM."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_client = OllamaClient(self.config.llm)
        self.video_processor = VideoProcessor(
            frames_to_extract=getattr(
                self.config.analysis, 'video_frames_to_extract', 5
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_video(
        self,
        file_path: Union[str, Path],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """Analyse a single video file and return aggregated results."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        logger.info(f"Analysing video: {file_path}")

        # Extract technical metadata (codec, fps, duration …)
        metadata = self.video_processor.get_video_metadata(file_path)

        # Extract frames into a disposable temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix='photo_analyzer_video_'))
        try:
            frame_paths = self.video_processor.extract_key_frames(
                file_path, output_dir=tmp_dir
            )

            if not frame_paths:
                return {
                    'file_path': str(file_path),
                    'media_type': 'video',
                    'success': False,
                    'error': 'No frames could be extracted from the video',
                    'metadata': metadata,
                }

            # Analyse each frame – reuse the existing batch helper
            frame_results = await self.llm_client.analyze_batch(
                frame_paths,
                max_concurrent=3,
            )

            return self._aggregate_frame_results(
                frame_results, file_path, metadata
            )

        except Exception as e:
            logger.error(f"Failed to analyse video {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'media_type': 'video',
                'success': False,
                'error': str(e),
                'metadata': metadata,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def analyze_batch(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int = 3,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Analyse multiple video files with bounded concurrency."""
        semaphore = asyncio.Semaphore(batch_size)
        completed = 0

        async def _analyse_one(path):
            nonlocal completed
            async with semaphore:
                result = await self.analyze_video(path)
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
                    'media_type': 'video',
                    'success': False,
                    'error': str(r),
                })
            else:
                processed.append(r)
        return processed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_frame_results(
        self,
        frame_results: List[Dict[str, Any]],
        video_path: Path,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine per-frame LLM results into a single video-level record."""
        successful = [
            r for r in frame_results
            if not r.get('error') and r.get('success', True)
        ]

        if not successful:
            return {
                'file_path': str(video_path),
                'media_type': 'video',
                'success': False,
                'error': 'All frame analyses failed',
                'metadata': metadata,
            }

        # Use the longest description as the primary one
        descriptions = [
            r.get('response', r.get('description', '')) for r in successful
        ]
        primary_description = max(descriptions, key=len, default='')

        # Collect unique tags (preserve insertion order, lower-case dedup)
        all_tags: List[str] = []
        for r in successful:
            all_tags.extend(r.get('tags', []))
        unique_tags = list(dict.fromkeys(t.lower() for t in all_tags if t))

        # Average confidence across frames
        confidences = [r.get('confidence', 0.0) for r in successful]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        model_used = successful[0].get('model', successful[0].get('model_used', 'unknown'))

        return {
            'file_path': str(video_path),
            'media_type': 'video',
            'success': True,
            'description': primary_description,
            'tags': unique_tags[:20],
            'confidence': round(avg_confidence, 3),
            'frames_analysed': len(successful),
            'suggested_filename': self._make_filename(
                primary_description, unique_tags, video_path
            ),
            'model_used': model_used,
            'metadata': metadata,
        }

    def _make_filename(
        self,
        description: str,
        tags: List[str],
        video_path: Path,
    ) -> str:
        """Derive a short descriptive filename stem from description/tags."""
        stop = {
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'of',
            'with', 'is', 'are', 'was', 'video', 'clip', 'showing', 'shows',
        }
        words = [
            w.strip('.,!?')
            for w in description.lower().split()
            if len(w) > 2 and w.strip('.,!?') not in stop
        ][:4]

        if not words and tags:
            words = tags[:3]
        if not words:
            words = [video_path.stem[:20]]

        return '_'.join(words) or 'video'
