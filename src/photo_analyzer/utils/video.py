"""Video processing utilities for frame extraction and metadata."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from photo_analyzer.core.logger import get_logger

logger = get_logger(__name__)

# Recognised video file extensions
VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv',
    '.flv', '.webm', '.3gp', '.mts', '.m2ts', '.ts',
}


def is_video_file(path: Path) -> bool:
    """Return True if the file has a recognised video extension."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


class VideoProcessor:
    """Extract frames and metadata from video files using OpenCV."""

    def __init__(self, frames_to_extract: int = 5):
        self.frames_to_extract = frames_to_extract

    def get_video_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract duration, FPS, resolution, and codec from a video file."""
        try:
            import cv2  # opencv-python is already a project dependency

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = ''.join(
                chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
            ).strip('\x00')
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()

            return {
                'fps': round(fps, 3),
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'codec': codec,
                'duration_seconds': round(duration, 2),
                'media_type': 'video',
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata from {video_path}: {e}")
            return {'media_type': 'video', 'error': str(e)}

    def extract_key_frames(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        n_frames: Optional[int] = None,
    ) -> List[Path]:
        """
        Extract N evenly-spaced key frames and save them as JPEG files.

        Returns a list of paths to the extracted frame images.
        A temporary directory is created when *output_dir* is not specified;
        the caller is responsible for cleaning it up.
        """
        import cv2

        n = n_frames or self.frames_to_extract

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video for frame extraction: {video_path}"
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        # Pick n evenly-spaced frame indices across the full video
        indices = [
            int(i * (total_frames - 1) / max(n - 1, 1))
            for i in range(n)
        ]

        if output_dir is None:
            tmp = tempfile.mkdtemp(prefix='photo_analyzer_video_')
            output_dir = Path(tmp)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: List[Path] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    f"Could not read frame {idx} from {video_path}"
                )
                continue

            frame_path = output_dir / f"{video_path.stem}_frame{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)

        cap.release()
        logger.info(
            f"Extracted {len(frame_paths)}/{n} frames from {video_path}"
        )
        return frame_paths
