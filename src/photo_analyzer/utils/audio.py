"""Audio processing utilities for metadata extraction and album-art retrieval."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from photo_analyzer.core.logger import get_logger

logger = get_logger(__name__)

# Recognised audio file extensions
AUDIO_EXTENSIONS = {
    '.mp3', '.flac', '.wav', '.aac', '.ogg', '.m4a',
    '.wma', '.opus', '.aiff', '.aif', '.alac',
}


def is_audio_file(path: Path) -> bool:
    """Return True if the file has a recognised audio extension."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


class AudioProcessor:
    """Extract metadata and optional album art from audio files using mutagen."""

    def get_audio_metadata(self, audio_path: Path) -> Dict[str, Any]:
        """
        Return a dict of technical and tag-based metadata:
        title, artist, album, year, genre, track, duration_seconds,
        bitrate, sample_rate, channels, codec.
        """
        try:
            import mutagen
            from mutagen import File as MutagenFile

            mf = MutagenFile(str(audio_path), easy=True)
            if mf is None:
                raise ValueError("mutagen could not open the file")

            def _first(tag, default=''):
                vals = mf.tags.get(tag) if mf.tags else None
                return vals[0] if vals else default

            info = mf.info if hasattr(mf, 'info') else None
            duration = round(info.length, 2) if info and hasattr(info, 'length') else 0.0
            bitrate = getattr(info, 'bitrate', None)
            sample_rate = getattr(info, 'sample_rate', None)
            channels = getattr(info, 'channels', None)

            return {
                'title': _first('title'),
                'artist': _first('artist'),
                'album': _first('album'),
                'year': _first('date') or _first('year'),
                'genre': _first('genre'),
                'track': _first('tracknumber'),
                'duration_seconds': duration,
                'bitrate_kbps': bitrate,
                'sample_rate_hz': sample_rate,
                'channels': channels,
                'codec': type(mf).__name__,
                'media_type': 'audio',
            }

        except Exception as e:
            logger.error(f"Failed to extract metadata from {audio_path}: {e}")
            return {'media_type': 'audio', 'error': str(e)}

    def extract_album_art(self, audio_path: Path) -> Optional[bytes]:
        """
        Return embedded album art as JPEG bytes, or None if absent.
        Tries MP3 ID3 APIC frames first, then the generic mutagen Picture approach
        used by FLAC, OGG, M4A, etc.
        """
        try:
            from mutagen import File as MutagenFile
            from mutagen.id3 import ID3NoHeaderError
            import mutagen.id3 as mid3
            import mutagen.mp4 as mp4
            import mutagen.flac as mflac
            import io

            mf = MutagenFile(str(audio_path))
            if mf is None:
                return None

            # ID3 (MP3 / AIFF)
            if hasattr(mf, 'tags') and mf.tags:
                for key in mf.tags.keys():
                    if key.startswith('APIC'):
                        return mf.tags[key].data

            # MP4 / M4A
            if hasattr(mf, 'tags') and mf.tags and 'covr' in mf.tags:
                cover = mf.tags['covr']
                if cover:
                    raw = bytes(cover[0])
                    # Convert to JPEG via Pillow for consistency
                    from PIL import Image
                    img = Image.open(io.BytesIO(raw)).convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    return buf.getvalue()

            # FLAC / OGG (Picture block)
            if hasattr(mf, 'pictures') and mf.pictures:
                pic = mf.pictures[0]
                return pic.data

        except Exception as e:
            logger.debug(f"No album art found in {audio_path}: {e}")

        return None

    def save_album_art_to_temp(self, audio_path: Path) -> Optional[Path]:
        """
        Extract album art and write it to a sibling temp JPEG file.
        Returns the path on success, None if no art is available.
        The caller is responsible for deleting the file when done.
        """
        import tempfile

        art_bytes = self.extract_album_art(audio_path)
        if not art_bytes:
            return None

        _, tmp_path = tempfile.mkstemp(
            suffix='.jpg', prefix=f'art_{audio_path.stem}_'
        )
        path = Path(tmp_path)
        path.write_bytes(art_bytes)
        return path
