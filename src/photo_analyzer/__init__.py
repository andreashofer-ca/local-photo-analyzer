"""Local Media Analyzer - Secure local LLM-based media analysis and organization."""

__version__ = "0.2.0"
__author__ = "Andreas Hofer"
__description__ = "Secure local LLM-based media analyzer for photos, videos, and audio"

from .core.config import Config, get_config
from .core.logger import get_logger, setup_logging
from .pipeline.analyzer import PhotoAnalyzer
from .pipeline.processor import PhotoProcessor
from .pipeline.organizer import PhotoOrganizer
from .pipeline.video_analyzer import VideoAnalyzer
from .pipeline.audio_analyzer import AudioAnalyzer

__all__ = [
    'Config',
    'get_config',
    'get_logger',
    'setup_logging',
    'PhotoAnalyzer',
    'PhotoProcessor',
    'PhotoOrganizer',
    'VideoAnalyzer',
    'AudioAnalyzer',
]