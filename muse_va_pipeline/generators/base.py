"""Placeholder interfaces for optional generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    output_path: Path | None
    metadata: dict


class MusicGenerator(Protocol):
    def generate(self, caption_full: str, caption_tags: str, output_dir: Path) -> GenerationResult:
        """Generate music from text captions."""


class ImageGenerator(Protocol):
    def generate(self, visual_caption: str, output_dir: Path) -> GenerationResult:
        """Generate an image from a visual caption."""


class NotConfiguredMusicGenerator:
    def generate(self, caption_full: str, caption_tags: str, output_dir: Path) -> GenerationResult:
        raise NotImplementedError(
            "Music generation is intentionally not implemented in this release. "
            "Implement MusicGenerator for your backend."
        )


class NotConfiguredImageGenerator:
    def generate(self, visual_caption: str, output_dir: Path) -> GenerationResult:
        raise NotImplementedError(
            "Image generation is intentionally not implemented in this release. "
            "Implement ImageGenerator for your backend."
        )
