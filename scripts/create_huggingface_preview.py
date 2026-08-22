"""Create the deterministic Hugging Face preview for the final curated dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaguars.visualization.final_dataset import (
    DEFAULT_DATASET_NAME,
    configure_fiftyone_environment,
    default_runtime_paths,
    validate_runtime_paths,
)

OUTPUT_PATH = Path("artifacts/huggingface/final-curated-v1-preview.png")
GRID_SIZE = 3
TILE_SIZE = (480, 360)
HEADER_HEIGHT = 72
FOOTER_HEIGHT = 52
BACKGROUND = (12, 20, 24)
ACCENT = (255, 190, 70)
MASK_COLOR = (255, 170, 0, 88)
BOX_COLOR = (77, 223, 196)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf") if bold else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)

    return ImageFont.load_default()


def _representative_samples(dataset: Any) -> list[Any]:
    by_identity: dict[str, list[Any]] = defaultdict(list)
    for sample in dataset:
        if sample.jaguar_id is None or sample.review_required:
            continue
        by_identity[str(sample.jaguar_id)].append(sample)

    if len(by_identity) < GRID_SIZE**2:
        raise RuntimeError(f"expected at least nine known identities, found {len(by_identity)}")

    representatives = [max(samples, key=_annotation_score) for samples in by_identity.values()]
    representatives.sort(key=lambda sample: (_annotation_score(sample), str(sample.jaguar_id)), reverse=True)
    return sorted(representatives[: GRID_SIZE**2], key=lambda sample: str(sample.jaguar_id))


def _annotation_score(sample: Any) -> tuple[int, float, float, str]:
    detections = sample.segmentations_body.detections
    detection = detections[0]
    confidence = float(detection.confidence or 0.0)
    x, y, width, height = (float(value) for value in detection.bounding_box)
    margins = (x, y, 1.0 - (x + width), 1.0 - (y + height))
    clear_edges = sum(margin > 0.015 for margin in margins)
    framing = -(abs(width - 0.65) + abs(height - 0.55))
    return clear_edges, framing, confidence, sample.filepath


def _annotated_image(sample: Any) -> Image.Image:
    image = Image.open(sample.filepath).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for detection in sample.segmentations_body.detections:
        x, y, width, height = detection.bounding_box
        x0 = max(0, round(x * image.width))
        y0 = max(0, round(y * image.height))
        x1 = min(image.width, round((x + width) * image.width))
        y1 = min(image.height, round((y + height) * image.height))

        if detection.mask is not None and x1 > x0 and y1 > y0:
            mask = Image.fromarray(np.asarray(detection.mask, dtype=np.uint8) * 255)
            mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
            colored = Image.new("RGBA", (x1 - x0, y1 - y0), MASK_COLOR)
            overlay.paste(colored, (x0, y0), mask)

        overlay_draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=BOX_COLOR, width=5)

    annotated = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    metadata_height = max(1, round(annotated.height * 0.09))
    ImageDraw.Draw(annotated).rectangle(
        (0, annotated.height - metadata_height, annotated.width, annotated.height),
        fill=BACKGROUND,
    )
    return annotated


def _tile(image: Image.Image) -> Image.Image:
    tile = Image.new("RGB", TILE_SIZE, BACKGROUND)
    fitted = image.copy()
    fitted.thumbnail(TILE_SIZE, Image.Resampling.LANCZOS)
    offset = ((TILE_SIZE[0] - fitted.width) // 2, (TILE_SIZE[1] - fitted.height) // 2)
    tile.paste(fitted, offset)
    return tile


def create_preview() -> Path:
    """Create the approved preview montage and return its path."""
    paths = validate_runtime_paths(default_runtime_paths())
    configure_fiftyone_environment(paths)

    import fiftyone as fo

    dataset = fo.load_dataset(DEFAULT_DATASET_NAME)
    samples = _representative_samples(dataset)

    width = GRID_SIZE * TILE_SIZE[0]
    height = HEADER_HEIGHT + GRID_SIZE * TILE_SIZE[1] + FOOTER_HEIGHT
    preview = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(preview)
    draw.text(
        (width // 2, HEADER_HEIGHT // 2),
        "Jaguar Camera Trap — Final Curated v1",
        fill=(242, 246, 244),
        font=_font(34, bold=True),
        anchor="mm",
    )

    for position, sample in enumerate(samples):
        row, column = divmod(position, GRID_SIZE)
        preview.paste(
            _tile(_annotated_image(sample)),
            (column * TILE_SIZE[0], HEADER_HEIGHT + row * TILE_SIZE[1]),
        )

    draw.text(
        (width // 2, height - FOOTER_HEIGHT // 2),
        "Images property of Jaguar ID Project",
        fill=ACCENT,
        font=_font(22),
        anchor="mm",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preview.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"created {OUTPUT_PATH} from {len(samples)} samples across {len({sample.jaguar_id for sample in samples})} identities")
    return OUTPUT_PATH


if __name__ == "__main__":
    create_preview()
