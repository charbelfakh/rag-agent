"""Extract embedded images from PDF pages (Sprint M M1)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 1024
MIN_IMAGE_DIMENSION = 50


@dataclass
class ExtractedImage:
    page: int
    index: int
    data: bytes
    width: int
    height: int
    ext: str = "png"


def extract_pdf_images(file_path: str) -> list[ExtractedImage]:
    """Return raster images extracted from a PDF (skips tiny icons)."""
    try:
        import pymupdf as fitz
    except ImportError:
        logger.warning("pymupdf unavailable — image extraction skipped")
        return []

    images: list[ExtractedImage] = []
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base = pdf.extract_image(xref)
                    data = base.get("image") or b""
                    if len(data) < MIN_IMAGE_BYTES:
                        continue
                    width = int(base.get("width") or 0)
                    height = int(base.get("height") or 0)
                    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                        continue
                    ext = base.get("ext") or "png"
                    images.append(
                        ExtractedImage(
                            page=page_num,
                            index=img_index,
                            data=data,
                            width=width,
                            height=height,
                            ext=ext,
                        )
                    )
                except Exception as exc:
                    logger.debug("Skip image p%s #%s: %s", page_num, img_index, exc)
    return images
