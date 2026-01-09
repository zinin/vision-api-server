from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException, UploadFile
import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_DIMENSION: Final[int] = 8192


@dataclass(slots=True, frozen=True)
class ProcessedImage:
    """Container for processed image data."""

    image: np.ndarray
    original_filename: str
    width: int
    height: int


async def validate_and_decode_image(
        file: UploadFile,
        max_file_size: int,
        allowed_extensions: frozenset[str]
) -> ProcessedImage:
    """
    Validate and decode uploaded image file.

    Args:
        file: Uploaded file
        max_file_size: Maximum allowed file size in bytes
        allowed_extensions: Set of allowed file extensions

    Returns:
        ProcessedImage with decoded image data

    Raises:
        HTTPException: If validation fails
    """
    filename = file.filename or "unknown"

    # Validate extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    # Check Content-Length header first (if available)
    if file.size is not None and file.size > max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {max_file_size / (1024 * 1024):.1f}MB"
        )

    # Read and validate actual size
    try:
        contents = await file.read()
    finally:
        # Reset file position for potential re-reads
        await file.seek(0)

    content_length = len(contents)
    if content_length > max_file_size:
        size_mb = content_length / (1024 * 1024)
        max_mb = max_file_size / (1024 * 1024)
        logger.warning(f"File too large: {size_mb:.2f}MB (max: {max_mb:.2f}MB)")
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.2f}MB. Maximum: {max_mb:.2f}MB"
        )

    if content_length == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )

    # Decode image
    nparr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(f"Failed to decode image: {filename}")
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. File may be corrupted or not a valid image."
        )

    height, width = image.shape[:2]

    # Validate image dimensions
    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        )

    logger.info(f"Image decoded: {filename}, size: {width}x{height}")

    return ProcessedImage(
        image=image,
        original_filename=filename,
        width=width,
        height=height
    )