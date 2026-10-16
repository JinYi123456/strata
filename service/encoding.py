"""Image decoding and encoding for the HTTP boundary.

Uploads are untrusted. Two things are enforced here rather than in the handlers:

* **A byte cap before decoding.** A small file can decode to an enormous array,
  so the check has to happen on the encoded bytes, not on the result.
* **A dimension cap after decoding.** Even a legitimate-looking PNG can be far
  larger than any capture the pipeline is tuned for, and a 20-megapixel frame
  would turn a one-second scan into a minute of unexplained waiting.

Rejecting early with a specific message is better than accepting and degrading:
the operator finds out immediately what is wrong, and the service is not the
thing that becomes slow.
"""

from __future__ import annotations

import base64
import binascii
import re

import cv2
import numpy as np

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_DIMENSION = 6000
MAX_FRAMES_PER_UPLOAD = 8

_DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


class ImageError(ValueError):
    """The supplied image could not be accepted."""


def decode_data_url(value: str) -> np.ndarray:
    """Decode a base64 image (with or without a data-URL prefix) to BGR uint8."""
    if not isinstance(value, str) or not value:
        raise ImageError("empty image payload")

    payload = _DATA_URL.sub("", value.strip())
    # Base64 is 4 characters per 3 bytes; this bounds the decode before it runs.
    approx_bytes = len(payload) * 3 // 4
    if approx_bytes > MAX_UPLOAD_BYTES:
        raise ImageError(
            f"image is about {approx_bytes // (1024 * 1024)} MB, "
            f"limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError(f"image payload is not valid base64: {exc}") from exc

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageError("image exceeds the upload limit once decoded")

    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageError("could not decode the image; expected PNG or JPEG")

    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        raise ImageError(
            f"image is {w}x{h}; the longest edge must be at most {MAX_DIMENSION}"
        )
    if min(h, w) < 32:
        raise ImageError(f"image is {w}x{h}; both edges must be at least 32 px")
    return img


def decode_bytes(raw: bytes) -> np.ndarray:
    """Decode encoded image bytes to BGR uint8, with the same size limits."""
    if not raw:
        raise ImageError("empty image payload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ImageError("image exceeds the upload limit")
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageError("could not decode the image; expected PNG or JPEG")
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        raise ImageError(
            f"image is {w}x{h}; the longest edge must be at most {MAX_DIMENSION}"
        )
    return img


def decode_frames(values: list[str] | None) -> list[np.ndarray]:
    """Decode a list of payloads, enforcing the per-upload frame cap."""
    if not values:
        return []
    if len(values) > MAX_FRAMES_PER_UPLOAD:
        raise ImageError(
            f"{len(values)} frames supplied; at most {MAX_FRAMES_PER_UPLOAD} per upload"
        )
    return [decode_data_url(v) for v in values]


def encode_png(image: np.ndarray) -> bytes:
    """Encode an array to PNG bytes."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - cv2 only fails on malformed input
        raise ImageError("could not encode the image as PNG")
    return buf.tobytes()


def encode_jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    """Encode an array to JPEG bytes. Used for photographic panels, not masks."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover
        raise ImageError("could not encode the image as JPEG")
    return buf.tobytes()


def to_data_url(image: np.ndarray, *, fmt: str = "png", quality: int = 88) -> str:
    """Inline an image as a data URL, for small previews only."""
    raw = encode_png(image) if fmt == "png" else encode_jpeg(image, quality)
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Scale so the longest edge is at most ``long_edge``; never upscales."""
    h, w = image.shape[:2]
    scale = long_edge / float(max(h, w))
    if scale >= 1.0:
        return image
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=interp
    )
