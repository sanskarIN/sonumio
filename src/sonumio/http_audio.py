"""Bounded HTTP byte-range delivery for local Sonumio WAV previews.

One immutable in-memory WAV snapshot is selected by LocalAudioStore.read()
before processing request headers. Multipart ranges are deliberately not
supported; malformed/unsupported ranges fall back to complete responses.
This service never advertises validators, so If-Range requests return 200.
"""
from __future__ import annotations

from fastapi.responses import Response


_UNSATISFIABLE = object()


def _parse_single_range(value: str, size: int) -> tuple[int, int] | object | None:
    """Return inclusive offsets, an unsatisfiable sentinel, or ignore range."""
    # Avoid expensive parsing and unbounded integer conversion on headers.
    if len(value) > 128 or "=" not in value:
        return None
    unit, spec = value.split("=", 1)
    if unit.strip().lower() != "bytes" or "," in spec:
        return None
    spec = spec.strip()
    if spec.count("-") != 1:
        return None
    first, last = spec.split("-", 1)
    if not first and not last:
        return None
    if not all(part.isascii() and part.isdecimal()
               for part in (first, last) if part):
        return None

    if not first:
        suffix_length = int(last)
        if suffix_length == 0 or size == 0:
            return _UNSATISFIABLE
        return max(0, size - suffix_length), size - 1

    start = int(first)
    if start >= size:
        return _UNSATISFIABLE
    if not last:
        return start, size - 1
    end = int(last)
    if end < start:
        return _UNSATISFIABLE
    return start, min(end, size - 1)


def wav_http_response(
    wav_bytes: bytes, *, range_header: str | None = None,
    if_range: str | None = None, head: bool = False,
) -> Response:
    """Serve a snapshot with HTTP single-range support and no caching.

    Invalid/multi-range requests receive 200 with the full representation;
    syntactically valid but unsatisfiable single ranges receive 416.
    Neither Last-Modified nor ETag is sent; If-Range is not honored.
    """
    size = len(wav_bytes)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Content-Disposition": 'attachment; filename="sonumio.wav"',
    }
    selection = (
        _parse_single_range(range_header, size)
        if not head and range_header is not None and if_range is None else None
    )
    if selection is _UNSATISFIABLE:
        headers["Content-Range"] = f"bytes */{size}"
        headers["Content-Length"] = "0"
        return Response(
            content=b"", status_code=416, media_type="audio/wav", headers=headers
        )

    if isinstance(selection, tuple):
        start, end = selection
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return Response(
            content=b"" if head else wav_bytes[start:end + 1],
            status_code=206, media_type="audio/wav", headers=headers,
        )

    headers["Content-Length"] = str(size)
    return Response(
        content=b"" if head else wav_bytes,
        status_code=200, media_type="audio/wav", headers=headers,
    )
