"""Parsing helpers for frame-aware foreign-object video grounding."""

import re


POINT_PATTERNS = (
    re.compile(r"<point>\s*(\d+)\s*,\s*(\d+)\s*</point>", re.IGNORECASE),
    re.compile(r"\((\d+)\s*,\s*(\d+)\)"),
)


def parse_normalized_point(text):
    """Parse a valid 0..1000 point from a VLM response."""
    if not text:
        return None
    for pattern in POINT_PATTERNS:
        match = pattern.search(text)
        if match:
            point = tuple(map(int, match.groups()))
            if point != (0, 0) and all(0 <= value <= 1000 for value in point):
                return point
            return None
    return None


def parse_frame_index(text, frame_count=4):
    """Use an optional frame hint; the verified prompt defaults to the last frame."""
    if not text:
        return frame_count - 1
    match = re.search(
        r"(?i)frame(?:_num|_number)?[\"']?\s*[:#=-]?\s*[\"']?(\d+)", text
    )
    if not match:
        match = re.search(r"第\s*(\d+)\s*帧", text)
    if match:
        frame_number = int(match.group(1))
        if 1 <= frame_number <= frame_count:
            return frame_number - 1
    return frame_count - 1
