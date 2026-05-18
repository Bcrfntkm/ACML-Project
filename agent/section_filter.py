"""Filter out non-content sections from parsed chemistry paper paragraphs."""
from __future__ import annotations
import re

# Section headers that indicate noise content to skip
NOISE_SECTION_PATTERNS = [
    r"^references?\b",
    r"^acknowledgements?\b",
    r"^acknowledgments?\b",
    r"^conflicts?\s+of\s+interest\b",
    r"^competing\s+interests?\b",
    r"^author\s+(contributions?|information)\b",
    r"^supporting\s+information\b",
    r"^associated\s+content\b",
    r"^notes\b",
    r"^abbreviations?\b",
    r"^funding\b",
    r"^data\s+availability\b",
    r"^ethics\s+(statement|approval)\b",
    r"^declaration\s+of",
    r"^orcid\b",
    r"^correspondence\b",
    r"^present\s+address\b",
]

_NOISE_RE = re.compile(
    "|".join(NOISE_SECTION_PATTERNS),
    re.IGNORECASE
)


def is_noise_paragraph(paragraph: str) -> bool:
    """Return True if this paragraph is a noise section header or its content."""
    stripped = paragraph.strip()
    # Check if the paragraph starts with a noise section header
    first_line = stripped.split("\n")[0].strip()
    return bool(_NOISE_RE.match(first_line))


def filter_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Remove noise sections from a list of paragraphs.
    Once a noise section header is encountered, all subsequent paragraphs
    in that section are also dropped (until the next non-noise section).
    Returns filtered list of content paragraphs.
    """
    filtered: list[str] = []
    in_noise_section = False

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        first_line = stripped.split("\n")[0].strip()

        if _NOISE_RE.match(first_line):
            in_noise_section = True
            continue

        # Heuristic: if we see a new ALL-CAPS or Title-Case section header
        # that is NOT a noise section, we exit noise mode
        if in_noise_section:
            # A new content section header resets noise mode
            # Detect section headers: short lines (< 80 chars) that look like titles
            if len(first_line) < 80 and (first_line.isupper() or first_line.istitle() or re.match(r"^[A-Z][A-Z\s]+$", first_line)):
                in_noise_section = False
                filtered.append(para)
            # else: still in noise section, skip
        else:
            filtered.append(para)

    return filtered
