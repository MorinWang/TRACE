"""Shared parsing utilities for the TRACE package."""

import re
from datetime import datetime
from typing import Optional


def try_parse_date(s: str) -> Optional[datetime]:
    """Try to parse a string as an absolute date."""
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y",
                "%Y-%m-%dT%H:%M", "%Y%m%d%H%M"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def extract_json_block(text: str) -> str:
    """Try to extract the outermost JSON object from text."""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def extract_session_num(key: str) -> int:
    """Extract session number from key like 'session_3_summary'."""
    m = re.search(r'session_(\d+)_summary', key)
    return int(m.group(1)) if m else 0
