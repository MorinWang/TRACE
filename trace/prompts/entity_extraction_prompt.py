"""Layer 5: Entity extraction prompt template and parser."""

import json
import re
from typing import List

PROMPT_VERSION = "entity_extraction_v1.0"

ENTITY_EXTRACTION_PROMPT = """\
Extract all person names and key concepts from the following question.

Question: {question}

Return a JSON array of strings containing:
1. All person names mentioned (e.g., "Alice", "Bob")
2. Key concepts or topics that could help find relevant information \
(e.g., "adoption", "dance studio", "tournament")

Example: ["Caroline", "LGBTQ+", "adoption"]

Entities and concepts:"""


def format_entity_extraction_prompt(question: str) -> str:
    return ENTITY_EXTRACTION_PROMPT.format(question=question)


def parse_entity_response(response: str) -> List[str]:
    """Parse entity extraction response into list of strings."""
    text = response.strip()

    # Try JSON array
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(e).strip() for e in result if str(e).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: comma-separated or line-separated
    items = re.split(r'[,\n]', text)
    return [
        item.strip().strip('"\'- ')
        for item in items
        if item.strip().strip('"\'- ')
    ]
