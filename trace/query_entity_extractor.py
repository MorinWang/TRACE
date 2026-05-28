"""Layer 5: Query entity/concept extraction.

Extracts named entities and salient concepts from the query,
used to find additional seed events via participant index lookup.
"""

import json
import logging
import re
from typing import List

from trace.prompts.entity_extraction_prompt import (
    format_entity_extraction_prompt,
    parse_entity_response,
)

logger = logging.getLogger(__name__)


class QueryEntityExtractor:
    """Extracts entities and concepts from queries for graph lookup."""

    def __init__(self, llm, temperature: float = 0.0):
        self.llm = llm
        self.temperature = temperature

    def extract(self, query: str) -> List[str]:
        """Extract entities and salient concepts from a query.

        Args:
            query: The question text.

        Returns:
            List of entity/concept strings. Returns empty list on failure.
        """
        prompt = format_entity_extraction_prompt(query)

        try:
            response = self.llm.get_completion(prompt, temperature=self.temperature)
            entities = parse_entity_response(response)
            if entities:
                logger.debug(f"Extracted entities from query: {entities}")
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
