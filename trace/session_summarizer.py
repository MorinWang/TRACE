"""Session summarization utilities for dataset adapters.

LongMemEval only provides raw user/assistant turns. TRACE's hierarchical graph
builder expects one information-dense paragraph per session, so this module
bridges that gap without changing the LoCoMo path.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """Summarize the following conversation session into a single paragraph.

Requirements:
- Write in third person narrative (for example, "The user discussed..." not "I discussed...")
- Include the session date ({session_date}) naturally in the text
- Capture all key events, decisions, facts, and personal details mentioned
- Preserve specific names, dates, numbers, and factual claims exactly
- Convert relative time references to absolute dates when possible based on the session date
- Include specific searchable details such as product names, amounts, locations, and people names
- Keep the summary concise but information-dense (150-300 words)
- Focus on what happened and what was discussed, not generic assistant responses

Conversation:
{formatted_turns}

Summary:"""


SHORT_SESSION_CHAR_THRESHOLD = 1000  # below this, dialogue-format directly without LLM call


class SessionSummarizer:
    """Create LoCoMo-style session summaries with a JSON cache."""

    def __init__(
        self,
        llm=None,
        cache_path: str = "cached_summaries/longmemeval_session_summaries.json",
    ):
        self.llm = llm
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, str] = self._load_cache()

    @property
    def cache(self) -> Dict[str, str]:
        return self._cache

    def summarize_session(
        self,
        turns: List[Mapping[str, str]],
        session_date: str,
        session_id: str,
    ) -> str:
        """Return a cached or newly generated summary for one session."""
        if session_id in self._cache:
            return self._cache[session_id]

        summary = self._generate_summary(turns, session_date)
        self._cache[session_id] = summary
        self.save_cache()
        return summary

    def batch_summarize(
        self,
        unique_sessions: Mapping[str, Mapping],
        batch_size: int = 20,
        limit: Optional[int] = None,
    ) -> Dict[str, str]:
        """Summarize de-duplicated sessions and flush progress per batch.

        The current implementation is sequential because the existing LLM
        controllers are synchronous and shared across TRACE.
        """
        remaining = [
            (sid, sdata)
            for sid, sdata in unique_sessions.items()
            if sid not in self._cache
        ]
        if limit is not None:
            remaining = remaining[:limit]

        for start in range(0, len(remaining), batch_size):
            batch = remaining[start:start + batch_size]
            for sid, sdata in batch:
                try:
                    turns = sdata.get("turns", [])
                    session_date = sdata.get("date", "")
                    self._cache[sid] = self._generate_summary(turns, session_date)
                except Exception as exc:
                    logger.error("Summarize failed for %s: %s", sid, exc)
            self.save_cache()
            logger.info("Summarized %d/%d remaining sessions", min(start + batch_size, len(remaining)), len(remaining))

        return self._cache

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.cache_path)

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Could not load summary cache %s: %s", self.cache_path, exc)
        return {}

    def _generate_summary(self, turns: List[Mapping[str, str]], session_date: str) -> str:
        total_chars = sum(len(str(t.get("content", ""))) for t in turns)
        if total_chars <= SHORT_SESSION_CHAR_THRESHOLD or self.llm is None:
            return self._format_as_dialogue(turns, session_date)
        return self._llm_summarize(turns, session_date)

    def _llm_summarize(self, turns: List[Mapping[str, str]], session_date: str) -> str:
        if self.llm is None:
            return self._format_as_dialogue(turns, session_date)
        prompt = SUMMARY_PROMPT.format(
            session_date=session_date or "unknown date",
            formatted_turns=self._format_turns(turns),
        )
        result = self.llm.get_completion(prompt, temperature=0.0)
        if result is None:
            logger.warning("LLM returned None for session summary, falling back to dialogue format")
            return self._format_as_dialogue(turns, session_date)
        return result.strip()

    @staticmethod
    def _format_turns(turns: Iterable[Mapping[str, str]]) -> str:
        lines = []
        for turn in turns:
            role = str(turn.get("role", "unknown")).strip() or "unknown"
            content = str(turn.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @classmethod
    def _format_as_dialogue(cls, turns: List[Mapping[str, str]], session_date: str) -> str:
        parts = [f"On {session_date}, the session contained these exchanges:"] if session_date else [
            "The session contained these exchanges:"
        ]
        for turn in turns:
            role = str(turn.get("role", "unknown")).strip().title() or "Unknown"
            content = str(turn.get("content", "")).strip()
            if content:
                parts.append(f"{role} said: {content}")
        return " ".join(parts).strip()
