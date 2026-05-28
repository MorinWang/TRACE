"""Dataset adapters for TRACE experiments.

The existing TRACE pipeline is LoCoMo-shaped: it expects session summaries,
conversation turns, and integer categories. This module keeps that contract
while adding LongMemEval support in new code paths only.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from load_dataset import load_locomo_dataset
from trace.session_summarizer import SessionSummarizer


CATEGORY_MAP = {
    "single-session-user": 1,
    "single-session-assistant": 1,
    "single-session-preference": 1,
    "multi-session": 2,
    "temporal-reasoning": 3,
    "knowledge-update": 4,
}


def parse_longmemeval_date(value: str):
    """Return a sortable datetime for LongMemEval timestamps."""
    if not value:
        return datetime.min
    cleaned = re.sub(r"\s+\([A-Za-z]{3}\)", "", value.strip())
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return datetime.min


def normalize_lme_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role == "assistant":
        return "Assistant"
    if role == "user":
        return "User"
    return role.title() if role else "Unknown"


def format_turn_for_memory(turn: Mapping[str, str]) -> str:
    return f"Speaker {normalize_lme_role(str(turn.get('role', '')))} says: {turn.get('content', '')}"


class DatasetAdapter(ABC):
    @abstractmethod
    def get_num_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_session_summaries(self, sample_idx: int) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def get_qa_pairs(self, sample_idx: int) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def get_conversation_turns(self, sample_idx: int) -> List[Tuple[str, List[Dict]]]:
        raise NotImplementedError

    @abstractmethod
    def get_session_metadata(self, sample_idx: int) -> Dict[str, Dict]:
        raise NotImplementedError


class LoCoMoAdapter(DatasetAdapter):
    """Thin adapter around the existing LoCoMo loader."""

    def __init__(self, dataset_path: str):
        self.dataset = load_locomo_dataset(dataset_path)

    def get_num_samples(self) -> int:
        return len(self.dataset)

    def get_session_summaries(self, sample_idx: int) -> Dict[str, str]:
        return dict(getattr(self.dataset[sample_idx], "session_summary", {}) or {})

    def get_qa_pairs(self, sample_idx: int) -> List[Dict]:
        pairs = []
        for qa in self.dataset[sample_idx].qa:
            pairs.append({
                "question": qa.question,
                "answer": qa.final_answer,
                "category": qa.category,
                "original_question_type": f"locomo_category_{qa.category}",
                "question_date": None,
            })
        return pairs

    def get_conversation_turns(self, sample_idx: int) -> List[Tuple[str, List[Dict]]]:
        result = []
        for _, session in sorted(self.dataset[sample_idx].conversation.sessions.items()):
            turns = [
                {"role": turn.speaker, "content": turn.text}
                for turn in session.turns
            ]
            result.append((session.date_time, turns))
        return result

    def get_session_metadata(self, sample_idx: int) -> Dict[str, Dict]:
        sample = self.dataset[sample_idx]
        metadata = {}
        for key in sample.session_summary:
            match = re.search(r"session_(\d+)", key)
            session_num = int(match.group(1)) if match else None
            session = sample.conversation.sessions.get(session_num) if session_num is not None else None
            metadata[key] = {
                "date_time": getattr(session, "date_time", ""),
                "session_id": session_num,
                "session_key": key,
            }
        return metadata


class LongMemEvalAdapter(DatasetAdapter):
    """Adapter for LongMemEval's one-question-per-record format."""

    def __init__(
        self,
        dataset_path: str,
        summarizer: Optional[SessionSummarizer] = None,
        sample_idx: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        self.dataset_path = Path(dataset_path)
        with self.dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"LongMemEval dataset must be a list: {dataset_path}")

        if sample_idx is not None:
            data = [data[sample_idx]]
        elif limit is not None:
            data = data[:limit]

        self.data: List[Dict] = data
        self.summarizer = summarizer
        self.unique_sessions = self._collect_unique_sessions(self.data)
        self.sorted_sessions = self._sort_sessions(self.unique_sessions)
        self.sid_to_idx = {
            sid: i
            for i, (sid, _) in enumerate(self.sorted_sessions)
        }

    def get_num_samples(self) -> int:
        return len(self.data)

    def get_session_summaries(self, sample_idx: int) -> Dict[str, str]:
        item = self.data[sample_idx]
        summaries = {}
        for sid in item.get("haystack_session_ids", []):
            key = self.summary_key_for_sid(sid)
            session = self.unique_sessions[sid]
            summaries[key] = self._summary_for_session(sid, session)
        return summaries

    def get_all_session_summaries(self) -> Dict[str, str]:
        summaries = {}
        for sid, session in self.sorted_sessions:
            summaries[self.summary_key_for_sid(sid)] = self._summary_for_session(sid, session)
        return summaries

    def get_qa_pairs(self, sample_idx: int) -> List[Dict]:
        return [self._qa_pair(self.data[sample_idx])]

    def get_all_qa_pairs(self) -> List[Dict]:
        return [self._qa_pair(item) for item in self.data]

    def get_conversation_turns(self, sample_idx: int) -> List[Tuple[str, List[Dict]]]:
        item = self.data[sample_idx]
        result = []
        for date, turns in zip(item.get("haystack_dates", []), item.get("haystack_sessions", [])):
            result.append((date, list(turns)))
        return result

    def get_all_conversation_turns(self) -> List[Tuple[str, str, List[Dict]]]:
        return [
            (sid, session["date"], session["turns"])
            for sid, session in self.iter_sorted_sessions()
        ]

    def get_session_metadata(self, sample_idx: int) -> Dict[str, Dict]:
        item = self.data[sample_idx]
        return {
            self.summary_key_for_sid(sid): self._metadata_for_sid(sid)
            for sid in item.get("haystack_session_ids", [])
        }

    def get_all_session_metadata(self) -> Dict[str, Dict]:
        return {
            self.summary_key_for_sid(sid): self._metadata_for_sid(sid)
            for sid, _ in self.sorted_sessions
        }

    def iter_sorted_sessions(self) -> Iterable[Tuple[str, Dict]]:
        return iter(self.sorted_sessions)

    def summary_key_for_sid(self, sid: str) -> str:
        return f"session_{self.sid_to_idx[sid] + 1}_summary"

    def session_node_id_for_sid(self, sid: str) -> str:
        return f"lme_session_{self.sid_to_idx[sid] + 1}"

    def original_sid_for_summary_key(self, key: str) -> Optional[str]:
        match = re.search(r"session_(\d+)_summary", key)
        if not match:
            return None
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= len(self.sorted_sessions):
            return None
        return self.sorted_sessions[idx][0]

    @staticmethod
    def subset_tag(sample_idx: Optional[int] = None, limit: Optional[int] = None) -> str:
        if sample_idx is not None:
            return f"sample_{sample_idx}"
        if limit is not None:
            return f"limit_{limit}"
        return "global"

    @staticmethod
    def _collect_unique_sessions(items: Sequence[Mapping]) -> "OrderedDict[str, Dict]":
        sessions: "OrderedDict[str, Dict]" = OrderedDict()
        for item in items:
            ids = item.get("haystack_session_ids", [])
            dates = item.get("haystack_dates", [])
            turns_list = item.get("haystack_sessions", [])
            if not (len(ids) == len(dates) == len(turns_list)):
                raise ValueError(
                    "LongMemEval record has mismatched haystack_session_ids, "
                    "haystack_dates, and haystack_sessions lengths"
                )
            for sid, date, turns in zip(ids, dates, turns_list):
                if sid in sessions:
                    continue
                sessions[str(sid)] = {
                    "session_id": str(sid),
                    "date": str(date),
                    "turns": [dict(t) for t in turns],
                }
        return sessions

    @staticmethod
    def _sort_sessions(unique_sessions: Mapping[str, Dict]) -> List[Tuple[str, Dict]]:
        return sorted(
            unique_sessions.items(),
            key=lambda item: (parse_longmemeval_date(item[1].get("date", "")), item[0]),
        )

    def _summary_for_session(self, sid: str, session: Mapping) -> str:
        if self.summarizer is None:
            return SessionSummarizer._format_as_dialogue(
                list(session.get("turns", [])),
                str(session.get("date", "")),
            )
        return self.summarizer.summarize_session(
            turns=list(session.get("turns", [])),
            session_date=str(session.get("date", "")),
            session_id=sid,
        )

    def _metadata_for_sid(self, sid: str) -> Dict:
        session = self.unique_sessions[sid]
        return {
            "date_time": session["date"],
            "session_id": sid,
            "session_key": self.summary_key_for_sid(sid),
            "session_node_id": self.session_node_id_for_sid(sid),
            "summary_index": self.sid_to_idx[sid] + 1,
            "num_turns": len(session.get("turns", [])),
        }

    def _qa_pair(self, item: Mapping) -> Dict:
        q_type = str(item.get("question_type", "unknown"))
        question_date = item.get("question_date")
        question = str(item.get("question", ""))
        if question_date:
            formatted_question = f"Date of user query: {question_date}\nUser: {question}"
        else:
            formatted_question = f"User: {question}"
        return {
            "question_id": item.get("question_id"),
            "question": formatted_question,
            "raw_question": question,
            "answer": item.get("answer"),
            "category": CATEGORY_MAP.get(q_type, 1),
            "original_question_type": q_type,
            "question_date": question_date,
            "haystack_session_ids": list(item.get("haystack_session_ids", [])),
            "answer_session_ids": list(item.get("answer_session_ids", [])),
        }
