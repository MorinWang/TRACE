"""TRACE memory agent + answer prompts.

Provides ``TRACEAgent`` — the A-Mem-backed conversational memory wrapper that
``ingest_locomo.py`` uses to build per-sample caches and that ``eval_locomo``
extends as ``TRACEGraphAgent`` — plus the three category-aware answer-prompt
templates and ``compute_prompt_hash`` for run-level prompt versioning.

Library only. Not invoked directly. Consumers:
  - ``ingest_locomo.py``  (cache build)
  - ``eval_locomo.py``    (graph-augmented QA, subclasses ``TRACEAgent``)
"""

import os
import sys

# Ensure top-level modules resolve when this file is imported from inside the
# trace/ package. Both consumers also extend sys.path, but this makes the
# import self-contained.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from trace.llm_text_io import parse_relevant_parts, parse_keywords_response
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("trace_eval")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT_CATEGORY_5 = (
    "Based on the context: {context}, answer the following question. {question}\n\n"
    "Select the correct answer: {option_a} or {option_b}  Short answer:"
)

ANSWER_PROMPT_CATEGORY_2 = (
    "Based on the context: {context}, answer the following question. "
    "Use DATE of CONVERSATION to answer with an approximate date.\n"
    "Please generate the shortest possible answer, using words from the conversation "
    "where possible, and avoid using any subjects.\n\n"
    "Question: {question} Short answer:"
)

ANSWER_PROMPT_DEFAULT = (
    "Based on the context: {context}, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {question} Short answer:"
)


def compute_prompt_hash() -> str:
    """SHA256 of all answer + memory-evolution prompt templates, for run versioning."""
    from trace.llm_text_io import (
        ANALYZE_CONTENT_PROMPT,
        EVOLUTION_DECISION_PROMPT,
        STRENGTHEN_DETAILS_PROMPT,
        UPDATE_NEIGHBORS_PROMPT,
        FOCUSED_KEYWORDS_PROMPT,
    )
    all_prompts = "\n---\n".join([
        ANALYZE_CONTENT_PROMPT,
        EVOLUTION_DECISION_PROMPT,
        STRENGTHEN_DETAILS_PROMPT,
        UPDATE_NEIGHBORS_PROMPT,
        FOCUSED_KEYWORDS_PROMPT,
        ANSWER_PROMPT_CATEGORY_5,
        ANSWER_PROMPT_CATEGORY_2,
        ANSWER_PROMPT_DEFAULT,
    ])
    return hashlib.sha256(all_prompts.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TRACEAgent:
    """A-Mem-backed memory agent. Subclassed by ``TRACEGraphAgent`` in eval_locomo."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 api_key=None, api_base=None,
                 skip_evolution=False,
                 embedding_model='all-MiniLM-L6-v2'):
        self.memory_system = RobustAgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base,
            skip_evolution=skip_evolution,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        prompt = (
            "Given the following conversation memories and a question, select the most "
            "relevant parts of the conversation that would help answer the question. "
            "Include the date/time if available.\n\n"
            f"Conversation memories:\n{memories_text}\n\n"
            f"Question: {query}\n\n"
            "Return only the relevant parts of the conversation that would help answer "
            "this specific question.\nIf no parts are relevant, return the input unchanged."
        )
        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        prompt = (
            "Given the following question, generate several keywords separated by commas.\n\n"
            f"Question: {question}\n\nKeywords:"
        )
        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str,
                        context_override: Optional[str] = None) -> tuple:
        if context_override is not None:
            context = context_override
            raw_context = "[FULL_CONTEXT_OVERRIDE]"
        else:
            keywords = self.generate_query_llm(question)
            raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
            context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            order_bit = int(hashlib.sha256(question.encode("utf-8")).hexdigest(), 16) % 2
            if order_bit == 0:
                option_a = "Not mentioned in the conversation"
                option_b = answer
            else:
                option_a = answer
                option_b = "Not mentioned in the conversation"
            user_prompt = ANSWER_PROMPT_CATEGORY_5.format(
                context=context, question=question,
                option_a=option_a, option_b=option_b,
            )
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = ANSWER_PROMPT_CATEGORY_2.format(context=context, question=question)
            temperature = 0.7
        else:
            user_prompt = ANSWER_PROMPT_DEFAULT.format(context=context, question=question)
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""
        return response, user_prompt, raw_context
