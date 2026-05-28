"""Type-specific LLM judge prompts for LongMemEval."""

PROMPT_VERSION = "longmemeval_judge_v1"

SYSTEM_MESSAGE = "You are an expert grader for long-term conversational memory benchmarks."

BASE_PROMPT = """
Your task is to judge whether a generated answer is correct for a LongMemEval question.

Question type: {question_type}
Question:
{question}

Gold answer:
{gold_answer}

Generated answer:
{generated_answer}

General grading rules:
- Return CORRECT if the generated answer contains the same essential information as the gold answer.
- Be generous about wording, formatting, and extra explanation.
- Return WRONG if the generated answer omits the key fact, gives a conflicting fact, or only says it cannot determine the answer.

{type_rules}

Return only a JSON object with this format:
{{"label": "CORRECT" or "WRONG"}}
"""

TYPE_RULES = {
    "temporal-reasoning": (
        "Temporal reasoning rule: accept equivalent dates or time periods. "
        "For durations counted in days, weeks, or months, tolerate an off-by-one counting difference "
        "when the intended time span is clearly the same."
    ),
    "knowledge-update": (
        "Knowledge update rule: the answer is correct if it contains the latest/current answer. "
        "Do not penalize extra old information unless it contradicts which answer is current."
    ),
    "single-session-preference": (
        "Preference rule: grade semantically. The generated answer does not need to match the exact "
        "words of the gold answer if it captures the user's stated preference."
    ),
    "single-session-user": (
        "Single-session rule: the generated answer should identify the same user-side fact as the gold answer."
    ),
    "single-session-assistant": (
        "Single-session rule: the generated answer should identify the same assistant-side response or fact as the gold answer."
    ),
    "multi-session": (
        "Multi-session rule: accept concise answers that correctly combine the relevant facts across sessions."
    ),
}

DEFAULT_RULES = "Use the general grading rules."


def format_longmemeval_judge_prompt(
    question: str,
    gold_answer: str,
    generated_answer: str,
    question_type: str,
) -> str:
    return BASE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
        question_type=question_type,
        type_rules=TYPE_RULES.get(question_type, DEFAULT_RULES),
    )
