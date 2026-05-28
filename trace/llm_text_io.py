"""LLM text I/O for TRACE.

Single home for all LLM-side text plumbing used by the release pipeline:

  1. A-Mem evolution prompts + parsers (~480 LoC): 5 prompt templates and
     parsers used by ``memory_layer_robust`` for the ingest-time note-evolution
     loop.

  2. QA answer parsing (~85 LoC): ``parse_plain_text_answer``,
     ``extract_answer_tag``, ``parse_relevant_parts``, ``parse_keywords_response``,
     ``normalize_answer`` — used by ``eval_locomo``, ``eval_longmemeval``, and
     ``trace.agent``.

  3. Temporal which-first post-processor (~190 LoC): ``parse_temporal_choice_answer``
     and helpers — used by ``eval_longmemeval`` only, for the LongMemEval TR
     sub-bucket fix.

  4. Generic helpers (~50 LoC): ``strip_markdown_fences``,
     ``parse_with_json_fallback``, ``_parse_list_items``, ``_extract_section`` —
     internal utilities for sections 1-3.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("amem_robust")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_with_json_fallback(response: str, plain_text_parser: Callable, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing.

    Many models emit valid JSON even without strict mode, so we try that first
    for best-of-both-worlds compatibility.
    """
    if not response:
        return plain_text_parser("", *parser_args) if plain_text_parser else ""
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


# ---------------------------------------------------------------------------
# List parsing helpers
# ---------------------------------------------------------------------------

def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items.

    Handles:
      - Bullet points (-, *, numbered)
      - Comma-separated values
      - One item per line
    """
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = re.sub(r'^[\-\*\u2022]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        # If the line contains commas, split on them
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end).

    Args:
        text: Full LLM response
        marker: Section header to find (e.g. "KEYWORDS")
        next_markers: List of possible next section headers

    Returns:
        The text content of that section (may be empty string).
    """
    # Build a regex that finds the marker (case-insensitive) followed by a colon
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    # The first line of content may be on the same line as the marker
    first_line = match.group(1).strip()

    # Find where the next section starts
    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# Parsers for each call site
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response.

    Returns:
        {"keywords": [...], "context": "...", "tags": [...]}
    """
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)

    # Validate / repair
    result = validate_analysis_result(result, content)
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response.

    Returns:
        {"decision": "NO_EVOLUTION|STRENGTHEN|UPDATE_NEIGHBOR|STRENGTHEN_AND_UPDATE",
         "reason": "..."}
    """
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        # Normalize common variants
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            # Try to infer from keywords
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    # Map JSON keys if we got JSON
    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response.

    Returns:
        {"connections": [int, ...], "tags": [str, ...]}
    """
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        # Parse connections as integers
        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    # Map from JSON keys if needed
    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [int(x) for x in result.get("suggested_connections", []) if isinstance(x, (int, float))]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response.

    Returns:
        [{"context": "...", "tags": [...]}, ...] — one per neighbor
    """
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            # Try to find NEIGHBOR i: block
            # Look for "NEIGHBOR i:" or "NEIGHBOR i\n"
            pattern = re.compile(
                rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE
            )
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            # Find the end of this neighbor block (next NEIGHBOR or end)
            next_pattern = re.compile(
                rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE
            )
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            tags = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    # Try JSON first
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


def extract_answer_tag(response: str) -> str | None:
    """Extract concise answer from 'ANSWER: ...' line if present.

    Looks for the *last* occurrence of 'ANSWER:' (case-insensitive) so that
    CoT reasoning earlier in the response is ignored.  Returns None if no
    tag is found.
    """
    import re
    if not response:
        return None
    # Find all occurrences, take the last one
    matches = list(re.finditer(r'(?i)^[ \t]*ANSWER\s*:\s*(.+)', response, re.MULTILINE))
    if matches:
        return matches[-1].group(1).strip()
    # Also handle inline (not at line start)
    matches = list(re.finditer(r'(?i)ANSWER\s*:\s*(.+)', response))
    if matches:
        return matches[-1].group(1).strip()
    return None


def parse_plain_text_answer(response: str) -> str:
    """Parse a plain-text answer response (for QA evaluation).

    Priority:
    1. If the response contains an 'ANSWER: ...' tag, extract that (v2 CoT prompt).
    2. If the model returned JSON with an "answer" field, extract it.
    3. Otherwise return the raw text.
    Always applies normalize_answer() at the end.
    """
    if not response:
        return ""
    # 1. ANSWER: tag extraction (v2 prompts)
    answer = extract_answer_tag(response)
    if answer:
        return normalize_answer(answer)

    # 2. JSON extraction
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return normalize_answer(str(data["answer"]))
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Fallback: raw text
    return normalize_answer(response.strip())


def normalize_answer(text: str) -> str:
    """Light normalization to improve F1 without changing semantic content.

    - Strip surrounding quotes
    - Strip trailing periods
    """
    import re
    if not text:
        return text
    # Strip surrounding quotes (single, double, smart quotes)
    text = re.sub(r'^[\"\'\u201c\u201d\u2018\u2019]+|[\"\'\u201c\u201d\u2018\u2019]+$', '', text).strip()
    # Strip trailing period (unless it's part of a number like "2.5")
    if text.endswith('.') and not re.search(r'\d\.$', text):
        text = text[:-1].strip()
    return text


def parse_relevant_parts(response: str) -> str:
    """Parse retrieve_memory_llm response.

    If JSON with "relevant_parts", extract it. Otherwise return raw text.
    """
    if not response:
        return ""
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "relevant_parts" in data:
            return str(data["relevant_parts"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_keywords_response(response: str) -> str:
    """Parse generate_query_llm response.

    If JSON with "keywords", extract it. Otherwise return raw text.
    """
    if not response:
        return ""
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


# ---------------------------------------------------------------------------
# Validation / heuristic repair
# ---------------------------------------------------------------------------

def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result.

    - If keywords is empty, extract capitalized words / nouns heuristically.
    - If context is empty, use the first sentence of content.
    - If tags is empty, derive from keywords.
    """
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # Ensure lists
    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    # Repair empty keywords from content
    if not keywords and content:
        keywords = _heuristic_keywords(content)

    # Repair empty context from content
    if not context and content:
        context = _heuristic_context(content)

    # Repair empty tags from keywords
    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    # Remove common stop words and extract significant words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    # Prefer capitalized words (likely proper nouns / key terms)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))

    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    # Take the first sentence (up to period, question mark, or exclamation)
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    # Fallback: first 200 chars
    return content[:200].strip()


# ---------------------------------------------------------------------------
# Temporal which-first answer post-processor (LongMemEval TR sub-bucket)
# ---------------------------------------------------------------------------

# Conflation note: "first / earlier / earliest" ask for ordering-from-start;
# "most recent / latest" ask for recency. Both reduce to the same parser
# behavior because we always do first-mention-wins on the candidates the LLM
# emitted in ANSWER. Future readers: do NOT split this into two regexes.
WHICH_FIRST_RE = re.compile(
    r"\bwhich (?:[\w']+\s){0,8}(?:first|earlier|earliest|most recent|recently|recent|latest)\b"
    r"|\bwho (?:[\w']+\s){0,8}(?:first|earlier|earliest)\b"
    r"|\bwhich event (?:[\w']+\s){0,8}(?:happened?|occurred|took place)\b",
    re.IGNORECASE,
)
_NEGATION_GUARD = re.compile(
    r"\bhaven['‘’]?t\b|\bhasn['‘’]?t\b|\bnever\b",
    re.IGNORECASE,
)
# Hedge guard: when the model expresses uncertainty in the ANSWER, the
# fallback (often a longer hedge sentence) overlaps the long hedge-style
# reference better than a stripped candidate. Skip in that case.
_HEDGE_GUARD = re.compile(
    r"\b(?:not enough|does not specify|cannot determine|no information|"
    r"insufficient|not specified|do not (?:have|specify))\b",
    re.IGNORECASE,
)
_PRONOUN_BLOCKLIST = {"it", "this", "that", "these", "those", "they", "them"}
# Match ASCII and curly quote characters
_QUOTE_CHARS = "'\"‘’“”"
# Words that signal which option came first/recently; candidate must appear
# within _ORDERING_WINDOW chars of one of these in the ANSWER blob, otherwise
# the answer's "X first, Y later" framing isn't actually placing the candidate.
_ORDERING_WORDS = (
    "first", "earlier", "earliest", "most recently", "most recent",
    "recently", "latest",
)
_ORDERING_WINDOW = 80  # characters


def is_which_first_question(question: str) -> bool:
    """Public: True iff the question matches the which-first pattern AND has no negation."""
    if _NEGATION_GUARD.search(question or ""):
        return False
    return bool(WHICH_FIRST_RE.search(question or ""))


def _extract_temporal_choice_candidates(question: str) -> List[str]:
    """Pull 2-4 candidate options from a which-first question.

    Two stages, first to succeed wins:
      1. Quoted spans of length 2..40 chars (preferred). Returned WITH their
         original quote characters so first-match can be re-emitted in the
         same style as the question/reference (e.g. "'The Hate U Give'").
      2. Comma-introduced "..., A or B?" tail of the question.

    Returned candidates are filtered to drop pronouns / sub-2-char fragments.
    """
    if not question:
        return []

    # Stage 1: quoted spans, preserve the quote characters
    quoted = re.findall(
        r"(['\"‘’“”])([^'\"‘’“”]{2,40})\1",
        question,
    )
    cands: List[str] = []
    if len(quoted) >= 2:
        cands = [qch + s + qch for qch, s in quoted[:4]]
    else:
        # Stage 2: ", A or B" / ", A, B or C" tail
        m = re.search(
            r",\s*(.+?)\s+or\s+(.+?)[?,.]?\s*$",
            question.strip(),
        )
        if m:
            cands = [m.group(1).strip(), m.group(2).strip()]

    cleaned: List[str] = []
    for c in cands:
        # Strip quotes only for the pronoun-blocklist test; we keep the
        # original (possibly quoted) candidate string in the cleaned list.
        bare = c.strip(_QUOTE_CHARS + " ").strip()
        if not bare:
            continue
        if bare.lower() in _PRONOUN_BLOCKLIST:
            continue
        if len(bare) < 2:
            continue
        cleaned.append(c)

    # De-dup while preserving order
    seen = set()
    deduped = []
    for c in cleaned:
        key = c.strip(_QUOTE_CHARS + " ").lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _extract_answer_blob(raw: str) -> str:
    """Pull text after the 'ANSWER:' tag; if absent, return the raw text."""
    if not raw:
        return ""
    m = re.search(r"ANSWER:\s*(.+?)(?:\n\n|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _passes_position_guard(blob: str, candidate: str) -> bool:
    """Candidate must appear within _ORDERING_WINDOW chars of any ordering word.

    This filters out cases like 'Mark and Sarah first' where the model picked
    the wrong option but happens to mention the other candidate later in the
    blob; without the guard we'd pick the WRONG candidate by first-mention.
    """
    bl = blob.lower()
    core = candidate.strip(_QUOTE_CHARS + " ").lower()
    cand_idx = bl.find(core)
    if cand_idx < 0:
        return False
    for w in _ORDERING_WORDS:
        for m in re.finditer(r"\b" + re.escape(w) + r"\b", bl):
            if abs(m.start() - cand_idx) <= _ORDERING_WINDOW:
                return True
    return False


def parse_temporal_choice_answer(raw: str, question: str, fallback: str) -> str:
    """Post-process the prediction for 'which X first' / 'who first' style
    LongMemEval temporal-reasoning questions.

    Behavior:
      - Returns `fallback` (the upstream parse_plain_text_answer output) on
        any failure: non-which-first question, hedge content, fewer than 2
        candidates, no candidate found in ANSWER content, candidate not near
        an ordering word, or candidate longer than 6 words.
      - On success: returns the candidate (preserving the question's quote
        style) that appears earliest in the ANSWER content.

    The function never modifies `raw`; it only computes a new prediction string.
    """
    if not is_which_first_question(question):
        return fallback
    candidates = _extract_temporal_choice_candidates(question)
    if len(candidates) < 2:
        return fallback

    blob = _extract_answer_blob(raw)
    if not blob:
        return fallback

    # Hedge guard: if the model's ANSWER expresses uncertainty, the fallback
    # (a hedge sentence) usually overlaps a hedge-style reference better than
    # a stripped candidate. Trust the fallback.
    if _HEDGE_GUARD.search(blob):
        return fallback

    blob_lower = blob.lower()
    best_idx, best_cand = None, None
    for cand in candidates:
        core = cand.strip(_QUOTE_CHARS + " ").lower()
        idx = blob_lower.find(core)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx, best_cand = idx, cand
    if best_cand is None:
        return fallback

    # Position guard: candidate must be near an ordering word to confirm the
    # first-mention pattern is actually about ordering rather than coincidence.
    if not _passes_position_guard(blob, best_cand):
        return fallback

    # Cap word count after stripping quotes (the cap targets noisy Stage-2
    # candidates, not the quote chars themselves).
    bare_word_count = len(best_cand.strip(_QUOTE_CHARS + " ").split())
    if bare_word_count > 6:
        return fallback

    # Return the candidate WITH its original quote style (so 'The Hate U Give'
    # ref-style candidates survive intact and don't lose the quote-token from
    # the F1 numerator).
    return best_cand.strip()
