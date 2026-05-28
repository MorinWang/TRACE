"""Layer 3: Enhanced QA prompts with CoT reasoning.

Adds TRACE-specific update-aware reasoning instructions to the conversational
QA prompt family.
"""

PROMPT_VERSION = "qa_prompt_v2.4"

# Shared suffix: forces concise final answer after CoT reasoning
_ANSWER_SUFFIX = """
After reasoning, write your final answer on a new line starting with "ANSWER: ".
The ANSWER must be a short factual phrase — no full sentences, no quotes, no filler words.
Bad: "I've known these friends for 4 years." Good: 4 years
Bad: "Camping next month (June 2023)." Good: June 2023"""

# A4: Relaxed suffix for open-domain questions that need brief explanations
_ANSWER_SUFFIX_OD = """
After reasoning, write your final answer on a new line starting with "ANSWER: ".
The ANSWER should be a brief response (1-2 short sentences). Include key details but stay concise.
Do not use quotes or filler words."""

QA_PROMPT_DEFAULT = """\
You are answering a question about a conversation using retrieved memory entries.

Reason through these steps:
1. Identify which memories are directly relevant to the question
2. If entries are marked [OUTDATED], prefer the newer information that replaced them
3. Connect information across multiple memories — if the question requires combining facts from different entries, explain the connection
4. For questions asking "how many" or "what are all", enumerate ALL instances found across all entries
5. When the answer involves a date, always use absolute dates (e.g. "May 2023") rather than relative ones ("yesterday", "last month")

Memories:
{context}

Question: {question}

Thought: Let me identify relevant entries and connect the information.
""" + _ANSWER_SUFFIX

QA_PROMPT_TEMPORAL = """\
You are answering a temporal question about a conversation using retrieved memory entries.

Reason through these steps:
1. Look for dates and timestamps in the memories (check "talk start time" fields)
2. Convert ALL relative references ("last week", "yesterday", "next month", "last year") to absolute dates using the "talk start time" of the memory entry where the phrase appears. For example, if a memory from "25 May 2023" says "yesterday", compute: 25 May 2023 minus 1 day = 24 May 2023.
3. If causal evidence shows temporal ordering, use it to narrow down the time
4. If an event is marked [OUTDATED], the update happened after the original event's date
5. Answer with the most specific absolute date or time period you can determine

Memories:
{context}

Question: {question}

Thought: Let me trace the timeline, converting every relative reference to an absolute date.

IMPORTANT: Your ANSWER must be an absolute date or time period (e.g. "7 May 2023", "June 2023", "2022", "Since 2016"). NEVER use relative words like "yesterday", "last year", "next month", "last week" in your ANSWER.
""" + _ANSWER_SUFFIX

# A4: OD prompt — same reasoning steps as DEFAULT but with relaxed answer suffix
QA_PROMPT_OD = """\
You are answering a question about a conversation using retrieved memory entries.

Reason through these steps:
1. Identify which memories are directly relevant to the question
2. If entries are marked [OUTDATED], prefer the newer information that replaced them
3. Connect information across multiple memories — if the question requires combining facts from different entries, explain the connection
4. For questions asking "how many" or "what are all", enumerate ALL instances found across all entries
5. When the answer involves a date, always use absolute dates (e.g. "May 2023") rather than relative ones ("yesterday", "last month")

Memories:
{context}

Question: {question}

Thought: Let me identify relevant entries and connect the information.
""" + _ANSWER_SUFFIX_OD

QA_PROMPT_PREFERENCE = """\
You are profiling a user's preferences from retrieved memory entries to help answer a recommendation, suggestion, or advice question.

The user is asking for a recommendation. Do NOT answer with concrete items (specific restaurants, products, books, websites). Instead, profile what kind of response THIS user would prefer based on their habits, interests, brand affinities, prior choices, and constraints visible in the memories.

Reasoning steps:
1. Find memories where the user expresses a preference, habit, brand affinity, hobby, recent purchase, or prior choice relevant to the question topic.
2. Identify what the user would value (specific brands, styles, contexts, prior successes, established tools/equipment).
3. Identify what the user would NOT prefer (incompatible options, things they explicitly disliked, or generic alternatives that ignore their established context).
4. If direct preference signals are sparse, infer plausible preferences from any topical signals in the memories (mentioned tools, brands, hobbies, prior actions). It is better to give a focused inferred preference than to say "no information found".

Memories:
{context}

Question: {question}

Thought: Let me identify the user's relevant habits and prior preferences.

After reasoning, write your final answer on a new line starting with "ANSWER: " using EXACTLY this two-sentence format:
ANSWER: The user would prefer <one short clause about what they would prefer, max 25 words>. They might not prefer <one short clause about what they would NOT prefer, max 25 words>.

If you genuinely cannot infer a "not prefer" clause from any signal, default the second clause to: "generic recommendations that ignore their stated interests and context".

Bad ANSWER: Adobe Premiere Pro tutorials, YouTube channels, Udemy courses
Good ANSWER: The user would prefer resources specifically tailored to Adobe Premiere Pro and its advanced settings. They might not prefer generic video editing tutorials or resources for other software.
"""
