"""TRACE retrieval pipeline - tokenopt variant.

Copy of trace_pipeline.py with hybrid context budget control layered on top of
_format_hybrid_context. Preserves the exact same retrieval / scoring / Phase 2
logic as the main pipeline; only the final context string is token-capped.

Two optimizations applied here (config-gated via TokenOptConfig):
  - #A Allocation-style truncation: notes get notes_budget tokens first;
    unused budget spills over to causal paths (up to path_budget); final
    string must fit under total_cap.
  - #C Event-ID deduplication: within the causal evidence section, if the
    same event_id appears across multiple paths, its state_change is only
    fully rendered once (later occurrences show a reference stub).

Token counting uses tiktoken's o200k_base (gpt-4o-mini family) by default.
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from trace.causal_graph import CausalGraph
from trace.path_reasoner import PathReasoner, ScoredPath

logger = logging.getLogger(__name__)


# v4: warn-once tracker for unknown dedup_section1 mode (typo guard)
_WARNED_UNKNOWN_DEDUP_MODE: set = set()

# v6: warn-once tracker for unknown note_format_mode (typo guard)
_WARNED_UNKNOWN_FORMAT_MODE: set = set()


# ---------------------------------------------------------------------------
# TokenOpt configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenOptConfig:
    """Per-run token-optimization knobs.

    Disabled by default (enabled=False) makes the pipeline byte-identical to
    trace_pipeline.py's behavior.
    """
    enabled: bool = False
    notes_budget: int = 3000           # target budget for Section 1 (A-Mem notes + neighborhood)
    path_budget: int = 1500            # target budget for Section 2 (causal paths + expanded notes)
    total_cap: int = 5000              # hard upper limit on the joined context
    dedup_by_event_id: bool = True     # #C: deduplicate events across paths in Section 2
    dedup_section1: str = "note_id"    # NEW v4 (A1): "off" | "note_id" | "content_hash"
    min_path_score: float = 0.30       # NEW v4 (B2): was hard-coded 0.15 inline
    source_note_query_filter: float = 0.40  # NEW v4 (B3): cosine threshold; 0.0 = off
    note_format_mode: str = "minimal"  # NEW v6: "full" (v4) | "no_context" (v5) | "minimal" (v6 default, ~-68% per memory)
    use_llmlingua: bool = False        # reserved for future compression pass (not implemented in round 1)
    tiktoken_encoding: str = "o200k_base"  # gpt-4o-mini / gpt-4o family
    # Per-category budget overrides (L3-aligned). Keys are LoCoMo categories 1-5 as strings.
    # Each override may set any subset of notes_budget / path_budget / total_cap.
    # Unset keys fall back to the root values above.
    # Example:
    #   "category_overrides": {
    #       "3": {"notes_budget": 10000, "path_budget": 5000, "total_cap": 15000},
    #       "4": {"notes_budget": 12000, "path_budget": 6000, "total_cap": 18000}
    #   }
    category_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def resolved_budgets(self, category: Optional[int]) -> Tuple[int, int, int]:
        """Return (notes_budget, path_budget, total_cap) for this category."""
        override = self.category_overrides.get(str(category), {}) if category is not None else {}
        return (
            override.get("notes_budget", self.notes_budget),
            override.get("path_budget", self.path_budget),
            override.get("total_cap", self.total_cap),
        )


_ENCODER_CACHE: Dict[str, object] = {}


def _get_encoder(encoding_name: str):
    """Return a cached tiktoken encoder; lazily import tiktoken to avoid hard dep when tokenopt disabled."""
    if encoding_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoding_name]
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    _ENCODER_CACHE[encoding_name] = enc
    return enc


def _count_tokens(text: str, encoding_name: str) -> int:
    if not text:
        return 0
    return len(_get_encoder(encoding_name).encode(text))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for TRACE pipeline v2 layers."""
    layer1_hybrid_context: bool = True
    layer1_neighborhood_expansion: bool = True
    layer3_enhanced_qa_prompt: bool = False
    layer4_graph_expansion: bool = False
    layer5_query_entity_extraction: bool = False
    # Method A: Session hyperedge augmentation
    session_augmentation: bool = False
    max_session_fanout: int = 3
    max_session_hops: int = 1
    # Method D: Answer-aware path search
    answer_aware_paths: bool = False
    answer_aware_top_k: int = 5
    # Method C: Nested hypergraph (topic clustering)
    topic_augmentation: bool = False
    max_topic_fanout: int = 2
    max_topic_hops: int = 1
    hierarchical_seed_selection: bool = False
    # C3 mechanism: update-aware seed augmentation (sequence seeds along
    # updates/contradicts edges so BFS can reach update paths). Default ON
    # preserves prior runtime behavior. Set False to ablate C3 in seeding layer.
    update_aware_seed_augmentation: bool = True


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Output of the TRACE retrieval pipeline."""

    context: str                              # formatted context for LLM answer generation
    top_paths: List[ScoredPath] = field(default_factory=list)
    explanation: str = ""                     # human-readable path explanation
    seed_note_ids: List[str] = field(default_factory=list)
    seed_event_ids: List[str] = field(default_factory=list)
    num_paths_found: int = 0
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# TRACEPipeline
# ---------------------------------------------------------------------------

class TRACEPipeline:
    """End-to-end TRACE retrieval pipeline with layered optimizations."""

    @staticmethod
    def _should_skip_note(note, seen: set, mode: str) -> bool:
        """A1 dedup helper. mode: 'off' | 'note_id' | 'content_hash'.

        Returns True (skip) if already seen. Side effect: adds key to seen on miss.
        With mode='off' returns False unconditionally (byte-equivalent legacy).
        Unknown mode warns once then behaves as 'off' (typo guard).
        """
        if mode == "off":
            return False
        if mode == "note_id":
            key = note.id
        elif mode == "content_hash":
            import hashlib
            key = hashlib.md5((note.content or "").encode("utf-8")).hexdigest()
        else:
            if mode not in _WARNED_UNKNOWN_DEDUP_MODE:
                _WARNED_UNKNOWN_DEDUP_MODE.add(mode)
                logger.warning(
                    f"Unknown dedup_section1 mode {mode!r}; expected "
                    f"'off'|'note_id'|'content_hash'. Treating as 'off'."
                )
            return False
        if key in seen:
            return True
        seen.add(key)
        return False

    @staticmethod
    def _cosine_sim(a, b) -> float:
        """B3 helper. Cosine similarity; returns 0.0 if either norm is zero."""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def __init__(
        self,
        graph: CausalGraph,
        reasoner: PathReasoner,
        config: Optional[PipelineConfig] = None,
        context_filter=None,
        entity_extractor=None,
        token_opt: Optional[TokenOptConfig] = None,
    ):
        self.graph = graph
        self.reasoner = reasoner
        self.config = config or PipelineConfig()
        self.context_filter = context_filter
        self.entity_extractor = entity_extractor
        self.token_opt = token_opt or TokenOptConfig()  # disabled by default
        self._note_id_to_idx: Optional[Dict[str, int]] = None

    def retrieve(
        self,
        query: str,
        retrieval_indices: np.ndarray,
        memory_system,
        query_embedding: np.ndarray,
        core_k: int = 0,
        final_k: int = 10,
        question_category: Optional[int] = None,
    ) -> RetrievalResult:
        """Full TRACE retrieval pipeline.

        Flow:
        1. [L5] Query entity extraction -> expand seeds
        2. [L4] Graph-augmented retrieval expansion
        3. Map neural retrieval -> seed events
        4. BFS path search + fusion scoring
        5. [L1] Format hybrid context (A-Mem + path evidence)
        """
        # Build note_id_to_idx mapping (cached after first call)
        if self._note_id_to_idx is None:
            self._note_id_to_idx = self.build_note_id_to_idx(memory_system)

        all_embeddings = self._get_embeddings(memory_system)

        # Layer 5: Query entity extraction -> supplement seed events
        entity_event_ids = []
        if self.config.layer5_query_entity_extraction and self.entity_extractor:
            entity_event_ids = self._extract_entity_seeds(query, memory_system)

        # Map neural retrieval results to seed events
        seed_note_ids, seed_event_ids = self.map_notes_to_seed_events(
            retrieval_indices, memory_system
        )

        # Update-aware seed augmentation: ensure both old and new versions
        # of updated facts are in the seed set so BFS can find update paths.
        if self.config.update_aware_seed_augmentation and seed_event_ids:
            update_seeds = []
            seen_seeds = set(seed_event_ids)
            for eid in list(seed_event_ids):
                event = self.graph.get_event(eid)
                if event is None:
                    continue
                # If this event has been superseded, add the updater
                if event.valid_until is not None:
                    for pred in self.graph.graph.predecessors(eid):
                        edge_data = self.graph.graph.edges[pred, eid]
                        if edge_data.get("edge_type") == "updates" and pred not in seen_seeds:
                            update_seeds.append(pred)
                            seen_seeds.add(pred)
                # If this event updates others, add the outdated events
                for succ in self.graph.graph.successors(eid):
                    edge_data = self.graph.graph.edges[eid, succ]
                    if edge_data.get("edge_type") in ("updates", "contradicts") and succ not in seen_seeds:
                        update_seeds.append(succ)
                        seen_seeds.add(succ)
            if update_seeds:
                seed_event_ids.extend(update_seeds)
                logger.info(f"Update-aware seed augmentation: +{len(update_seeds)} events")

        # Merge entity-based seeds
        if entity_event_ids:
            seen = set(seed_event_ids)
            for eid in entity_event_ids:
                if eid not in seen:
                    seen.add(eid)
                    seed_event_ids.append(eid)

        # Method C: Hierarchical seed selection (topic -> session -> event)
        if (self.config.hierarchical_seed_selection
                and self.config.topic_augmentation
                ):
            hier_event_ids = self._hierarchical_seed_selection(
                query, memory_system, seed_event_ids
            )
            seen_hier = set(seed_event_ids)
            for eid in hier_event_ids:
                if eid not in seen_hier:
                    seen_hier.add(eid)
                    seed_event_ids.append(eid)

        # Layer 4: Graph-augmented retrieval expansion
        if self.config.layer4_graph_expansion and seed_event_ids:
            retrieval_indices = self.graph_expand_retrieval(
                seed_event_ids, memory_system, retrieval_indices
            )

        if not seed_event_ids:
            logger.info("No seed events found, falling back to A-Mem retrieval")
            fallback_context = self._format_amem_context(retrieval_indices, memory_system)
            return RetrievalResult(
                context=fallback_context,
                seed_note_ids=seed_note_ids,
                used_fallback=True,
            )

        # Run path reasoning
        # Attach event description embeddings to graph for direct evidence search
        if hasattr(self, '_event_desc_cache') and self._event_desc_cache is not None:
            self.graph._desc_embedding_cache = self._event_desc_cache
        scored_paths = self.reasoner.find_and_score_paths(
            seed_event_ids=seed_event_ids,
            graph=self.graph,
            query_embedding=query_embedding,
            note_id_to_idx=self._note_id_to_idx,
            all_embeddings=all_embeddings,
        )

        # Format context - use core indices for Section 1 (fair comparison with baseline)
        context_indices = retrieval_indices[:core_k] if core_k > 0 else retrieval_indices
        context = self.format_context(
            scored_paths, memory_system, context_indices,
            query=query,
            query_embedding=query_embedding,
            final_k=final_k,
            question_category=question_category,
        )

        # Build explanation string - BFS paths + direct evidence
        explanation_parts = []
        bfs_count = 0
        direct_count = 0
        for sp in scored_paths:
            if bfs_count + direct_count >= 3:
                break
            is_direct = "[Direct Evidence]" in sp.explanation
            if is_direct and direct_count < 2:
                direct_count += 1
                explanation_parts.append(f"--- Direct Evidence {direct_count} ---")
                explanation_parts.append(sp.explanation)
            elif not is_direct and bfs_count < 2:
                bfs_count += 1
                explanation_parts.append(f"--- Path {bfs_count} ---")
                explanation_parts.append(sp.explanation)
        # Ensure at least one direct evidence if available
        if direct_count == 0:
            for sp in scored_paths:
                if "[Direct Evidence]" in sp.explanation:
                    explanation_parts.append("--- Direct Evidence ---")
                    explanation_parts.append(sp.explanation)
                    break
        explanation = "\n".join(explanation_parts)

        return RetrievalResult(
            context=context,
            top_paths=scored_paths,
            explanation=explanation,
            seed_note_ids=seed_note_ids,
            seed_event_ids=seed_event_ids,
            num_paths_found=len(scored_paths),
            used_fallback=False,
        )

    def retrieve_with_support(
        self,
        query: str,
        answer: str,
        retrieval_indices: np.ndarray,
        memory_system,
        query_embedding: np.ndarray,
        answer_embedding: np.ndarray,
        skip_expansion: bool = False,
        final_k: int = 10,
        question_type: str = "default",
    ) -> RetrievalResult:
        """Phase 2: Find answer-aware support paths.

        Uses the generated answer to find paths that faithfully support it.
        Category-adaptive: passes question_type to the reasoner.
        """
        from trace.answer_aware_reasoner import AnswerAwarePathReasoner

        if self._note_id_to_idx is None:
            self._note_id_to_idx = self.build_note_id_to_idx(memory_system)

        # Ensure event description cache exists
        if not hasattr(self, '_event_desc_cache') or self._event_desc_cache is None:
            retriever = getattr(memory_system, 'retriever', None)
            model = getattr(retriever, 'model', None) if retriever else None
            if model is not None:
                all_events = self.graph.get_all_events()
                if all_events:
                    descriptions = [ev.state_change for ev in all_events]
                    event_id_list = [ev.event_id for ev in all_events]
                    desc_embs = model.encode(descriptions)
                    from numpy.linalg import norm
                    norms = norm(desc_embs, axis=1)
                    norms[norms == 0] = 1.0
                    self._event_desc_cache = {
                        'event_ids': event_id_list,
                        'embeddings': desc_embs,
                        'norms': norms,
                    }

        # Attach cache to graph for the reasoner
        if hasattr(self, '_event_desc_cache') and self._event_desc_cache is not None:
            self.graph._desc_embedding_cache = self._event_desc_cache

        aa_reasoner = AnswerAwarePathReasoner(
            max_depth=self.config.max_session_hops + 3,  # allow deeper for session hops
            top_k=self.config.answer_aware_top_k,
        )

        scored_paths = aa_reasoner.find_support_paths(
            query=query,
            answer=answer,
            graph=self.graph,
            query_embedding=query_embedding,
            answer_embedding=answer_embedding,
            note_id_to_idx=self._note_id_to_idx,
            all_embeddings=self._get_embeddings(memory_system),
            question_type=question_type,
        )

        # Build explanation from support paths
        explanation_parts = []
        for i, sp in enumerate(scored_paths[:2]):
            explanation_parts.append(f"--- Support Path {i+1} ---")
            explanation_parts.append(sp.explanation)
        explanation = "\n".join(explanation_parts)

        # Map seeds for metadata
        seed_note_ids, seed_event_ids = self.map_notes_to_seed_events(
            retrieval_indices, memory_system
        )

        return RetrievalResult(
            context="",  # Phase 2 doesn't produce context
            top_paths=scored_paths,
            explanation=explanation,
            seed_note_ids=seed_note_ids,
            seed_event_ids=seed_event_ids,
            num_paths_found=len(scored_paths),
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    # Note-to-event mapping
    # ------------------------------------------------------------------

    def map_notes_to_seed_events(
        self,
        indices: np.ndarray,
        memory_system,
    ) -> Tuple[List[str], List[str]]:
        """Map retriever indices -> note IDs -> event IDs."""
        all_memories = list(memory_system.memories.values())
        note_ids = []
        for idx in indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(all_memories):
                note_ids.append(all_memories[idx_int].id)

        event_ids = []
        seen = set()
        for nid in note_ids:
            for event in self.graph.get_events_by_note(nid):
                if event.event_id not in seen:
                    seen.add(event.event_id)
                    event_ids.append(event.event_id)

        logger.info(
            f"Mapped {len(indices)} retrieval indices -> "
            f"{len(note_ids)} notes -> {len(event_ids)} seed events"
        )
        return note_ids, event_ids

    def build_note_id_to_idx(self, memory_system) -> Dict[str, int]:
        """Build mapping from note ID -> index in memories list."""
        return {
            note.id: i
            for i, note in enumerate(memory_system.memories.values())
        }

    # ------------------------------------------------------------------
    # Context formatting (Layer 1)
    # ------------------------------------------------------------------

    def format_context(
        self,
        top_paths: List[ScoredPath],
        memory_system,
        fallback_indices: np.ndarray,
        query: str = "",
        query_embedding: Optional[np.ndarray] = None,
        final_k: int = 10,
        question_category: Optional[int] = None,
    ) -> str:
        """Format context for LLM answer generation.

        When layer1_hybrid_context=True: A-Mem full retrieval as primary
        context + causal path evidence as supplement.
        When False: legacy path-only context.
        """
        if self.config.layer1_hybrid_context:
            return self._format_hybrid_context(
                top_paths, memory_system, fallback_indices, query,
                query_embedding=query_embedding,
                final_k=final_k, question_category=question_category,
            )
        return self._format_context_legacy(top_paths, memory_system, fallback_indices)

    def _format_hybrid_context(
        self,
        top_paths: List[ScoredPath],
        memory_system,
        fallback_indices: np.ndarray,
        query: str = "",
        query_embedding: Optional[np.ndarray] = None,
        final_k: int = 10,
        question_category: Optional[int] = None,
    ) -> str:
        """Layer 1: Hybrid context = A-Mem retrieval + causal path evidence.

        Section 1 replicates A-Mem's find_related_memories_raw() exactly:
        - No deduplication (duplicates reinforce important information)
        - Neighborhood expansion follows note.links with cap at k
        Section 2 appends TRACE causal path evidence (deduped against Section 1).

        TokenOpt hook: when self.token_opt.enabled, the section-1 parts and
        section-2 parts are collected separately so they can be budget-trimmed
        and event-id-deduped before being joined. When disabled, the path is
        byte-identical to trace_pipeline.py.
        """
        # v6: bind note_format_mode (default "full" if token_opt disabled = legacy)
        _fmt_mode = (
            self.token_opt.note_format_mode if self.token_opt.enabled else "full"
        )
        fmt = partial(self._format_note_amem, format_mode=_fmt_mode)
        section1_parts: List[str] = []
        section2_parts: List[str] = []
        all_memories = list(memory_system.memories.values())
        k = len(fallback_indices)
        mem_counter = 1  # Running counter for Memory N labels

        active_indices = fallback_indices

        # Section 1: A-Mem style retrieval - exact replica of find_related_memories_raw
        # TokenOpt v4: A1 dedup driven by self.token_opt.dedup_section1
        # `seen_section1_note_ids` is ALWAYS tracked (preserves pre-v4 invariant
        # that Section 2 never re-emits notes already in Section 1)
        section1_dedup_mode = (
            self.token_opt.dedup_section1 if self.token_opt.enabled else "off"
        )
        seen_keys: set = set()                  # A1 mode-aware
        seen_section1_note_ids: set = set()     # always-on (Section 2 cross-ref)

        for idx in active_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(all_memories):
                note = all_memories[idx_int]
                if self._should_skip_note(note, seen_keys, section1_dedup_mode):
                    continue
                seen_section1_note_ids.add(note.id)
                section1_parts.append(fmt(note, mem_counter))
                mem_counter += 1

                # Neighborhood expansion
                if self.config.layer1_neighborhood_expansion:
                    j = 0
                    for neighbor_idx in getattr(note, 'links', []):
                        if 0 <= neighbor_idx < len(all_memories):
                            n = all_memories[neighbor_idx]
                            if self._should_skip_note(n, seen_keys, section1_dedup_mode):
                                continue
                            seen_section1_note_ids.add(n.id)
                            section1_parts.append(fmt(n, mem_counter))
                            mem_counter += 1
                            j += 1
                        if j >= k:
                            break

        # Section 2: Causal path evidence (TRACE's unique contribution)
        # B2: min_path_score config-driven; B3: source_note query filter; A1: shared seen_keys
        # `seen_section1_note_ids` already populated above; reuse it
        if top_paths:
            section2_parts.append("\n=== Causal Evidence ===")
            # B2: threshold from config (legacy=0.15)
            min_path_score = (
                self.token_opt.min_path_score if self.token_opt.enabled else 0.15
            )
            quality_paths = [sp for sp in top_paths if sp.total_score >= min_path_score]

            # B3: prepare query-similarity filter for source_notes
            # Use pipeline-cached lookups (process_query lazily builds these)
            sn_filter = (
                self.token_opt.source_note_query_filter
                if self.token_opt.enabled else 0.0
            )
            if self._note_id_to_idx is None:
                self._note_id_to_idx = self.build_note_id_to_idx(memory_system)
            note_id_to_idx = self._note_id_to_idx
            all_embs = self._get_embeddings(memory_system)
            sn_filter_active = (
                sn_filter > 0.0
                and query_embedding is not None
                and note_id_to_idx is not None
                and all_embs is not None
            )

            # #C: track event_ids already fully rendered when dedup_by_event_id is on.
            seen_event_ids: set = set()
            dedup_on = self.token_opt.enabled and self.token_opt.dedup_by_event_id

            for i, sp in enumerate(quality_paths[:3]):  # Cap at 3 paths (was 5)
                if dedup_on:
                    # Render this path but mark events already seen with a short reference stub.
                    section2_parts.append(
                        f"Path {i+1}: {self._format_path_summary_dedup(sp, seen_event_ids)}"
                    )
                    for event in sp.events:
                        seen_event_ids.add(event.event_id)
                else:
                    section2_parts.append(f"Path {i+1}: {self._format_path_summary(sp)}")

                for event in sp.events:
                    for nid in event.source_note_ids:
                        note = memory_system.memories.get(nid)
                        if note is None:
                            continue
                        # ALWAYS skip notes already emitted in Section 1 (preserves pre-v4 invariant)
                        if nid in seen_section1_note_ids:
                            continue
                        seen_section1_note_ids.add(nid)  # avoid intra-Section-2 re-emission too
                        # A1: mode-aware dedup
                        if self._should_skip_note(note, seen_keys, section1_dedup_mode):
                            continue
                        # B3: drop notes too far from query
                        if sn_filter_active:
                            ix = note_id_to_idx.get(nid)
                            if ix is None or ix >= len(all_embs):
                                continue
                            sim = self._cosine_sim(query_embedding, all_embs[ix])
                            if sim < sn_filter:
                                continue
                        section2_parts.append(fmt(note, mem_counter))
                        mem_counter += 1

        # TokenOpt: apply budget-based truncation across the two sections.
        if self.token_opt.enabled:
            section1_parts, section2_parts = self._apply_token_opt_truncation(
                section1_parts, section2_parts, category=question_category,
            )

        return "\n".join(section1_parts + section2_parts)

    # ------------------------------------------------------------------
    # TokenOpt helpers (tokenopt-only)
    # ------------------------------------------------------------------

    def _apply_token_opt_truncation(
        self,
        section1_parts: List[str],
        section2_parts: List[str],
        category: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Allocation-style budget trim (#A).

        Fill section1 up to notes_budget; unused budget spills into section2 (on top of
        path_budget). Final combined length must be <= total_cap. Parts are kept atomic
        (no mid-part slicing); drop from the tail if over budget.

        Budgets are resolved per-category (L3-aligned): when category_overrides is set
        in TokenOptConfig, categories with extra-wide budgets are honored here.
        """
        tok = self.token_opt
        enc = tok.tiktoken_encoding
        notes_budget, path_budget, total_cap = tok.resolved_budgets(category)

        # First pass: count per-part tokens
        s1_counts = [_count_tokens(p, enc) for p in section1_parts]
        s2_counts = [_count_tokens(p, enc) for p in section2_parts]

        # Fill section 1 up to notes_budget (atomic parts, no slicing)
        s1_keep = []
        s1_total = 0
        for p, c in zip(section1_parts, s1_counts):
            if s1_total + c <= notes_budget:
                s1_keep.append(p)
                s1_total += c
            else:
                break

        # Unused section-1 budget spills into section-2 allowance
        s2_allowance = path_budget + max(0, notes_budget - s1_total)

        # Always preserve the "=== Causal Evidence ===" header if present
        s2_keep = []
        s2_total = 0
        for p, c in zip(section2_parts, s2_counts):
            # Header line is cheap; keep it even if it would exceed by a few tokens
            if p.startswith("\n=== Causal Evidence ==="):
                s2_keep.append(p)
                s2_total += c
                continue
            if s2_total + c <= s2_allowance:
                s2_keep.append(p)
                s2_total += c
            else:
                break

        # Global cap enforcement - if somehow over total_cap, drop from section 2 tail
        combined_total = s1_total + s2_total
        while combined_total > total_cap and len(s2_keep) > 1:
            last = s2_keep.pop()
            combined_total -= _count_tokens(last, enc)

        logger.info(
            "tokenopt_log cat=%s s1_in=%d s1_kept=%d s1_in_tok=%d s1_kept_tok=%d "
            "s2_in=%d s2_kept=%d s2_in_tok=%d s2_kept_tok=%d total=%d cap=%d "
            "notes_budget=%d path_budget=%d",
            category,
            len(section1_parts), len(s1_keep), sum(s1_counts), s1_total,
            len(section2_parts), len(s2_keep), sum(s2_counts), s2_total,
            combined_total, total_cap,
            notes_budget, path_budget,
        )
        return s1_keep, s2_keep

    @staticmethod
    def _format_path_summary_dedup(sp: ScoredPath, seen_event_ids: set) -> str:
        """Like _format_path_summary but collapses already-seen events into [see above]."""
        chain = []
        for i, event in enumerate(sp.events):
            if event.event_id in seen_event_ids:
                # Compact reference - avoids re-rendering full state_change + time_anchor
                chain.append(f"[event {event.event_id}: see above]")
            else:
                status = ""
                if event.update_val == 0.0:
                    status = " [OUTDATED]"
                elif event.update_val == 0.5:
                    status = " [PARTIALLY UPDATED]"
                chain.append(f"{event.state_change}{status} ({event.time_anchor})")
            if i < len(sp.edge_types):
                chain.append(f" --[{sp.edge_types[i]}]--> ")
        return "".join(chain)

    def _format_context_legacy(
        self,
        top_paths: List[ScoredPath],
        memory_system,
        fallback_indices: np.ndarray,
    ) -> str:
        """Legacy v1 context format (path-only, for ablation baseline)."""
        if not top_paths:
            return self._format_amem_context(fallback_indices, memory_system)

        parts = []
        parts.append("=== Evidence from reasoning paths ===")
        seen_note_ids = set()

        for i, sp in enumerate(top_paths[:3]):
            for event in sp.events:
                for nid in event.source_note_ids:
                    if nid not in seen_note_ids:
                        seen_note_ids.add(nid)
                        note = memory_system.memories.get(nid)
                        if note:
                            ts = getattr(note, 'timestamp', '')
                            content = getattr(note, 'content', '')
                            context_desc = getattr(note, 'context', '')
                            parts.append(f"[{ts}] {content}")
                            if context_desc and context_desc != "General":
                                parts.append(f"  Context: {context_desc}")

        all_memories = list(memory_system.memories.values())
        supplement_count = 0
        for idx in fallback_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(all_memories):
                note = all_memories[idx_int]
                if note.id not in seen_note_ids:
                    seen_note_ids.add(note.id)
                    ts = getattr(note, 'timestamp', '')
                    content = getattr(note, 'content', '')
                    parts.append(f"[{ts}] {content}")
                    supplement_count += 1
                    if supplement_count >= 3:
                        break

        return "\n".join(parts)

    @staticmethod
    def _format_note_amem(note, note_idx: int = 0, *, format_mode: str = "full") -> str:
        """Format a note in A-Mem's original format (v2.2 proven path).

        format_mode controls per-memory verbosity:
          - "full"       : 5 fields (ts + content + context + keywords + tags) — v4 byte-equiv
          - "no_context" : 4 fields (ts + content + keywords + tags) — v5
          - "minimal"    : 2 fields (ts + content only) — v6 default (~-68% per memory)

        Default "full" preserves byte-equiv for legacy callers.
        Unknown mode warns once, treats as "full" (typo guard).
        """
        ts = getattr(note, 'timestamp', '')
        content = getattr(note, 'content', '')
        parts = ["talk start time:" + ts, "memory content: " + content]

        if format_mode == "full":
            parts.append("memory context: " + getattr(note, 'context', ''))
            parts.append("memory keywords: " + str(getattr(note, 'keywords', [])))
            parts.append("memory tags: " + str(getattr(note, 'tags', [])))
        elif format_mode == "no_context":
            parts.append("memory keywords: " + str(getattr(note, 'keywords', [])))
            parts.append("memory tags: " + str(getattr(note, 'tags', [])))
        elif format_mode == "minimal":
            pass  # only timestamp + content
        else:
            if format_mode not in _WARNED_UNKNOWN_FORMAT_MODE:
                _WARNED_UNKNOWN_FORMAT_MODE.add(format_mode)
                logger.warning(
                    f"Unknown note_format_mode {format_mode!r}; expected "
                    f"'full'|'no_context'|'minimal'. Treating as 'full'."
                )
            parts.append("memory context: " + getattr(note, 'context', ''))
            parts.append("memory keywords: " + str(getattr(note, 'keywords', [])))
            parts.append("memory tags: " + str(getattr(note, 'tags', [])))

        return "".join(parts)

    @staticmethod
    def _format_path_summary(sp: ScoredPath) -> str:
        """Format a scored path as a concise chain with update markers."""
        chain = []
        for i, event in enumerate(sp.events):
            status = ""
            if event.update_val == 0.0:
                status = " [OUTDATED]"
            elif event.update_val == 0.5:
                status = " [PARTIALLY UPDATED]"
            chain.append(f"{event.state_change}{status} ({event.time_anchor})")
            if i < len(sp.edge_types):
                chain.append(f" --[{sp.edge_types[i]}]--> ")
        return "".join(chain)

    def _format_amem_context(
        self,
        indices: np.ndarray,
        memory_system,
    ) -> str:
        """Build full A-Mem style context (fallback when no paths found).

        Replicates find_related_memories_raw() exactly: no dedup, with neighborhood.
        """
        # v6: same flag binding (legacy path, used when layer1_hybrid_context=False)
        _fmt_mode = (
            self.token_opt.note_format_mode if self.token_opt.enabled else "full"
        )
        fmt = partial(self._format_note_amem, format_mode=_fmt_mode)
        all_memories = list(memory_system.memories.values())
        k = len(indices)
        parts = []
        mem_counter = 1
        for idx in indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(all_memories):
                note = all_memories[idx_int]
                parts.append(fmt(note, mem_counter))
                mem_counter += 1
                if self.config.layer1_neighborhood_expansion:
                    j = 0
                    for neighbor_idx in getattr(note, 'links', []):
                        if 0 <= neighbor_idx < len(all_memories):
                            parts.append(fmt(all_memories[neighbor_idx], mem_counter))
                            mem_counter += 1
                            j += 1
                        if j >= k:
                            break
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Layer 4: Graph expansion
    # ------------------------------------------------------------------

    def graph_expand_retrieval(
        self,
        seed_event_ids: List[str],
        memory_system,
        original_indices: np.ndarray,
    ) -> np.ndarray:
        """Use graph structure to discover additional notes beyond neural retrieval.

        Conservative expansion: 1 hop bidirectional from seed events,
        capped at max_expand_notes new notes to avoid context dilution.

        NOTE: The cap check at lines below is on the outer eid/neighbor loops,
        NOT on the inner `for nid in event.source_note_ids` loop. Because
        session-based event extraction emits events with multiple source_notes
        (LoCoMo mean=22.6, LongMemEval mean=10.9 source_notes/event), a single
        inner-loop iteration can append more than max_expand_notes indices,
        relaxing the documented cap. Treated as documented design intent: the
        published TRACE main-table numbers were measured WITH this behavior,
        so adding the inner cap check would invalidate the paper baseline
        without an offsetting rerun. Do not change without rerunning the
        baseline.
        """
        max_expand_notes = 8

        all_memories = list(memory_system.memories.values())
        note_id_to_idx = {note.id: i for i, note in enumerate(all_memories)}

        original_set = set(int(idx) for idx in original_indices)
        new_indices = []

        # 1-hop bidirectional from seeds
        for eid in seed_event_ids:
            if len(new_indices) >= max_expand_notes:
                break
            # Forward edges
            for neighbor in self.graph.graph.successors(eid):
                if len(new_indices) >= max_expand_notes:
                    break
                event = self.graph.get_event(neighbor)
                if event:
                    for nid in event.source_note_ids:
                        idx = note_id_to_idx.get(nid)
                        if idx is not None and idx not in original_set:
                            original_set.add(idx)
                            new_indices.append(idx)
            # Backward edges
            for neighbor in self.graph.graph.predecessors(eid):
                if len(new_indices) >= max_expand_notes:
                    break
                event = self.graph.get_event(neighbor)
                if event:
                    for nid in event.source_note_ids:
                        idx = note_id_to_idx.get(nid)
                        if idx is not None and idx not in original_set:
                            original_set.add(idx)
                            new_indices.append(idx)

        logger.info(
            f"Graph expansion: {len(original_indices)} + {len(new_indices)} new = "
            f"{len(original_indices) + len(new_indices)} indices"
        )
        # Preserve original relevance ranking, append new indices at end
        if new_indices:
            return np.concatenate([original_indices, np.array(new_indices)])
        return original_indices

    # ------------------------------------------------------------------
    # Layer 5: Query entity extraction
    # ------------------------------------------------------------------

    def _extract_entity_seeds(self, query: str, memory_system=None) -> List[str]:
        """Extract entities from query and find matching events in graph.

        Two lookup channels:
        1. Participant index: exact match on person names
        2. Concept search: embed concept terms, find events whose state_change
           has high cosine similarity (uses retriever's SentenceTransformer)
        """
        entities = self.entity_extractor.extract(query)
        if not entities:
            return []

        event_ids = []
        seen = set()

        # Channel 1: Participant index lookup (person names), capped per entity
        # Sort by semantic similarity to query (via cached event embeddings)
        # instead of recency, to prioritize answer-relevant events.
        MAX_PARTICIPANT_SEEDS_PER_ENTITY = 8
        for entity in entities:
            entity_lower = entity.lower()
            matching_eids = self.graph._participant_index.get(entity_lower, set())
            candidate_eids = [eid for eid in matching_eids if eid not in seen]
            if len(candidate_eids) > MAX_PARTICIPANT_SEEDS_PER_ENTITY:
                # Prefer similarity-based ranking if cache available
                if hasattr(self, '_event_desc_cache') and self._event_desc_cache is not None:
                    cache = self._event_desc_cache
                    eid_to_cache_idx = {eid: i for i, eid in enumerate(cache['event_ids'])}
                    retriever = getattr(memory_system, 'retriever', None) if memory_system else None
                    model = getattr(retriever, 'model', None) if retriever else None
                    if model is not None:
                        query_emb = model.encode([query])[0]
                        from numpy.linalg import norm as np_norm
                        q_norm = np_norm(query_emb)
                        if q_norm > 0:
                            def _sim(eid):
                                ci = eid_to_cache_idx.get(eid)
                                if ci is None:
                                    return 0.0
                                return float(cache['embeddings'][ci] @ query_emb / (cache['norms'][ci] * q_norm))
                            candidate_eids.sort(key=_sim, reverse=True)
                else:
                    # Fallback: sort by recency
                    candidate_eids.sort(
                        key=lambda eid: getattr(self.graph.get_event(eid), 'time_anchor', '') or '',
                        reverse=True,
                    )
                candidate_eids = candidate_eids[:MAX_PARTICIPANT_SEEDS_PER_ENTITY]
            for eid in candidate_eids:
                seen.add(eid)
                event_ids.append(eid)

        # Channel 2: Concept embedding search (non-person terms)
        # Use retriever's model to embed concept terms and match against
        # event state_change descriptions
        if memory_system is not None:
            retriever = getattr(memory_system, 'retriever', None)
            model = getattr(retriever, 'model', None) if retriever else None

            if model is not None:
                # Filter to non-person concepts (not already found via participant index)
                person_names = set()
                for entity in entities:
                    if self.graph._participant_index.get(entity.lower()):
                        person_names.add(entity.lower())

                concept_terms = [e for e in entities if e.lower() not in person_names]

                if concept_terms:
                    # Cache event embeddings (computed once per pipeline instance)
                    if not hasattr(self, '_event_desc_cache') or self._event_desc_cache is None:
                        all_events = self.graph.get_all_events()
                        if all_events:
                            descriptions = [ev.state_change for ev in all_events]
                            event_id_list = [ev.event_id for ev in all_events]
                            desc_embs = model.encode(descriptions)
                            from numpy.linalg import norm
                            norms = norm(desc_embs, axis=1)
                            norms[norms == 0] = 1.0
                            self._event_desc_cache = {
                                'event_ids': event_id_list,
                                'embeddings': desc_embs,
                                'norms': norms,
                            }
                            logger.info(f"Cached {len(all_events)} event description embeddings")
                        else:
                            self._event_desc_cache = None

                    if self._event_desc_cache is not None:
                        cache = self._event_desc_cache
                        concept_query = " ".join(concept_terms)
                        concept_emb = model.encode([concept_query])[0]
                        from numpy.linalg import norm as np_norm
                        concept_norm = np_norm(concept_emb)
                        if concept_norm > 0:
                            sims = cache['embeddings'] @ concept_emb / (cache['norms'] * concept_norm)

                            # Top-8 most similar events
                            top_k = min(8, len(sims))
                            top_indices = np.argsort(sims)[-top_k:][::-1]
                            for idx in top_indices:
                                if sims[idx] >= 0.3:
                                    eid = cache['event_ids'][idx]
                                    if eid not in seen:
                                        seen.add(eid)
                                        event_ids.append(eid)

        if event_ids:
            logger.info(f"Entity extraction found {len(event_ids)} events for entities: {entities}")
        return event_ids

    # ------------------------------------------------------------------
    # Method C: Hierarchical seed selection
    # ------------------------------------------------------------------

    def _hierarchical_seed_selection(
        self,
        query: str,
        memory_system,
        existing_seed_ids: List[str],
    ) -> List[str]:
        """Multi-granularity seed selection: Topic -> Session -> Event.

        Finds relevant topics first, drills down to sessions, then collects
        events as additional seeds. Provides seeds that embedding-based
        retrieval might miss.
        """
        if not self.graph._topics:
            return []

        retriever = getattr(memory_system, 'retriever', None)
        model = getattr(retriever, 'model', None) if retriever else None
        if model is None:
            return []

        query_emb = model.encode([query])[0]
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            return []

        # Step 1: Score topics by query relevance
        topic_scores = []
        for tid, tdata in self.graph._topics.items():
            text_parts = [tdata.get("label", ""), tdata.get("description", "")]
            for sid in tdata.get("session_ids", []):
                sdata = self.graph._sessions.get(sid, {})
                text_parts.append(sdata.get("summary", "")[:200])
            topic_text = " ".join(text_parts)

            topic_emb = model.encode([topic_text])[0]
            t_norm = np.linalg.norm(topic_emb)
            if t_norm > 0:
                sim = float(np.dot(query_emb, topic_emb) / (q_norm * t_norm))
                topic_scores.append((tid, sim))

        if not topic_scores:
            return []

        # Step 2: Select top-2 topics
        topic_scores.sort(key=lambda x: -x[1])
        top_topics = topic_scores[:2]

        # Step 3: Drill down to sessions within selected topics
        candidate_sids = []
        for tid, _ in top_topics:
            tdata = self.graph._topics[tid]
            candidate_sids.extend(tdata.get("session_ids", []))

        # Step 4: Score sessions by query relevance, pick top-3
        session_scores = []
        for sid in candidate_sids:
            sdata = self.graph._sessions.get(sid, {})
            summary = sdata.get("summary", "")
            if not summary:
                continue
            s_emb = model.encode([summary])[0]
            s_norm = np.linalg.norm(s_emb)
            if s_norm > 0:
                sim = float(np.dot(query_emb, s_emb) / (q_norm * s_norm))
                session_scores.append((sid, sim))

        session_scores.sort(key=lambda x: -x[1])
        top_sessions = session_scores[:3]

        # Step 5: Collect events from top sessions (capped to avoid flooding)
        MAX_HIERARCHICAL_SEEDS = 20
        existing_set = set(existing_seed_ids)
        new_event_ids = []
        for sid, _ in top_sessions:
            if len(new_event_ids) >= MAX_HIERARCHICAL_SEEDS:
                break
            sdata = self.graph._sessions.get(sid, {})
            for eid in sdata.get("event_ids", []):
                if len(new_event_ids) >= MAX_HIERARCHICAL_SEEDS:
                    break
                if eid not in existing_set and eid in self.graph._events:
                    existing_set.add(eid)
                    new_event_ids.append(eid)

        if new_event_ids:
            logger.info(
                f"Hierarchical seeds: {len(top_topics)} topics -> "
                f"{len(top_sessions)} sessions -> {len(new_event_ids)} new events"
            )
        return new_event_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_embeddings(memory_system) -> Optional[np.ndarray]:
        """Extract embeddings array from memory system's retriever."""
        retriever = getattr(memory_system, 'retriever', None)
        if retriever is None:
            return None
        return getattr(retriever, 'embeddings', None)
