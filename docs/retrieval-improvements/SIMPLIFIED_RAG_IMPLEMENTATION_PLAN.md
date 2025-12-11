# Simplified RAG Retrieval Implementation Plan

**Date:** January 2025  
**Version:** 1.0  
**Status:** Implementation Complete

## Overview

This plan focuses on implementing the **80/20 improvements** that solve most question type issues with minimal complexity. Instead of implementing 7 different retrieval strategies, we enhance the existing single-query approach with:

1. **Metadata Filtering** - SQL WHERE clauses for episode, speaker, and date
2. **Adaptive Parameters** - Dynamic doc limits and thresholds based on question type
3. **Optional Diversity Ranking** - MMR algorithm to reduce redundancy (if needed)

**Estimated Total Effort:** 1-2 weeks  
**Expected Impact:** Solves ~80% of question type issues

---

## Phase 1: Query Understanding & Metadata Filtering (Week 1)

### Goal

Extract structured information from queries and filter by metadata before vector search.

### Implementation Steps

#### 1.1 Create Query Parser Module

**File:** `server/src/gent_disagreement_chat/core/query_parser.py`

**Purpose:** Extract episode numbers, speaker names, and date ranges from natural language queries.

**Key Functions:**

- `extract_episode_number(query)` - Regex patterns for "episode 180", "ep 180", "#180"
- `extract_speaker(query)` - Pattern matching for "Ricky", "Brendan", "hosts", "guest"
- `extract_date_range(query)` - Basic date parsing for "in 2023", "last 3 months", "since 2024"
- `classify_question_type(query)` - Simple heuristics to determine question type
- `extract_filters(query)` - Main method that extracts all filters

**Example Implementation:**

```python
def extract_episode_number(query):
    patterns = [
        r"episode\s+(\d+)",
        r"ep\s+(\d+)",
        r"#(\d+)",
        r"episode\s+number\s+(\d+)",
        r"ep\.\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.I)
        if match:
            return int(match.group(1))
    return None
```

**Speaker Mappings:**

- "ricky" → "Ricky Ghoshroy"
- "brendan" → "Brendan Kelly"
- "hosts"/"host"/"both" → None (special case for both hosts)
- "guest" → "guest"
- "professor" → "professor"

**Date Range Patterns:**

- "in 2023" → (2023-01-01, 2023-12-31)
- "last 3 months" → (3 months ago, today)
- "since 2024" → (2024-01-01, today)
- "between 2023 and 2024" → (2023-01-01, 2024-12-31)

**Question Type Classification:**

- `episode_scoped` - Has episode number
- `speaker_specific` - Mentions specific speaker
- `temporal` - Has date range
- `comparative` - Contains comparison keywords
- `factual` - Contains factual lookup keywords
- `analytical` - Contains analytical keywords
- `topical` - Default for general questions

#### 1.2 Enhance VectorSearch with Filtering

**File:** `server/src/gent_disagreement_chat/core/vector_search.py`

**Changes:**

- Add `find_relevant_with_filters()` method that accepts filter dictionary
- Build dynamic WHERE clauses based on filters
- Maintain backward compatibility with existing `find_relevant_above_adaptive_threshold()`

**SQL Pattern:**

```sql
SELECT ...
FROM transcript_segments ts
JOIN episodes e ON ts.episode_id = e.episode_number
JOIN speakers s ON ts.speaker_id = s.id
WHERE 1 - (ts.embedding <=> %s::vector) >= %s
  AND e.episode_number = %s  -- if episode filter
  AND s.name = %s              -- if speaker filter
  AND e.date_published BETWEEN %s AND %s  -- if date filter
ORDER BY similarity DESC
LIMIT %s
```

**Method Signature:**

```python
def find_relevant_with_filters(
    self,
    query,
    filters=None,
    min_docs=3,
    max_docs=10,
    similarity_threshold=0.6
):
```

**Filter Dictionary Structure:**

```python
filters = {
    'episode_number': 180,  # int
    'speaker': 'Ricky Ghoshroy',  # str
    'date_range': (start_date, end_date)  # tuple of datetime objects
}
```

**Implementation Details:**

1. Build dynamic WHERE clause based on provided filters
2. Use same adaptive threshold approach (try threshold first, fallback if needed)
3. Apply filters in both threshold query and fallback query
4. Maintain parameterized queries for SQL injection prevention

#### 1.3 Update RAGService to Use Query Parser

**File:** `server/src/gent_disagreement_chat/core/rag_service.py`

**Changes:**

- Import and instantiate QueryParser in `__init__`
- In `ask_question_text_stream()`, parse query before search
- Pass filters to vector search method
- Keep existing flow for backward compatibility

**Integration Point:**

```python
def ask_question_text_stream(self, question, model="gpt-4o-mini-2024-07-18"):
    # Parse query to extract filters
    filters = self.query_parser.extract_filters(question)
    question_type = self.query_parser.classify_question_type(question)

    # Get adaptive parameters
    params = self._get_adaptive_params(question_type, filters)

    # Search with filters
    search_results = self.vector_search.find_relevant_with_filters(
        question,
        filters=filters,
        **params
    )
    # ... rest of existing code
```

### Testing

- Test episode-scoped queries: "Summarize episode 180"
- Test speaker-specific queries: "What does Ricky think about tariffs?"
- Test temporal queries: "What did they discuss in 2023?"
- Verify backward compatibility (queries without filters still work)
- Test combined filters: "What did Ricky say in episode 180?"

---

## Phase 2: Adaptive Parameters (Week 1-2)

### Goal

Adjust `max_docs`, `min_docs`, and `similarity_threshold` based on question type.

### Implementation Steps

#### 2.1 Create Parameter Configuration

**File:** `server/src/gent_disagreement_chat/core/rag_service.py`

**Add Configuration Dictionary:**

```python
QUESTION_TYPE_PARAMS = {
    'episode_scoped': {
        'min_docs': 5,
        'max_docs': 30,
        'similarity_threshold': 0.4,  # Lower threshold for episode summaries
    },
    'speaker_specific': {
        'min_docs': 3,
        'max_docs': 15,
        'similarity_threshold': 0.5,
    },
    'temporal': {
        'min_docs': 5,
        'max_docs': 20,
        'similarity_threshold': 0.5,
    },
    'factual': {
        'min_docs': 1,
        'max_docs': 5,
        'similarity_threshold': 0.5,
    },
    'topical': {
        'min_docs': 3,
        'max_docs': 10,
        'similarity_threshold': 0.6,
    },
    'analytical': {
        'min_docs': 5,
        'max_docs': 25,
        'similarity_threshold': 0.5,
    },
    'comparative': {
        'min_docs': 5,
        'max_docs': 20,
        'similarity_threshold': 0.6,
    },
    'default': {
        'min_docs': 3,
        'max_docs': 10,
        'similarity_threshold': 0.6,
    }
}
```

**Rationale:**

- **Episode-scoped**: Need more segments (20-30) for complete summaries, lower threshold (0.4) to include all relevant content
- **Speaker-specific**: Moderate number (10-15) to capture speaker's views without overwhelming
- **Temporal**: Moderate number (15-20) to show evolution over time
- **Factual**: Few segments (3-5) for precise answers
- **Topical**: Standard parameters (3-10) for general questions
- **Analytical**: More segments (20-25) to capture patterns and themes
- **Comparative**: Moderate number (15-20) to represent both sides

#### 2.2 Add Parameter Selection Method

**File:** `server/src/gent_disagreement_chat/core/rag_service.py`

**Method:**

```python
def _get_adaptive_params(self, question_type, filters):
    """Get adaptive parameters based on question type and filters."""
    # Episode-scoped questions need more docs
    if filters.get('episode_number'):
        return self.QUESTION_TYPE_PARAMS['episode_scoped']

    # Use question type params or default
    return self.QUESTION_TYPE_PARAMS.get(
        question_type,
        self.QUESTION_TYPE_PARAMS['default']
    )
```

**Priority Logic:**

1. If episode number filter exists → use `episode_scoped` params (highest priority)
2. Otherwise, use params for classified question type
3. Fallback to `default` if question type not recognized

### Testing

- Verify episode-scoped queries return 20-30 segments
- Verify factual queries return 3-5 segments
- Verify default behavior unchanged for unclassified queries
- Test parameter selection priority (episode filter overrides question type)

---

## Phase 3: Diversity Ranking (Optional, Week 2)

### Goal

Reduce redundancy in results using Maximal Marginal Relevance (MMR) algorithm.

### Implementation Steps

#### 3.1 Implement MMR Algorithm

**File:** `server/src/gent_disagreement_chat/core/diversity_ranking.py` (new file)

**Implementation:**

```python
def apply_mmr(results, lambda_param=0.7, max_results=None):
    """
    Apply Maximal Marginal Relevance to reduce redundancy.

    Args:
        results: List of results with 'similarity' and 'text' fields
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        max_results: Maximum number of results to return

    Returns:
        Re-ranked list of results
    """
    if not results or len(results) <= 1:
        return results

    selected = [results[0]]  # Start with most relevant
    remaining = results[1:]
    max_results = max_results or len(results)

    while len(selected) < max_results and remaining:
        best_score = -float('inf')
        best_idx = 0

        for i, candidate in enumerate(remaining):
            # Relevance score
            relevance = candidate['similarity']

            # Diversity score (max similarity to already selected)
            max_sim = max(
                _text_similarity(candidate['text'], sel['text'])
                for sel in selected
            )

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
```

**MMR Formula:**

```
MMR = λ * relevance - (1 - λ) * max_similarity_to_selected
```

Where:

- `λ = 0.7` means 70% weight on relevance, 30% on diversity
- Higher λ = more relevance-focused
- Lower λ = more diversity-focused

**Text Similarity Function:**

Need to implement `_text_similarity()` helper function that computes similarity between two text strings. Options:

- Simple word overlap
- Jaccard similarity
- Embedding-based similarity (more accurate but slower)

#### 3.2 Integrate MMR into RAGService

**File:** `server/src/gent_disagreement_chat/core/rag_service.py`

**Conditional Application:**

- Apply MMR only for topical and analytical questions
- Skip for episode-scoped, speaker-specific, and factual questions (they need all relevant results)

**Integration:**

```python
# After retrieval, before formatting
if question_type in ['topical', 'analytical']:
    from .diversity_ranking import apply_mmr
    search_results = apply_mmr(
        search_results,
        lambda_param=0.7,
        max_results=params['max_docs']
    )
```

**When to Apply:**

- ✅ **Apply MMR for:**
  - Topical questions (to ensure diverse episode representation)
  - Analytical questions (to capture varied examples)
- ❌ **Skip MMR for:**
  - Episode-scoped (want all segments from that episode)
  - Speaker-specific (want all relevant segments from that speaker)
  - Factual (want most relevant, not diverse)
  - Temporal (want chronological representation)

### Testing

- Verify topical queries show more diverse episodes
- Verify episode-scoped queries still return all relevant segments
- Measure diversity metrics (episode distribution, speaker distribution)
- Test MMR with different lambda values (0.5, 0.7, 0.9)

---

## File Structure Changes

```
server/src/gent_disagreement_chat/core/
├── __init__.py
├── database_manager.py          (no changes)
├── embedding_service.py          (no changes)
├── rag_service.py                (enhanced)
├── vector_search.py              (enhanced)
├── query_parser.py               (new)
└── diversity_ranking.py         (new, optional)
```

---

## Implementation Order

1. **Day 1-2:** Query parser module (episode, speaker extraction)
2. **Day 3-4:** VectorSearch filtering (SQL WHERE clauses)
3. **Day 5:** RAGService integration (parse query, pass filters)
4. **Day 6-7:** Adaptive parameters (question type classification, param selection)
5. **Day 8-10:** Testing and refinement
6. **Day 11-14:** Optional MMR implementation (if needed)

---

## Success Metrics

### Quantitative

- Episode-scoped queries: Return segments only from specified episode
- Speaker-specific queries: Return segments only from specified speaker
- Temporal queries: Return segments within date range
- Document limits: Appropriate number of segments per question type
- Filter accuracy: Correct extraction of episode numbers, speakers, dates

### Qualitative

- User satisfaction with episode summaries (should improve significantly)
- User satisfaction with speaker-specific answers (should improve)
- Reduced "wrong episode" or "wrong speaker" errors
- Better coverage for multi-episode topics (if MMR implemented)

---

## Rollback Plan

- Keep existing `find_relevant_above_adaptive_threshold()` method unchanged
- Add new methods alongside existing ones
- Can revert by simply not calling query parser (fallback to default behavior)
- All changes are backward compatible

**Fallback Strategy:**

If query parsing fails or returns no filters, system falls back to:

- Default parameters (min_docs=3, max_docs=10, threshold=0.6)
- No filtering (searches all episodes)
- Existing behavior maintained

---

## Future Enhancements (Not in This Plan)

These are explicitly **not** included in this simplified plan but can be added later if needed:

- **Hybrid search (BM25 + vector)** - Only if factual lookups are common
- **Multi-faceted retrieval** - Only if comparative questions are frequent
- **Two-stage hierarchical retrieval** - Only if broad topical questions need it
- **Iterative/adaptive retrieval** - Likely overkill for current use case
- **Query enhancement (HyDE, expansion)** - Can be added if needed
- **LLM-based intent classification** - More accurate than heuristics

---

## Notes

- This plan prioritizes **simplicity and maintainability** over complex multi-strategy approaches
- All changes are **backward compatible** - existing queries continue to work
- Implementation is **incremental** - can stop after Phase 1 or Phase 2 if sufficient
- MMR (Phase 3) is **optional** - only implement if redundancy is a real problem
- Query parser uses **simple heuristics** - can be enhanced with LLM-based classification later
- Filter extraction is **conservative** - only extracts when confident (reduces false positives)

---

## Implementation Status

✅ **Phase 1: Query Understanding & Metadata Filtering** - Complete

- Query parser module created
- VectorSearch filtering implemented
- RAGService integration complete

✅ **Phase 2: Adaptive Parameters** - Complete

- Parameter configuration added
- Parameter selection method implemented
- Integrated into RAGService flow

⏸️ **Phase 3: Diversity Ranking** - Optional (Not Implemented)

- MMR algorithm not yet implemented
- Can be added if redundancy becomes an issue

---

## Related Documents

- `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES_V2.md` - Full analysis of question types and retrieval strategies
- `CURRENT_RETRIEVAL_PROCESS_ANALYSIS_V2.md` - Current system analysis
- `EVALUATION_STRATEGIES.md` - Evaluation framework
