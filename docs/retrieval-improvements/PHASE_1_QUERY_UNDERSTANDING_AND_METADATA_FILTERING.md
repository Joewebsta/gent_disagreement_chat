# Phase 1: Query Understanding & Metadata Filtering

**Date:** January 2025  
**Version:** 1.0  
**Status:** Implementation Complete  
**Part of:** Simplified RAG Retrieval Implementation Plan

## Overview

This phase focuses on extracting structured information from queries and filtering by metadata before vector search. This is the foundation for improving retrieval accuracy by ensuring results match user intent (episode, speaker, date constraints).

**Estimated Effort:** Week 1 (Days 1-5)  
**Expected Impact:** Solves episode-scoped, speaker-specific, and temporal query issues

---

## Goal

Extract structured information from queries and filter by metadata before vector search.

---

## Implementation Steps

### 1.1 Create Query Parser Module

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

### 1.2 Enhance VectorSearch with Filtering

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

### 1.3 Update RAGService to Use Query Parser

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

---

## Testing

- Test episode-scoped queries: "Summarize episode 180"
- Test speaker-specific queries: "What does Ricky think about tariffs?"
- Test temporal queries: "What did they discuss in 2023?"
- Verify backward compatibility (queries without filters still work)
- Test combined filters: "What did Ricky say in episode 180?"

---

## File Structure Changes

```
server/src/gent_disagreement_chat/core/
├── __init__.py
├── database_manager.py          (no changes)
├── embedding_service.py          (no changes)
├── rag_service.py                (enhanced)
├── vector_search.py              (enhanced)
└── query_parser.py               (new)
```

---

## Success Metrics

### Quantitative

- Episode-scoped queries: Return segments only from specified episode
- Speaker-specific queries: Return segments only from specified speaker
- Temporal queries: Return segments within date range
- Filter accuracy: Correct extraction of episode numbers, speakers, dates

### Qualitative

- Reduced "wrong episode" or "wrong speaker" errors
- User satisfaction with episode summaries (should improve significantly)
- User satisfaction with speaker-specific answers (should improve)

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

## Implementation Status

✅ **Phase 1: Query Understanding & Metadata Filtering** - Complete

- Query parser module created
- VectorSearch filtering implemented
- RAGService integration complete

---

## Related Documents

- `SIMPLIFIED_RAG_IMPLEMENTATION_PLAN.md` - Full implementation plan
- `PHASE_2_ADAPTIVE_PARAMETERS.md` - Next phase
- `PHASE_3_DIVERSITY_RANKING.md` - Optional phase
- `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES_V2.md` - Full analysis of question types and retrieval strategies
- `CURRENT_RETRIEVAL_PROCESS_ANALYSIS_V2.md` - Current system analysis
- `EVALUATION_STRATEGIES.md` - Evaluation framework
