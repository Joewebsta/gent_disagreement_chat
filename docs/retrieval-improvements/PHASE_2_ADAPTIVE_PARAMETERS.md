# Phase 2: Adaptive Parameters

**Date:** January 2025  
**Version:** 1.0  
**Status:** Implementation Complete  
**Part of:** Simplified RAG Retrieval Implementation Plan

## Overview

This phase focuses on adjusting `max_docs`, `min_docs`, and `similarity_threshold` based on question type. Different question types require different amounts of context - episode summaries need more segments, factual queries need fewer.

**Estimated Effort:** Week 1-2 (Days 6-7)  
**Expected Impact:** Improves answer quality by providing appropriate context volume for each question type

**Prerequisites:** Phase 1 must be completed (query parser and question type classification)

---

## Goal

Adjust `max_docs`, `min_docs`, and `similarity_threshold` based on question type.

---

## Implementation Steps

### 2.1 Create Parameter Configuration

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

### 2.2 Add Parameter Selection Method

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

---

## Testing

- Verify episode-scoped queries return 20-30 segments
- Verify factual queries return 3-5 segments
- Verify default behavior unchanged for unclassified queries
- Test parameter selection priority (episode filter overrides question type)

---

## File Structure Changes

```
server/src/gent_disagreement_chat/core/
├── __init__.py
├── database_manager.py          (no changes)
├── embedding_service.py          (no changes)
├── rag_service.py                (enhanced - adds parameter config)
├── vector_search.py              (no changes)
└── query_parser.py               (no changes)
```

---

## Success Metrics

### Quantitative

- Document limits: Appropriate number of segments per question type
  - Episode-scoped: 20-30 segments
  - Factual: 3-5 segments
  - Speaker-specific: 10-15 segments
  - Analytical: 20-25 segments
  - Default: 3-10 segments

### Qualitative

- Better answer quality for episode summaries (more complete context)
- More focused answers for factual queries (less noise)
- Appropriate context volume for each question type

---

## Rollback Plan

- Parameter configuration is additive - existing default behavior remains
- Can revert by removing parameter selection method call
- All changes are backward compatible
- Default parameters match original system behavior

**Fallback Strategy:**

If parameter selection fails or question type is unknown, system falls back to:

- Default parameters (min_docs=3, max_docs=10, threshold=0.6)
- Existing behavior maintained

---

## Implementation Status

✅ **Phase 2: Adaptive Parameters** - Complete

- Parameter configuration added
- Parameter selection method implemented
- Integrated into RAGService flow

---

## Related Documents

- `SIMPLIFIED_RAG_IMPLEMENTATION_PLAN.md` - Full implementation plan
- `PHASE_1_QUERY_UNDERSTANDING_AND_METADATA_FILTERING.md` - Previous phase (prerequisite)
- `PHASE_3_DIVERSITY_RANKING.md` - Next optional phase
- `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES_V2.md` - Full analysis of question types and retrieval strategies
- `CURRENT_RETRIEVAL_PROCESS_ANALYSIS_V2.md` - Current system analysis
- `EVALUATION_STRATEGIES.md` - Evaluation framework
