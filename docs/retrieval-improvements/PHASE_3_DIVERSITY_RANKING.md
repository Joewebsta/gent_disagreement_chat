# Phase 3: Diversity Ranking (Optional)

**Date:** January 2025  
**Version:** 1.0  
**Status:** Not Implemented (Optional)  
**Part of:** Simplified RAG Retrieval Implementation Plan

## Overview

This phase focuses on reducing redundancy in results using Maximal Marginal Relevance (MMR) algorithm. This is an **optional** phase that should only be implemented if redundancy becomes a real problem in retrieval results.

**Estimated Effort:** Week 2 (Days 11-14)  
**Expected Impact:** Reduces redundant results, improves coverage for multi-episode topics

**Prerequisites:** Phase 1 and Phase 2 should be completed first

**Note:** This phase is **optional** - only implement if redundancy is a real problem

---

## Goal

Reduce redundancy in results using Maximal Marginal Relevance (MMR) algorithm.

---

## Implementation Steps

### 3.1 Implement MMR Algorithm

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

### 3.2 Integrate MMR into RAGService

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

---

## Testing

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
├── rag_service.py                (enhanced - adds MMR integration)
├── vector_search.py              (no changes)
├── query_parser.py               (no changes)
└── diversity_ranking.py         (new, optional)
```

---

## Success Metrics

### Quantitative

- Diversity metrics: Episode distribution across results
- Diversity metrics: Speaker distribution across results
- Reduced redundancy: Fewer duplicate or highly similar segments

### Qualitative

- Better coverage for multi-episode topics (if MMR implemented)
- More diverse examples in analytical answers
- Reduced repetitive content in topical answers

---

## Implementation Status

⏸️ **Phase 3: Diversity Ranking** - Optional (Not Implemented)

- MMR algorithm not yet implemented
- Can be added if redundancy becomes an issue

---

## Decision Criteria

**Implement MMR if:**

- Users report redundant results for topical/analytical questions
- Analysis shows same episodes appearing multiple times unnecessarily
- Diversity metrics indicate poor episode/speaker distribution

**Skip MMR if:**

- Current results are already diverse enough
- Redundancy is not a reported issue
- Phase 1 and Phase 2 improvements are sufficient

---

## Related Documents

- `SIMPLIFIED_RAG_IMPLEMENTATION_PLAN.md` - Full implementation plan
- `PHASE_1_QUERY_UNDERSTANDING_AND_METADATA_FILTERING.md` - Previous phase
- `PHASE_2_ADAPTIVE_PARAMETERS.md` - Previous phase
- `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES_V2.md` - Full analysis of question types and retrieval strategies
- `CURRENT_RETRIEVAL_PROCESS_ANALYSIS_V2.md` - Current system analysis
- `EVALUATION_STRATEGIES.md` - Evaluation framework
