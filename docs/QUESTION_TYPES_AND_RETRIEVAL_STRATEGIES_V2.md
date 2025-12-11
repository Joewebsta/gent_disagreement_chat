# Question Types and Retrieval Strategies Analysis

**Date:** January 2025  
**Version:** 2.0 (Accurate to Current Implementation)  
**Purpose:** Comprehensive analysis of question types users can ask and the optimal retrieval strategies needed to answer them effectively.

## Executive Summary

The current RAG system uses a **simple one-size-fits-all** retrieval strategy that works reasonably well for general topical questions but struggles with episode-scoped, speaker-specific, temporal, comparative, and factual queries. This document categorizes the full range of question types and identifies the specific retrieval methods needed for each.

**Key Finding:** The system uses a straightforward single-query vector search approach with fixed parameters. It does not currently implement query enhancement, intent classification, or metadata filtering. Implementing intent-aware routing and metadata filtering could dramatically improve answer quality across all question types.

---

## Question Type Taxonomy

### 1. Episode-Scoped Questions

Questions that explicitly reference a specific episode or bounded set of episodes.

**Examples:**

- "Summarize episode 180"
- "What did they discuss in the latest episode?"
- "What happened in episodes 150-160?"
- "Tell me about the episode from January 2024"

**Current System Limitation:**

- No episode-specific filtering before semantic search
- Will retrieve semantically similar content from ANY episode
- For "summarize episode 180," it might return segments from episodes 45, 92, and 180 based purely on semantic similarity to the word "summarize"
- Only returns 10 segments maximum (insufficient for full episode summary)
- No query parsing to extract episode numbers or date ranges

**Optimal Retrieval Method:**

- **Structured query parsing** to extract episode constraints (episode numbers, ranges, "latest")
- **Metadata filtering** BEFORE vector search using SQL WHERE clause on `episode_number` or `date_published`
- **Different ranking strategy**: Temporal ordering within episode, not just similarity
- **Higher document limit**: May need 20-50 segments to summarize a full episode
- **Possibly no similarity threshold**: Want ALL segments from that episode, not just semantically similar ones

**Implementation Priority:** HIGH - Common user question type with clear solution path

---

### 2. Multi-Episode Topical Questions

Questions asking about a topic discussed across many or all episodes.

**Examples:**

- "Tell me about all the times the hosts discuss tariffs"
- "What are their overall views on constitutional law?"
- "How do they analyze Supreme Court cases?"
- "What do they think about economics?"

**Current System Limitation:**

- Returns only top 10 segments across ALL episodes (fixed `max_docs=10`)
- May miss important discussions if >10 episodes discuss the topic
- Episode grouping helps organize results but still limited by `max_docs=10`
- No awareness of how many episodes cover the topic
- Results may cluster in 1-2 episodes, missing diversity across the corpus

**Optimal Retrieval Method:**

- **Higher document retrieval** (20-50 segments for corpus-wide topics)
- **Episode-level aggregation**: Find all episodes about topic, then select best segments per episode
- **Diversity-aware ranking**: Ensure representation across multiple episodes, not clustering in 2-3 episodes
- **Temporal distribution awareness**: Sample from different time periods
- **Two-stage retrieval:**
  1. Find relevant episodes (using episode-level embeddings or aggregated scores)
  2. Extract best segments from top N episodes (e.g., top 2 segments from each of 10 episodes)

**Implementation Priority:** MEDIUM - Current system works but could be significantly improved

---

### 3. Speaker-Specific Questions

Questions about what a specific host or guest said.

**Examples:**

- "What does Ricky think about economics?"
- "What are Brendan's views on the Supreme Court?"
- "What did the law professor guest say?"
- "Has Ricky ever discussed tariffs?"

**Current System Limitation:**

- Has speaker metadata (`name` field) in results but doesn't use it for filtering
- Returns segments from both speakers mixed together
- LLM has to filter speaker-specific content in-context (inefficient, may miss content)
- No structured guest metadata to identify "law professor" vs hosts
- No query parsing to extract speaker names from natural language

**Optimal Retrieval Method:**

- **Speaker filtering** in SQL query: `WHERE speakers.name = 'Ricky Ghoshroy'`
- **Speaker name entity extraction** from query using NER or pattern matching
- **Speaker-specific embeddings** or namespace filtering in vector search
- **Guest identification**: Structured metadata about which episodes have guests and their expertise
- **For "both hosts" questions**: Separate retrieval for each speaker, then compare/contrast

**Implementation Priority:** HIGH - Clear user need, straightforward implementation via SQL filtering

---

### 4. Comparative/Contrasting Questions

Questions asking to compare viewpoints, topics, or time periods.

**Examples:**

- "Do the hosts disagree on tariffs?"
- "Compare their views on Trump vs Biden"
- "How do their stances differ on constitutional interpretation?"
- "What changed between 2023 and 2024 in their economic analysis?"

**Current System Limitation:**

- No awareness that this is a comparative query
- Returns mixed segments without ensuring representation of different viewpoints
- May return 10 segments all from Ricky, none from Brendan
- May return all Trump segments, no Biden segments
- LLM has to work with biased/unbalanced retrieval results
- No balanced sampling across comparison dimensions

**Optimal Retrieval Method:**

- **Intent-aware retrieval**: Detect comparative intent (requires intent classification implementation)
- **Balanced sampling**: Ensure equal representation of both comparison dimensions
  - For speaker comparison: 50% Ricky, 50% Brendan
  - For topic comparison: 50% topic A, 50% topic B
  - For temporal comparison: 50% time period 1, 50% time period 2
- **Disagreement detection**: Find segments where both hosts discuss same topic in same episode
- **Multi-faceted retrieval**: Separate searches for each comparison dimension
  - Example: Search "Trump" + "Ricky", search "Trump" + "Brendan", search "Biden" + "Ricky", search "Biden" + "Brendan"
  - Combine with quotas to ensure balance

**Implementation Priority:** MEDIUM-HIGH - High user value, requires intent classification and multi-faceted retrieval

---

### 5. Temporal/Chronological Questions

Questions about how things changed over time or what happened in a time period.

**Examples:**

- "What have they discussed in the last 3 months?"
- "How has their view on tariffs evolved over time?"
- "What were their predictions in 2023?"
- "Track their discussion of inflation from 2020 to now"

**Current System Limitation:**

- Has `date_published` metadata but doesn't use it for filtering or ranking
- No temporal filtering in retrieval
- Returns segments sorted by similarity, not time
- Can't answer "how has X changed" because temporal ordering is lost
- No time range parsing from natural language ("last 3 months", "in 2023")

**Optimal Retrieval Method:**

- **Date range filtering** in SQL: `WHERE date_published BETWEEN X AND Y`
- **Natural language date parsing**: "last 3 months" → calculate date range
- **Temporal ordering** as secondary sort (after similarity or instead of it)
- **Time-aware chunking**: Group results by time periods (quarters, years) for "evolution" questions
- **Trend detection**: Multiple searches across time slices, then compare
- **Recency boosting**: Optionally weight recent content higher for "current views" questions

**Implementation Priority:** MEDIUM - Requires date parsing logic but high value for understanding evolution

---

### 6. Factual Lookup Questions

Questions with a specific factual answer findable in transcripts.

**Examples:**

- "Which episode discussed the Chevron case?"
- "When did they last talk about inflation?"
- "What episode number featured Professor Smith?"
- "Have they ever discussed Bitcoin?"

**Current System Limitation:**

- Pure semantic search may miss exact keyword matches
- No hybrid search (BM25 + vector)
- "Chevron case" might not semantically match if transcripts say "Chevron v. Natural Resources Defense Council"
- May return "topically related" content instead of exact mentions
- No exact phrase matching capability

**Optimal Retrieval Method:**

- **Hybrid search**: BM25/keyword matching + vector search, then fuse rankings
- **Entity recognition**: Extract entities (case names, people, dates) and do exact matching
- **Metadata indexing**: Episode titles, guest names in structured fields for direct lookup
- **Lower similarity threshold**: Accept lower semantic scores if keyword match is strong
- **Different response format**: Return episode list with citations, not full segment text
- **Exact phrase matching**: PostgreSQL full-text search for quoted phrases

**Implementation Priority:** MEDIUM - Adds capability current system lacks, moderate complexity

---

### 7. Analytical/Opinion Questions

Questions asking for analysis, interpretation, or deep understanding.

**Examples:**

- "Why do the hosts focus on constitutional law?"
- "What underlying philosophy drives their economic analysis?"
- "How do they approach disagreement?"
- "What patterns emerge in their political commentary?"

**Current System Limitation:**

- These need broad context, not just similar segments
- Current 10-segment limit may be too narrow for deep patterns
- Needs representative sampling across many discussions
- May need meta-level analysis the transcripts don't explicitly state
- Answers often require synthesis across multiple episodes and topics

**Optimal Retrieval Method:**

- **Broader retrieval**: 20-50 segments to capture patterns
- **Diversity maximization**: Ensure varied examples across episodes, topics, time
- **Pattern detection**: Find recurring themes/phrases across segments
- **Multi-hop retrieval**: Use initial results to inform second retrieval round
- **Abstract/summary retrieval**: If episode summaries exist, search those first for high-level themes
- **Clustering**: Group semantically similar segments to identify recurring patterns

**Implementation Priority:** LOW-MEDIUM - Complex to implement, current system partially handles this

---

### 8. Summarization Questions

Questions asking for summaries or overviews.

**Examples:**

- "Summarize the key themes of the podcast"
- "What are the main topics they cover?"
- "Give me an overview of their discussion style"
- "What topics do they return to most often?"

**Current System Limitation:**

- Segment-level retrieval doesn't capture high-level themes
- Need episode-level or corpus-level summaries
- Top 10 segments can't represent entire podcast corpus
- No frequency/popularity metrics for topics
- No pre-computed summaries or topic models

**Optimal Retrieval Method:**

- **Hierarchical retrieval**: Episode summaries → topic summaries → corpus summary
- **Clustering**: Group episodes by topic, return cluster representatives
- **Separate summary index**: Pre-generated summaries with their own embeddings
- **MapReduce approach**: Summarize episodes individually, then combine summaries
- **No similarity threshold**: Want representative sampling, not just most similar
- **Topic modeling**: Use LDA or BERTopic to identify main themes
- **Frequency analysis**: Track which topics/entities appear most often

**Implementation Priority:** LOW - Requires significant infrastructure (pre-computed summaries, topic models)

---

## Retrieval Method Categories

Based on the question types above, here are the distinct retrieval methods needed:

### Method 1: Filtered Vector Search

**Description:** Apply metadata filters BEFORE vector search, then rank by similarity within filtered set.

**Implementation:**

```sql
WHERE episode_number = X
WHERE speakers.name = 'Ricky Ghoshroy'
WHERE date_published BETWEEN '2024-01-01' AND '2024-12-31'
```

**Best for:** Episode-scoped, speaker-specific, temporal questions

**Complexity:** LOW - SQL filtering, already have indexes

**Impact:** HIGH - Solves multiple question types

---

### Method 2: Hybrid Search (Vector + Keyword)

**Description:** Combine BM25/keyword search with vector search, fuse rankings.

**Implementation:**

- PostgreSQL full-text search (ts_vector) for keyword component
- Vector search for semantic component
- Reciprocal Rank Fusion (RRF) to combine rankings
- Weight tuning (e.g., 70% vector, 30% keyword)

**Best for:** Factual lookup, named entity questions, exact phrase matching

**Complexity:** MEDIUM - Requires additional indexing and rank fusion logic

**Impact:** MEDIUM - Improves precision for factual queries

---

### Method 3: Multi-Faceted Retrieval

**Description:** Run separate searches for different facets of the query, combine with diversity constraints.

**Implementation:**

- Identify facets (speakers, topics, time periods)
- Separate retrieval for each facet
- Allocate quota per facet (e.g., 5 segments per speaker)
- Interleave results to ensure balanced representation

**Best for:** Comparative, speaker-specific, analytical questions

**Complexity:** MEDIUM - Requires facet extraction and quota management

**Impact:** HIGH - Dramatically improves comparative questions

---

### Method 4: Two-Stage Hierarchical Retrieval

**Description:** Stage 1 finds relevant episodes/documents, Stage 2 extracts best segments.

**Implementation:**

1. Episode-level scoring: Aggregate segment similarities or use episode embeddings
2. Select top N episodes (e.g., top 10)
3. Extract top K segments from each episode (e.g., top 3 per episode)
4. Ensures diverse episode representation

**Best for:** Multi-episode topical, summarization questions

**Complexity:** MEDIUM - Requires episode-level aggregation logic

**Impact:** MEDIUM-HIGH - Prevents clustering all results in 1-2 episodes

---

### Method 5: Diversity-Aware Retrieval

**Description:** Balance relevance with diversity using MMR or similar algorithms.

**Implementation:**

- Maximal Marginal Relevance (MMR): `λ * similarity - (1-λ) * max_similarity_to_selected`
- Iteratively select documents that are relevant but dissimilar to already-selected
- Ensures coverage of different aspects of topic

**Best for:** Multi-episode topical, analytical, comparative questions

**Complexity:** MEDIUM - MMR algorithm straightforward to implement

**Impact:** MEDIUM - Improves result diversity, reduces redundancy

---

### Method 6: Temporal-Ordered Retrieval

**Description:** Primary sort by time/date, secondary filter by similarity threshold.

**Implementation:**

- Parse date constraints from query
- Filter by date range
- Sort by `date_published` (ascending or descending)
- Apply minimum similarity threshold to ensure relevance

**Best for:** Temporal/chronological questions, "evolution" queries

**Complexity:** LOW-MEDIUM - Requires date parsing, simple sorting change

**Impact:** MEDIUM - Enables temporal analysis capabilities

---

### Method 7: Iterative/Adaptive Retrieval

**Description:** Use initial results to refine query, perform multi-hop reasoning.

**Implementation:**

1. Initial retrieval with original query
2. Extract key terms/entities from top results
3. Expand/refine query with extracted terms
4. Second retrieval with refined query
5. Combine results with deduplication

**Best for:** Complex analytical questions, multi-hop reasoning

**Complexity:** HIGH - Requires term extraction, query reformulation logic

**Impact:** MEDIUM - Improves complex queries, adds latency

---

## Current System Analysis

### What Works Well ✓

- **General topical questions** - Single-query vector search with adaptive thresholding works reasonably well
- **Episode grouping** - Hierarchical organization by episode in formatting is clear and helpful
- **Adaptive thresholding** - Two-query approach (threshold-first, fallback) ensures minimum context without overload
- **Speaker attribution** - Maintains speaker metadata through pipeline (available in results, used in formatting)
- **Simple and reliable** - Straightforward implementation with predictable behavior

### What Struggles ✗

- **Episode-scoped questions** - No pre-filtering by episode metadata
- **Speaker-specific questions** - No speaker filtering in retrieval
- **Temporal questions** - Doesn't use dates for filtering or ranking
- **Comparative questions** - No balanced sampling across facets
- **Factual lookup** - No keyword/exact match component
- **Questions needing >10 segments** - Fixed max limit too restrictive
- **Questions needing diversity** - No diversity-aware ranking
- **No query understanding** - No parsing of episode numbers, speaker names, dates, or intent

### Current System Flow

The system uses a **simple one-size-fits-all** retrieval strategy:

1. **Single query vector search** - Direct embedding search with no query enhancement
2. **Adaptive thresholding** - Try threshold-based retrieval (similarity >= 0.6), fallback to top-N if insufficient results
3. **Fixed parameters** - `min_docs=3`, `max_docs=10`, `threshold=0.6` for all queries
4. **Top 10 by similarity** - Results sorted by similarity score only
5. **Group by episode** - Post-retrieval grouping for formatting (doesn't affect retrieval)
6. **Pass to LLM** - Formatted context sent to GPT-4o-mini for response generation

**Actual Implementation Flow:**

```
User Query
    ↓
[RAGService] ask_question_text_stream()
    ↓
[VectorSearch] find_relevant_above_adaptive_threshold()
    ↓
    ├─→ [EmbeddingService] generate_embedding() → OpenAI API
    ├─→ [Query 1] Try threshold-based retrieval (similarity >= 0.6, LIMIT 10)
    └─→ [Query 2] Fallback to top-N if < min_docs (LIMIT 3)
    ↓
[Episode Grouping] group_by_episode() (post-retrieval formatting)
    ↓
[Context Formatting] _format_search_results()
    ↓
[LLM Streaming] _generate_simple_text_stream() → OpenAI GPT-4o-mini
```

This works reasonably for general topical questions but **fails to leverage**:

- Available metadata (episode_number, speaker name, date_published) for filtering
- Query parsing to extract structured constraints (episode numbers, speaker names, dates)
- Different ranking strategies for different question types
- Structured filtering capabilities of SQL
- Adaptive document limits based on question scope
- Intent classification or query understanding

---

## Recommendations Summary

To handle the full range of question types, the system needs:

### 1. Query Understanding Layer

**Purpose:** Extract structured information from natural language queries.

**Components:**

- **Episode extraction**: Parse episode numbers, ranges, "latest", "recent"
- **Speaker extraction**: Identify "Ricky", "Brendan", "hosts", "guest", "professor"
- **Date parsing**: "last 3 months", "in 2023", "since 2024", "between X and Y"
- **Entity recognition**: Named entities (people, court cases, organizations)
- **Intent classification**: Classify queries into categories (comparative, factual, analytical, etc.)
- **Comparison detection**: Identify comparison dimensions (speaker vs speaker, topic vs topic, time vs time)

**Implementation:**

- Regex patterns for episode numbers and common phrases
- Spacy or NLTK for NER
- dateparser library for natural language dates
- LLM-based or rule-based intent classification
- Pattern matching for comparison detection

---

### 2. Retrieval Strategy Router

**Purpose:** Route queries to appropriate retrieval method based on intent and extracted entities.

**Routing Logic:**

```python
if episode_number_extracted:
    → Episode-Scoped Retrieval (Method 1: Filtered Vector Search)
    → High doc limit (20-50), metadata filter, temporal ordering

elif speaker_extracted and query_is_comparative:
    → Multi-Faceted Retrieval (Method 3)
    → Balanced sampling per speaker

elif date_range_extracted:
    → Temporal-Ordered Retrieval (Method 6)
    → Date filter, temporal ordering

elif intent == "comparative":
    → Multi-Faceted Retrieval (Method 3)
    → Identify comparison dimensions, balanced sampling

elif intent == "factual":
    → Hybrid Search (Method 2)
    → BM25 + vector, favor exact matches

elif query_is_broad_topical:
    → Two-Stage Hierarchical Retrieval (Method 4)
    → Find episodes, then extract segments

else:
    → Current Single-Query Approach
    → With diversity-aware ranking (Method 5)
```

**Implementation:**

- Strategy pattern with different retrieval classes
- Configuration-driven routing rules
- Fallback to default strategy if routing uncertain

---

### 3. Enhanced Ranking

**Purpose:** Improve result quality with intent-aware ranking and diversity.

**Components:**

- **Diversity-aware ranking**: MMR algorithm to reduce redundancy
- **Intent-specific ranking weights**: Adjust similarity vs diversity vs recency
- **Metadata-aware scoring**: Boost recent episodes for "current views", balance speakers for comparative
- **Episode-level ranking**: Aggregate scores, prevent single-episode domination
- **Cross-encoder reranking**: Optional second-stage reranking for top results

**Implementation:**

- MMR algorithm (straightforward)
- Configurable ranking weights per intent
- Episode aggregation logic (already partially implemented in grouping)

---

### 4. Adaptive Parameters

**Purpose:** Adjust retrieval parameters based on question type.

**Parameters to Adapt:**

| Question Type    | Doc Limit | Similarity Threshold | Ranking Strategy             |
| ---------------- | --------- | -------------------- | ---------------------------- |
| Episode-scoped   | 20-50     | 0.3 (lower)          | Temporal ordering            |
| Speaker-specific | 10-20     | 0.5                  | Similarity + speaker balance |
| Temporal         | 15-30     | 0.5                  | Temporal ordering            |
| Comparative      | 10-20     | 0.6                  | Balanced sampling            |
| Factual          | 3-5       | 0.4                  | Keyword match priority       |
| Topical (broad)  | 20-40     | 0.6                  | Diversity-aware              |
| Topical (narrow) | 5-10      | 0.7                  | Similarity only              |
| Analytical       | 20-50     | 0.5                  | Diversity-aware              |
| Summarization    | 30-50     | 0.4                  | Representative sampling      |

**Implementation:**

- Configuration object per intent type
- Override defaults based on routing decision

---

## Implementation Roadmap

### Phase 1: Foundation (High Priority, Quick Wins)

**Goal:** Enable metadata filtering for episode-scoped and speaker-specific questions.

**Tasks:**

1. Add episode number extraction from queries (regex patterns)
2. Add speaker name extraction from queries (pattern matching)
3. Implement filtered vector search (Method 1) - add WHERE clauses to SQL
4. Route episode-scoped questions to filtered search
5. Route speaker-specific questions to filtered search
6. Add adaptive document limits based on question type

**Estimated Effort:** 1-2 weeks  
**Impact:** HIGH - Solves common pain points with straightforward SQL filtering

---

### Phase 2: Query Understanding and Intent-Aware Retrieval (High Priority)

**Goal:** Add query understanding and route queries to appropriate retrieval strategies.

**Tasks:**

1. Implement query understanding layer (episode, speaker, date extraction)
2. Add intent classification (LLM-based or rule-based)
3. Implement retrieval strategy router
4. Add comparative question handling (Method 3: Multi-faceted)
5. Add date parsing and temporal filtering (Method 6)
6. Implement adaptive parameters per intent
7. Add diversity-aware ranking (Method 5: MMR)

**Estimated Effort:** 3-4 weeks  
**Impact:** HIGH - Dramatically improves multiple question types

---

### Phase 3: Hybrid Search (Medium Priority)

**Goal:** Add keyword matching for factual lookup questions.

**Tasks:**

1. Add PostgreSQL full-text search indexes
2. Implement BM25 scoring
3. Implement rank fusion (RRF)
4. Route factual questions to hybrid search
5. Tune keyword vs vector weights

**Estimated Effort:** 2-3 weeks  
**Impact:** MEDIUM - New capability, moderate user demand

---

### Phase 4: Advanced Retrieval (Lower Priority)

**Goal:** Handle complex analytical and summarization questions.

**Tasks:**

1. Implement two-stage hierarchical retrieval (Method 4)
2. Add episode-level embeddings/scoring
3. Implement iterative retrieval (Method 7)
4. Add topic modeling for summarization
5. Pre-compute episode summaries

**Estimated Effort:** 4-6 weeks  
**Impact:** MEDIUM - Improves edge cases, high complexity

---

## Evaluation Metrics

To measure improvement after implementing these strategies:

### Per Question Type:

- **Relevance**: Are retrieved segments relevant to the query?
- **Coverage**: For multi-episode questions, are multiple episodes represented?
- **Balance**: For comparative questions, are both sides represented equally?
- **Precision**: For factual questions, is the answer in the top results?
- **Completeness**: For episode summaries, are key points covered?

### Overall Metrics:

- **Answer Quality**: LLM-as-judge scoring of final answers
- **User Satisfaction**: Explicit feedback on answers
- **Retrieval Diversity**: Measure episode/speaker distribution in results
- **Latency**: Impact of more complex retrieval on response time

### A/B Testing:

- Route 50% of queries to new strategy, 50% to baseline
- Compare answer quality, user satisfaction, latency
- Measure improvement per question type

---

## Conclusion

The current RAG system has a solid, simple foundation but uses a one-size-fits-all approach that limits its effectiveness across diverse question types. The biggest opportunity is leveraging already-available metadata (episode, speaker, date) and implementing query understanding to route queries to appropriate retrieval strategies.

**Quick wins:**

1. Metadata filtering for episode-scoped questions (HIGH impact, LOW effort)
2. Speaker filtering for speaker-specific questions (HIGH impact, LOW effort)
3. Query understanding and intent-aware routing (HIGH impact, MEDIUM effort)

**Long-term improvements:**

4. Hybrid search for factual queries (MEDIUM impact, MEDIUM effort)
5. Two-stage retrieval for broad topical questions (MEDIUM impact, MEDIUM effort)
6. Pre-computed summaries for corpus-level questions (MEDIUM impact, HIGH effort)

The evaluation framework already in place (`evaluate_simple.py`, evaluation results) provides the tooling to measure improvements systematically as these enhancements are implemented.
