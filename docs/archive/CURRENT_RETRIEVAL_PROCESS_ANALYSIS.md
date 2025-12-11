# Current Retrieval Process: Complete Flow Analysis

## Key Takeaway

The retrieval system has a solid foundation with sophisticated query enhancement, but uses a one-size-fits-all execution strategy that doesn't leverage the available metadata or intent classification. The
biggest opportunity is routing different query types to specialized retrieval strategies - the infrastructure is already in place, it just needs to be connected!

## Overview

This document provides a detailed, step-by-step analysis of how the retrieval process currently works in the RAG system. It traces a query from the API endpoint through query enhancement, vector search, result formatting, and response generation.

**Date:** 2025-11-17
**System:** Gentleman's Disagreement Podcast RAG Chat Application

---

## Complete Retrieval Flow Diagram

```
User Query
    ↓
[API Endpoint] (/api/chat)
    ↓
[RAGService] ask_question_text_stream()
    ↓
┌─────────────────────────────────────┐
│  Query Enhancement Decision         │
│  (if use_query_enhancement=True)    │
└─────────────────────────────────────┘
    ↓
[QueryEnhancer] enhance_query()
    ↓
    ├─→ [Intent Classification] → intent type (9 categories)
    ├─→ [HyDE Generation] → hypothetical answer (GPT-4o-mini)
    ├─→ [Query Expansion] → expanded keywords
    └─→ [Query Cleaning] → normalized query
    ↓
[RAGService] _multi_query_search()
    ↓
    ├─→ [Search 1] Original query (min=3, max=10, threshold=0.6)
    ├─→ [Search 2] HyDE query (min=2, max=5, threshold=0.6)
    └─→ [Search 3] Expanded query (min=2, max=5, threshold=0.6) *conditional
    ↓
[VectorSearch] find_relevant_above_adaptive_threshold() ×3
    ↓
    ├─→ [EmbeddingService] generate_embedding() → OpenAI API
    ├─→ [Query 1] Try threshold-based retrieval (similarity >= 0.6)
    └─→ [Query 2] Fallback to top-N if < min_docs
    ↓
[Deduplication] Remove duplicate texts using set()
    ↓
[Sorting] Sort by similarity DESC, limit to max_docs (10)
    ↓
[Episode Grouping] group_by_episode()
    ↓
[Context Formatting] _format_search_results()
    ↓
[Prompt Creation] _create_prompt()
    ↓
[LLM Streaming] _generate_simple_text_stream() → OpenAI GPT-4o-mini
    ↓
Response Stream (text/plain)
    ↓
Client
```

---

## Detailed Step-by-Step Flow

### Step 1: API Entry Point

**File:** `main.py:32-73`

**Function:** `POST /api/chat`

**Input:**

```json
{
  "messages": [
    {
      "role": "user",
      "text": "What do the hosts think about tariffs?"
    }
  ]
}
```

**Process:**

1. **Extract messages** from request body
2. **Get last user message** (stateless - doesn't maintain conversation history)
3. **Extract text content** with dual-format support:
   - Primary: `message.text` (text streaming format)
   - Fallback: `message.parts[].text` (parts format)
4. **Validate** user text is not empty
5. **Instantiate RAGService** (new instance per request)
6. **Call** `rag_service.ask_question_text_stream(user_text)`
7. **Return** StreamingResponse with:
   - `media_type="text/plain"` (not JSON)
   - `Cache-Control: no-cache`
   - `Connection: keep-alive`

**Output:** StreamingResponse wrapping generator

---

### Step 2: RAG Service Orchestration

**File:** `rag_service.py:24-52`

**Function:** `ask_question_text_stream(question, model="gpt-4o-mini-2024-07-18")`

**Input:** `"What do the hosts think about tariffs?"`

**Process:**

1. **Check enhancement flag** (`use_query_enhancement`, default: True)
2. **Branch A** - If enhancement enabled:
   - Call `self.query_enhancer.enhance_query(question)`
   - Call `self._multi_query_search(enhanced_query)`
3. **Branch B** - If enhancement disabled:
   - Call `self.vector_search.find_relevant_above_adaptive_threshold()` directly
   - Use default params: `min_docs=3`, `max_docs=10`, `threshold=0.6`
4. **Format results** via `_format_search_results(search_results)`
5. **Create prompt** via `_create_prompt(formatted_results, question)`
6. **Generate stream** via `_generate_simple_text_stream(prompt, model)`

**Default configuration:**

```python
DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_DOCS = 3
DEFAULT_MAX_DOCS = 10
```

**Output:** Generator yielding text chunks

---

### Step 3: Query Enhancement

**File:** `query_enhancer.py:72-88`

**Function:** `enhance_query(question)`

**Input:** `"What do the hosts think about tariffs?"`

**Process:**

This method coordinates four enhancement strategies in parallel:

#### 3.1 Intent Classification

**Function:** `_classify_query_intent(question)` (line 142-209)

**Method:** Keyword-based pattern matching (not LLM)

**Logic:**

1. Convert query to lowercase
2. Check for keyword presence in ordered categories:
   - `"court", "constitutional", "legal"` → `"legal_analysis"`
   - `"tariff", "trade", "economy"` → `"economic_policy"` ✓ **MATCHES**
   - `"history", "past", "compare"` → `"historical_comparison"`
   - `"professor", "expert", "guest"` → `"expert_opinion"`
   - `"what", "who", "when", "where"` → `"factual"`
   - `"how", "why", "explain"` → `"analytical"`
   - `"compare", "versus", "difference"` → `"comparative"`
   - `"opinion", "think", "believe"` → `"opinion"`
   - Default → `"general"`

**Output for example:** `"economic_policy"`

**⚠️ Critical Note:** Intent is classified but **NOT USED** to adapt retrieval strategy!

---

#### 3.2 HyDE (Hypothetical Document Embeddings)

**Function:** `_generate_hypothetical_answer(question)` (line 90-122)

**Method:** LLM-based hypothetical answer generation

**Process:**

1. **Classify intent** (calls `_classify_query_intent()` again)
2. **Select intent-specific prompt** based on classification:
   - For `"economic_policy"` (our example):
     ```
     "Focus on economic analysis, policy implications, market trends,
     and data-driven discussion typical of economic experts."
     ```
3. **Construct full prompt:**
   ```
   Based on the question "What do the hosts think about tariffs?",
   generate a hypothetical answer that might appear in the
   "A Gentleman's Disagreement" podcast transcript.
   Focus on economic analysis, policy implications, market trends,
   and data-driven discussion typical of economic experts.
   Keep it concise (1-2 sentences) and match the intellectual,
   conversational tone of the podcast.
   ```
4. **Call OpenAI API:**
   - Model: `gpt-4o-mini`
   - Max tokens: 100
   - Temperature: 0.7 (creative but controlled)
5. **Graceful degradation:** On error, return original question

**Example output:**

```
"The hosts would likely discuss how tariffs impact both domestic
industries and consumer prices, weighing protectionist arguments
against free trade principles while analyzing recent policy decisions."
```

**Rationale:** Search for documents that are semantically similar to this hypothetical answer, which may better match actual transcript content than the question itself.

---

#### 3.3 Query Expansion

**Function:** `_expand_query_keywords(question)` (line 124-140)

**Method:** Vocabulary-based keyword expansion + synonyms

**Process:**

1. **Start with original terms:** `["What", "do", "the", "hosts", "think", "about", "tariffs?"]`

2. **Check vocabulary dictionaries** for matches:

   ```python
   podcast_vocabulary = {
       "hosts": ["Ricky Ghoshroy", "Ricky", "Brendan Kelly", ...],
       "legal_topics": ["Supreme Court", "SCOTUS", ...],
       "economic_topics": ["tariffs", "trade", "BLS", "Federal Reserve", ...], ✓
       ...
   }
   ```

3. **For our example:** "tariffs" matches `economic_topics`, so add:

   - "trade", "Bureau of Labor Statistics", "BLS", "Federal Reserve", "Fed", "markets", "inflation", "labor", "economy"

4. **Also:** "hosts" matches `hosts` category, so add:

   - "Ricky Ghoshroy", "Ricky", "Brendan Kelly", "Brendan", "speakers", "panelists"

5. **Add synonyms** via `_get_synonyms()`:

   ```python
   synonym_map = {
       "hosts": ["speakers", "panelists", "Ricky", "Brendan", ...],
       "opinion": ["view", "perspective", "stance", ...],
       "economy": ["economic", "markets", "trade", ...],
       ...
   }
   ```

   - "hosts" → adds: "speakers", "panelists", host names
   - "think" → matches "opinion" → adds: "view", "perspective", "stance", "position"

6. **Deduplicate** using `set()`
7. **Join** into single string

**Example output:**

```
"What do the hosts think about tariffs Ricky Ghoshroy Brendan Kelly
speakers panelists trade BLS Federal Reserve Fed markets inflation
labor economy view perspective stance position"
```

**Note:** This is a bag-of-words approach - no semantic structure preserved.

---

#### 3.4 Query Cleaning

**Function:** `_clean_and_normalize(question)` (line 211-233)

**Method:** Stop word removal + whitespace normalization

**Process:**

1. **Remove extra whitespace:** `re.sub(r"\s+", " ", question.strip())`
2. **Remove stop words:**
   ```python
   stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]
   ```
3. **Join remaining words**

**Example output:**

```
"What do hosts think about tariffs?"
```

**Note:** This cleaned version is returned but **NOT USED** in current retrieval!

---

**Final Enhanced Query Object:**

```python
{
    "original": "What do the hosts think about tariffs?",
    "hyde": "The hosts would likely discuss how tariffs impact both domestic industries...",
    "expanded": "What do the hosts think about tariffs Ricky Ghoshroy Brendan Kelly speakers...",
    "intent": "economic_policy",
    "cleaned": "What do hosts think about tariffs?"
}
```

---

### Step 4: Multi-Query Search

**File:** `rag_service.py:71-121`

**Function:** `_multi_query_search(enhanced_query)`

**Input:** Enhanced query object from Step 3

**Process:**

This performs **up to 3 separate vector searches** and combines results:

#### Search 1: Original Query

**Query text:** `"What do the hosts think about tariffs?"`

**Parameters:**

- `min_docs=3` (DEFAULT_MIN_DOCS)
- `max_docs=10` (DEFAULT_MAX_DOCS)
- `similarity_threshold=0.6` (DEFAULT_THRESHOLD)

**Process:**

1. Call `vector_search.find_relevant_above_adaptive_threshold()`
2. Add results to `all_results` list
3. Track seen texts in `seen_texts` set for deduplication

**Deduplication logic:**

```python
for result in original_results:
    if result["text"] not in seen_texts:
        all_results.append(result)
        seen_texts.add(result["text"])
```

---

#### Search 2: HyDE Query (Conditional)

**Condition:** `enhanced_query["hyde"] != enhanced_query["original"]`

**Query text:** `"The hosts would likely discuss how tariffs impact both domestic industries..."`

**Parameters:**

- `min_docs=2` (reduced from 3)
- `max_docs=5` (reduced from 10)
- `similarity_threshold=0.6` (same)

**Rationale:** HyDE is supplementary, so retrieve fewer docs

**Process:** Same as Search 1, with deduplication

---

#### Search 3: Expanded Query (Conditional)

**Condition:** Expanded query is >1.5x longer than original

```python
if len(enhanced_query["expanded"].split()) > len(enhanced_query["original"].split()) * 1.5:
```

**For our example:**

- Original: 8 words
- Expanded: ~30 words
- 30 > (8 × 1.5 = 12) ✓ **EXECUTES**

**Query text:** `"What do the hosts think about tariffs Ricky Ghoshroy..."`

**Parameters:**

- `min_docs=2`
- `max_docs=5`
- `similarity_threshold=0.6`

**Process:** Same as Search 1, with deduplication

---

#### Result Aggregation

**After 3 searches:**

1. **Deduplicated results** in `all_results` (could be 3-20 segments)
2. **Sort by similarity** descending:
   ```python
   all_results.sort(key=lambda x: x["similarity"], reverse=True)
   ```
3. **Limit to DEFAULT_MAX_DOCS (10):**
   ```python
   return all_results[:self.DEFAULT_MAX_DOCS]
   ```

**Output:** List of up to 10 highest-similarity unique segments across all 3 searches

---

### Step 5: Vector Search (Per Query)

**File:** `vector_search.py:39-104`

**Function:** `find_relevant_above_adaptive_threshold(query, min_docs, max_docs, similarity_threshold)`

**Input:** One of the 3 query variations + parameters

**Process:**

#### 5.1 Generate Embedding

**Call:** `embedding_service.generate_embedding(query)`

**File:** `embedding_service.py:17-22`

**Process:**

1. Call OpenAI API:
   ```python
   response = client.embeddings.create(
       model="text-embedding-3-small",
       input=text
   )
   ```
2. Extract embedding vector from response
3. Return list of floats (dimension: 1536 for text-embedding-3-small)

**Latency:** ~50-100ms per API call

---

#### 5.2 Adaptive Thresholding (Two-Query Strategy)

**Query 1: Threshold-Based Retrieval**

**SQL:**

```sql
SELECT
    s.name,                                  -- Speaker name
    ts.text,                                 -- Segment text
    e.episode_number,                        -- Episode number
    e.title,                                 -- Episode title
    e.date_published,                        -- Publication date
    1 - (ts.embedding <=> %s::vector) as similarity  -- Cosine similarity
FROM transcript_segments ts
JOIN episodes e ON ts.episode_id = e.episode_number
JOIN speakers s ON ts.speaker_id = s.id
WHERE 1 - (ts.embedding <=> %s::vector) >= %s  -- Threshold filter
ORDER BY similarity DESC
LIMIT %s  -- max_docs
```

**Parameters:** `(embedding, embedding, similarity_threshold, max_docs)`

**Example:** `(vector_array, vector_array, 0.6, 10)`

**Vector operator:** `<=>` is pgvector's cosine distance operator

- Returns distance in range [0, 2]
- `1 - distance` converts to similarity in range [0, 1]
- Smaller distance = more similar

**Decision logic:**

```python
if len(results) >= min_docs:
    return results  # Success - have enough quality results
else:
    # Fall through to Query 2
```

---

**Query 2: Fallback Top-N Retrieval**

**Executes if:** Query 1 returned < `min_docs` results

**SQL:**

```sql
SELECT
    s.name,
    ts.text,
    e.episode_number,
    e.title,
    e.date_published,
    1 - (ts.embedding <=> %s::vector) as similarity
FROM transcript_segments ts
JOIN episodes e ON ts.episode_id = e.episode_number
JOIN speakers s ON ts.speaker_id = s.id
ORDER BY similarity DESC
LIMIT %s  -- min_docs (no WHERE threshold!)
```

**Parameters:** `(embedding, min_docs)`

**Example:** `(vector_array, 3)`

**Purpose:** Ensure minimum context even if nothing exceeds quality threshold

**Result:** Always returns exactly `min_docs` results (or fewer if database has fewer segments)

---

**Output from each vector search:**

```python
[
    {
        "name": "Brendan Kelly",
        "text": "I think tariffs are a blunt instrument that often backfire...",
        "episode_number": 145,
        "title": "Trade Wars and Consequences",
        "date_published": "2024-03-15",
        "similarity": 0.847
    },
    {
        "name": "Ricky Ghoshroy",
        "text": "The historical evidence on tariffs shows mixed results...",
        "episode_number": 145,
        "title": "Trade Wars and Consequences",
        "date_published": "2024-03-15",
        "similarity": 0.823
    },
    ...
]
```

---

### Step 6: Episode Grouping

**File:** `rag_service.py:123-153`

**Function:** `group_by_episode(search_results)`

**Input:** Flat list of 10 segments (possibly from multiple episodes)

**Process:**

1. **Initialize dictionary** keyed by episode_number
2. **Iterate through results:**

   ```python
   for result in search_results:
       episode_key = result["episode_number"]

       if episode_key not in episode_groups:
           # Create new episode group
           episode_groups[episode_key] = {
               "episode": result["episode_number"],
               "title": result["title"],
               "date_published": result["date_published"],
               "segments": [],
               "similarities": []
           }

       # Add segment to its episode group
       episode_groups[episode_key]["segments"].append(result)
       episode_groups[episode_key]["similarities"].append(result["similarity"])
   ```

3. **Calculate average similarity per episode:**

   ```python
   for group in episode_groups.values():
       group["avg_similarity"] = sum(group["similarities"]) / len(group["similarities"])
       group["segments"].sort(key=lambda x: x["similarity"], reverse=True)
   ```

4. **Sort episodes by average similarity:**
   ```python
   grouped_results = list(episode_groups.values())
   grouped_results.sort(key=lambda x: x["avg_similarity"], reverse=True)
   ```

**Output:**

```python
[
    {
        "episode": 145,
        "title": "Trade Wars and Consequences",
        "date_published": "2024-03-15",
        "avg_similarity": 0.835,
        "segments": [
            {"name": "Brendan Kelly", "text": "...", "similarity": 0.847},
            {"name": "Ricky Ghoshroy", "text": "...", "similarity": 0.823},
            ...
        ],
        "similarities": [0.847, 0.823, ...]
    },
    {
        "episode": 132,
        "title": "Economic Policy Debates",
        "date_published": "2024-01-20",
        "avg_similarity": 0.712,
        "segments": [...]
    },
    ...
]
```

**Benefit:** Hierarchical organization makes it easier for LLM to understand episode context

---

### Step 7: Context Formatting

**File:** `rag_service.py:155-173`

**Function:** `_format_search_results(search_results)`

**Input:** Episode-grouped results from Step 6

**Process:**

1. **Call** `group_by_episode(search_results)` (if not already grouped)
2. **Format as hierarchical markdown:**

```python
formatted_result = ""
for episode_group in grouped_results:
    # Episode header
    formatted_result += f"## Episode {episode_group['episode']}: {episode_group['title']}\n"
    formatted_result += f"**Relevance**: {episode_group['avg_similarity']:.2f} (Published: {episode_group['date_published']})\n\n"

    # Segments within episode
    for result in episode_group["segments"]:
        formatted_result += f"**{result['name']}**: {result['text']}\n"
        formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"
```

**Example Output:**

```markdown
## Episode 145: Trade Wars and Consequences

**Relevance**: 0.84 (Published: 2024-03-15)

**Brendan Kelly**: I think tariffs are a blunt instrument that often backfire on the very industries they're meant to protect. We saw this with the steel tariffs...
_Similarity: 0.85_

**Ricky Ghoshroy**: The historical evidence on tariffs shows mixed results. While they can provide short-term protection, the long-term economic costs usually outweigh the benefits...
_Similarity: 0.82_

## Episode 132: Economic Policy Debates

**Relevance**: 0.71 (Published: 2024-01-20)

**Brendan Kelly**: When we talk about protectionism, we need to distinguish between strategic trade policy and emotional nationalism...
_Similarity: 0.71_
```

**Format characteristics:**

- Clear episode boundaries
- Speaker attribution for each segment
- Similarity scores for transparency
- Chronological metadata (publish date)
- Hierarchical structure (episode → segments)

---

### Step 8: Prompt Creation

**File:** `rag_service.py:175-196`

**Function:** `_create_prompt(formatted_results, question)`

**Input:**

- `formatted_results`: Markdown context from Step 7
- `question`: Original user question

**Process:**

Creates structured prompt with:

1. **Role definition:** "expert analyst of A Gentleman's Disagreement Podcast"
2. **Task instruction:** Answer based on transcript segments
3. **Formatting instructions:** Use Markdown, headings, bold, quotes
4. **Context provision:** The formatted transcript segments
5. **Question statement:** User's original question
6. **Response trigger:** "Please provide a comprehensive answer..."

**Full template:**

```python
f"""# A Gentleman's Disagreement Podcast Analysis

You are an expert analyst of **A Gentleman's Disagreement Podcast**. Your task is to provide insightful answers based on the provided transcript segments.

## Instructions
- Use the relevant transcript segments below to answer the user's question
- If the segments aren't relevant to the question, clearly state this
- Maintain the conversational tone of the podcast in your analysis
- **Format your response in clean, well-structured Markdown**
- Use proper headings (## or ###), and paragraph breaks for readability
- Bold important points and use quotes for direct transcript references

## Available Transcript Segments
{formatted_results}

## User Question
**{question}**

## Your Response
Please provide a comprehensive answer in Markdown format based on the transcript segments and your knowledge of the podcast:"""
```

**Prompt engineering choices:**

- Explicit role creates consistent persona
- Clear formatting instructions ensure readable output
- Hierarchical structure guides LLM attention
- Escape hatch for irrelevant segments (quality control)
- Markdown output matches client rendering capabilities

---

### Step 9: LLM Streaming

**File:** `rag_service.py:54-69`

**Function:** `_generate_simple_text_stream(prompt, model)`

**Input:**

- `prompt`: Complete prompt from Step 8
- `model`: `"gpt-4o-mini-2024-07-18"` (default)

**Process:**

1. **Create OpenAI chat completion stream:**

   ```python
   stream = self.client.chat.completions.create(
       model=model,
       messages=[{"role": "user", "content": prompt}],
       stream=True
   )
   ```

2. **Yield text chunks as they arrive:**

   ```python
   for chunk in stream:
       if chunk.choices[0].delta.content:
           yield chunk.choices[0].delta.content
   ```

3. **Error handling:**
   ```python
   except Exception as e:
       print(f"Error in simple text streaming: {e}")
       yield f"Error: {str(e)}"
   ```

**Characteristics:**

- **Pure text streaming:** No JSON wrapping, no SSE formatting
- **AI SDK compatible:** Works with TextStreamChatTransport
- **Low overhead:** Minimal processing, just pass through chunks
- **Simple error handling:** Yield error message in stream

**Example stream chunks:**

```
"The"
" hosts"
" have"
" nuanced"
" views"
" on"
" tariffs"
...
```

**Latency:** First token typically arrives in 200-500ms after request

---

### Step 10: Response Delivery

**File:** `main.py:66-73`

**Return to API endpoint**

**Process:**

FastAPI `StreamingResponse` wraps the generator:

```python
StreamingResponse(
    content=rag_service.ask_question_text_stream(user_text),
    media_type="text/plain",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    },
)
```

**HTTP Response:**

```
HTTP/1.1 200 OK
Content-Type: text/plain
Cache-Control: no-cache
Connection: keep-alive
Transfer-Encoding: chunked

The hosts have nuanced views on tariffs...
```

**Client receives:** Plain text stream rendered in real-time

---

## Performance Characteristics

### Latency Breakdown (Typical Query)

| Step                     | Operation                  | Approx Time |
| ------------------------ | -------------------------- | ----------- |
| 1                        | API request handling       | 1-5ms       |
| 2                        | RAG service initialization | 1-5ms       |
| 3                        | Query enhancement (HyDE)   | 200-500ms   |
| 4                        | Embedding generation (×3)  | 150-300ms   |
| 5                        | Database queries (×6)      | 30-100ms    |
| 6                        | Deduplication & sorting    | 1-5ms       |
| 7                        | Episode grouping           | 1-5ms       |
| 8                        | Context formatting         | 1-5ms       |
| 9                        | Prompt creation            | <1ms        |
| 10                       | LLM first token            | 200-500ms   |
| **Total to first token** | **~580-1420ms**            |

**Streaming after first token:** Continuous, typically 20-50 tokens/second

---

### API Calls Per Query

With query enhancement enabled (default):

| Service             | Calls  | Purpose                                     |
| ------------------- | ------ | ------------------------------------------- |
| OpenAI (HyDE)       | 1      | Generate hypothetical answer                |
| OpenAI (Embeddings) | 3      | Original, HyDE, expanded queries            |
| PostgreSQL          | 6      | 2 queries per search (threshold + fallback) |
| OpenAI (Chat)       | 1      | Generate final answer (streaming)           |
| **Total**           | **11** | **5 OpenAI + 6 PostgreSQL**                 |

Without query enhancement:

- **4 total calls** (1 embedding + 2 DB + 1 chat)

---

### Database Query Pattern

**For each of 3 vector searches:**

1. **Threshold query:** `WHERE similarity >= 0.6 LIMIT 10`
2. **Conditional fallback:** `LIMIT 3` (no WHERE) if threshold query returns < min_docs

**Optimization opportunities:**

- ❌ No query result caching
- ❌ No embedding caching
- ❌ No connection pooling
- ✓ Efficient pgvector index usage
- ✓ Smart LIMIT to reduce result set size

---

## Key Design Decisions & Rationale

### 1. Multi-Query Search (3 queries)

**Decision:** Search with original, HyDE, and expanded queries

**Rationale:**

- Improves recall by capturing different semantic angles
- Original: User's exact intent
- HyDE: Answer-oriented matching (bridges question-answer gap)
- Expanded: Catches domain-specific terminology

**Cost:** 3x embedding API calls, 6x database queries

---

### 2. Adaptive Thresholding

**Decision:** Two-query fallback strategy (threshold → top-N)

**Rationale:**

- Prefer high-quality results (>= 0.6 similarity)
- Guarantee minimum context (at least 3 segments)
- Avoid "no results" failures

**Tradeoff:** Up to 2 database queries per search (worst case: 6 total)

---

### 3. Set-Based Deduplication

**Decision:** Use `seen_texts` set with exact text matching

**Rationale:**

- Simple and fast (O(1) lookup)
- Prevents identical segments from multi-query overlap
- Guarantees unique context for LLM

**Limitation:** Only catches exact duplicates, not near-duplicates

---

### 4. Episode Grouping

**Decision:** Organize results by episode with avg similarity

**Rationale:**

- Provides episode context for LLM
- Makes citations clearer
- Shows temporal distribution
- Helps LLM understand discussion continuity

**Benefit:** Improves answer quality through better context organization

---

### 5. Intent Classification (Computed but Unused)

**Decision:** Classify query intent but don't adapt retrieval

**Current state:**

- ✓ Intent is classified (9 categories)
- ✓ Intent influences HyDE prompt
- ❌ Intent does NOT change retrieval parameters
- ❌ Intent does NOT route to different retrieval strategies

**Opportunity:** Biggest untapped potential for improvement

---

### 6. Fixed Parameters

**Decision:** Same params for all query types

**Current values:**

- `DEFAULT_THRESHOLD = 0.6`
- `DEFAULT_MIN_DOCS = 3`
- `DEFAULT_MAX_DOCS = 10`

**Limitations:**

- Episode summaries need more segments (20-50)
- Factual queries might need fewer segments (3-5)
- Comparative queries need balanced sampling
- Temporal queries need date filtering

---

### 7. No Metadata Filtering

**Decision:** Pure vector similarity, no SQL WHERE clauses (except threshold)

**Available but unused metadata:**

- `episode_number` - Could filter episode-scoped queries
- `speakers.name` - Could filter speaker-specific queries
- `date_published` - Could filter temporal queries

**Impact:** Misses opportunities for precision on scoped queries

---

### 8. Text-Only Streaming

**Decision:** Stream plain text, no JSON/SSE/metadata

**Rationale:**

- AI SDK TextStreamChatTransport compatibility
- Lower bandwidth
- Simpler client parsing
- Faster time-to-first-token

**Tradeoff:** Can't send sources/confidence alongside stream

---

## Current System Strengths ✓

1. **Multi-query search** - Good recall through query variations
2. **HyDE implementation** - Bridges question-answer vocabulary gap
3. **Domain vocabulary** - Podcast-specific expansion improves matching
4. **Adaptive thresholding** - Balances quality and availability
5. **Episode grouping** - Clear context organization
6. **Graceful degradation** - Fallbacks prevent total failures
7. **Simple streaming** - Fast, compatible text output
8. **Speaker attribution** - Maintains who said what

---

## Current System Limitations ✗

1. **One-size-fits-all retrieval** - Same strategy for all query types
2. **Intent classification unused** - Computed but not leveraged
3. **No metadata filtering** - Doesn't use episode/speaker/date fields
4. **Fixed document limits** - Always 3-10 segments regardless of need
5. **No diversity optimization** - Can return redundant similar segments
6. **No caching** - Repeated queries re-compute everything
7. **No connection pooling** - New DB connection per query
8. **No keyword/hybrid search** - Pure vector, misses exact matches
9. **No conversation context** - Stateless per-request
10. **No reranking** - Single-stage retrieval only

---

## Critical Flow Insights

### What Actually Determines Retrieved Segments?

**Primary factors (in order of impact):**

1. **Vector similarity** (cosine distance between embeddings)
2. **Deduplication** (exact text matching)
3. **Similarity threshold** (0.6 floor)
4. **Max documents limit** (10 ceiling)
5. **Multi-query inclusion** (up to 3 search variations)

**NOT used (but could be):**

- Query intent (classified but ignored)
- Episode metadata (available but not filtered)
- Speaker metadata (available but not filtered)
- Publish date (available but not filtered)
- Diversity measures (not calculated)
- Cross-encoder reranking (not implemented)

---

### Where Does Enhancement Help Most?

**HyDE is most effective for:**

- Complex analytical questions
- Questions with technical vocabulary mismatch
- Abstract concept queries
- "Why/how" questions

**Query expansion is most effective for:**

- Questions using general terms ("hosts" → specific names)
- Queries missing domain vocabulary
- Broad topical questions

**Enhancement is less effective for:**

- Factual lookup ("which episode discussed X")
- Episode-scoped queries ("summarize episode 145")
- Queries already using precise terms
- Speaker-specific questions (expansion adds noise)

---

### Why 3-10 Segments Exactly?

**This is a hard-coded sweet spot for:**

- General topical questions
- Single-perspective answers
- Typical LLM context window utilization
- Reasonable response latency

**But inadequate for:**

- Episode summaries (need 20-50)
- Corpus-wide analysis (need 30-50)
- Simple factual queries (need 1-3)
- Comparative analysis (need balanced sampling)

**⚠️ This fixed range is a major limitation for diverse query types**

---

## Comparison to QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES.md

The existing strategy document identified:

### Correctly Identified Strengths:

- ✓ Multi-query search works well for general topics
- ✓ Query enhancement (HyDE + expansion) improves recall
- ✓ Episode grouping provides good organization
- ✓ Speaker attribution is maintained

### Correctly Identified Gaps:

- ✓ Intent classification computed but unused
- ✓ No metadata filtering (episode, speaker, date)
- ✓ No diversity-aware ranking
- ✓ Fixed document limits too restrictive
- ✓ No hybrid search (keyword + vector)
- ✓ One-size-fits-all approach doesn't adapt

### Confirmed Through Code Analysis:

**The biggest opportunity is indeed:**

1. **Use intent classification** to route to different strategies
2. **Add metadata filtering** for scoped queries
3. **Implement adaptive parameters** based on query type
4. **Add diversity optimization** for broad queries

---

## Next Steps for Improvement

Based on this analysis, the highest-impact improvements would be:

### Priority 1: Metadata Filtering (Quick Win)

- Add episode number extraction
- Add speaker name extraction
- Filter SQL queries with WHERE clauses
- **Estimated effort:** 1-2 weeks
- **Impact:** HIGH - Solves episode-scoped and speaker-specific queries

### Priority 2: Intent-Aware Routing (High Impact)

- Create retrieval strategy router
- Implement different parameter sets per intent
- Add comparative query handling (balanced sampling)
- **Estimated effort:** 2-3 weeks
- **Impact:** HIGH - Improves multiple query types

### Priority 3: Adaptive Parameters (Medium Effort)

- Intent-specific document limits
- Intent-specific similarity thresholds
- Intent-specific ranking strategies
- **Estimated effort:** 1-2 weeks
- **Impact:** MEDIUM-HIGH - Better quality across all types

### Priority 4: Diversity Optimization (Medium Impact)

- Implement MMR algorithm
- Episode-level diversity constraints
- Speaker balance for comparative queries
- **Estimated effort:** 2-3 weeks
- **Impact:** MEDIUM - Reduces redundancy

---

## Conclusion

The current retrieval process is a well-implemented **general-purpose RAG system** with:

✓ **Strong foundation:**

- Multi-query search with HyDE
- Domain-specific enhancement
- Adaptive thresholding
- Clean streaming architecture

✗ **Key limitations:**

- One-size-fits-all strategy
- Unused intent classification
- No metadata filtering
- Fixed parameters

The system is **well-positioned for improvement** because:

1. Infrastructure is solid and extensible
2. Intent classification already exists
3. Metadata is available in database
4. Episode grouping shows hierarchical thinking

The **biggest bang-for-buck improvement** is enabling intent-aware routing with metadata filtering, which would unlock precision for episode-scoped, speaker-specific, temporal, and comparative queries with relatively modest engineering effort.
