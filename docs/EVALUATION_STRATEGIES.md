# Evaluation Strategies for RAG Retrieval Improvements

**Date:** November 2025
**Purpose:** Practical strategies for evaluating whether retrieval improvements actually improve chat output quality.

## Overview

When making changes to the retrieval functionality, it's critical to measure whether improvements actually help. This document provides practical evaluation strategies ranging from automated metrics to manual testing approaches.

> **⚠️ Current Implementation Status:**
>
> The complex automated evaluation framework (evaluation_runner.py, automated_metrics.py, etc.) has been **removed** in favor of starting with the simplest, most practical approach.
>
> **Current approach:** Use the simple manual evaluation script (`evaluate_simple.py`) described in Section 9.
>
> You can add more sophisticated evaluation approaches later as needed, but start simple and only add complexity when you have a clear need for it.

---

## 1. ~~Use Your Existing Evaluation Framework~~ (REMOVED)

> **⚠️ This section describes a complex evaluation framework that has been removed.**
>
> **Instead, use the simple manual evaluation approach in Section 9.**
>
> The sophisticated evaluation system (`evaluation_runner.py`, `automated_metrics.py`, `precision_tracker.py`, `ground_truth_generator.py`) with automated metrics, ground truth generation, and SQLite database tracking has been removed in favor of starting with the simplest approach.
>
> If you find you need more sophisticated evaluation capabilities after using the simple approach, you can implement specific features from this section as needed. But start simple first.

### Original Approach (For Reference)

This section described an automated evaluation framework with:
- Baseline creation and comparison tracking
- Automated metrics (similarity, diversity, coverage, coherence)
- SQLite database for historical comparison
- Ground truth generation with LLM assistance

**Why it was removed:**
- High complexity (1,900+ lines of code)
- Not actively being used
- Over-engineered for initial evaluation needs
- Better to start simple and add complexity only when needed

---

## 2. Question Type-Specific Test Suite

Based on `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES.md`, create test questions for each category to ensure improvements help the intended question types.

### Test Question Categories

```python
test_questions = {
    "episode_scoped": [
        "Summarize episode 180",
        "What did they discuss in the latest episode?",
        "What happened in episodes 150-160?",
        "Tell me about the episode from January 2024"
    ],
    "speaker_specific": [
        "What does Ricky think about economics?",
        "What are Brendan's views on the Supreme Court?",
        "What did Ricky say about tariffs?",
        "Has Brendan ever discussed Bitcoin?"
    ],
    "comparative": [
        "Do the hosts disagree on tariffs?",
        "Compare their views on Trump vs Biden",
        "How do their stances differ on constitutional interpretation?"
    ],
    "temporal": [
        "What have they discussed in the last 3 months?",
        "How has their view on tariffs evolved over time?",
        "What were their predictions in 2023?",
        "Track their discussion of inflation from 2020 to now"
    ],
    "factual": [
        "Which episode discussed the Chevron case?",
        "When did they last talk about inflation?",
        "Have they ever discussed Bitcoin?",
        "What episode number featured Professor Smith?"
    ],
    "analytical": [
        "Why do the hosts focus on constitutional law?",
        "What underlying philosophy drives their economic analysis?",
        "How do they approach disagreement?"
    ],
    "topical_broad": [
        "What do the hosts think about tariffs?",
        "What are their overall views on constitutional law?",
        "How do they analyze Supreme Court cases?"
    ],
    "summarization": [
        "Summarize the key themes of the podcast",
        "What are the main topics they cover?",
        "What topics do they return to most often?"
    ]
}
```

### Evaluation Approach

For each improvement:
1. **Target question type**: Does it improve the intended question type?
2. **Regression check**: Does it degrade other question types?
3. **Latency impact**: What's the performance overhead?

### Creating Test Questions

- Use actual user queries if available
- Cover edge cases (date parsing, name variations, etc.)
- Include both simple and complex examples
- Test boundary conditions (very recent episodes, very old episodes)

---

## 3. Side-by-Side Comparison (Manual)

Most practical for iterative development and quick feedback.

### Setup

```python
# In your code, add a comparison mode
def compare_retrieval_strategies(question):
    # Strategy A: Current system
    results_a = vector_search.find_relevant_above_adaptive_threshold(
        question,
        min_docs=3,
        max_docs=10,
        similarity_threshold=0.6
    )

    # Strategy B: New approach (e.g., with metadata filtering)
    results_b = new_filtered_search(
        question,
        min_docs=3,
        max_docs=10,
        similarity_threshold=0.6,
        filter_by_episode=True  # New parameter
    )

    return {
        "question": question,
        "strategy_a": {
            "results": results_a,
            "episodes": [r["episode_number"] for r in results_a],
            "speakers": [r["name"] for r in results_a],
            "similarities": [r["similarity"] for r in results_a]
        },
        "strategy_b": {
            "results": results_b,
            "episodes": [r["episode_number"] for r in results_b],
            "speakers": [r["name"] for r in results_b],
            "similarities": [r["similarity"] for r in results_b]
        }
    }
```

### Manual Evaluation Checklist

For each question, compare:
- **Relevance**: Are segments actually relevant to the question?
- **Diversity**: Do results cover different episodes/perspectives?
- **Episode coverage**: For multi-episode questions, are multiple episodes represented?
- **Speaker balance**: For comparative questions, are both speakers represented?
- **Quality**: Which set would produce a better final answer?

### Tools to Help

- Print segment IDs, episode numbers, speakers, similarity scores
- Use JSON diff tools to compare result sets
- Build simple web UI to display both strategies side-by-side
- Create markdown reports with formatted comparisons

### Example Comparison Output

```
Question: "What does Ricky think about tariffs?"

Strategy A (Current):
- 10 segments from episodes: 45, 45, 92, 92, 92, 103, 180, 180, 180, 201
- Speakers: Ricky (6), Brendan (4)  ← Problem: 40% wrong speaker!
- Similarity range: 0.78 - 0.62

Strategy B (With speaker filtering):
- 10 segments from episodes: 45, 92, 103, 145, 180, 180, 201, 215, 215, 230
- Speakers: Ricky (10), Brendan (0)  ← Fixed: 100% correct speaker!
- Similarity range: 0.76 - 0.58
- Episode diversity: 8 unique episodes (vs 5 in Strategy A)

Winner: Strategy B
```

---

## 4. LLM-as-Judge Evaluation

Use GPT-4 to evaluate answer quality at scale - more scalable than pure manual evaluation.

### Implementation

```python
def llm_judge(question, answer_a, answer_b, ground_truth=None):
    """Use GPT-4 to compare two answers"""
    prompt = f"""You are evaluating two answers to the same question about a podcast.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

{f"Ground Truth Reference: {ground_truth}" if ground_truth else ""}

Evaluate both answers on these criteria (rate 1-5 for each):
1. **Accuracy**: Is the information correct and factual?
2. **Completeness**: Does it fully answer the question?
3. **Relevance**: Is it focused on the question asked?
4. **Citation quality**: Are sources/episodes properly cited?

Provide:
- Scores for each criterion (A and B)
- Overall winner (A or B or Tie)
- Brief explanation of why

Format your response as JSON:
{{
    "scores": {{
        "answer_a": {{"accuracy": X, "completeness": X, "relevance": X, "citation_quality": X}},
        "answer_b": {{"accuracy": X, "completeness": X, "relevance": X, "citation_quality": X}}
    }},
    "winner": "A" | "B" | "Tie",
    "explanation": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### Advantages

- Scales better than pure manual evaluation
- Consistent criteria application
- Can process dozens of comparisons quickly
- Provides detailed explanations

### Limitations

- Still has bias (LLM preferences)
- Costs money (OpenAI API calls)
- Can be inconsistent across runs
- May not catch subtle quality differences
- Works best with ground truth for calibration

### Best Practices

- Run multiple evaluations per question pair (3-5 times) to check consistency
- Use temperature=0 for more deterministic results
- Calibrate with manual review of a subset
- Track cost (can add up quickly)

---

## 5. Metrics to Track Per Question Type

Based on `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES.md`, measure specific metrics for each question type.

### Episode-Scoped Questions

```python
def evaluate_episode_scoped(question, results, target_episode):
    """Metrics for episode-scoped questions"""
    return {
        "episode_precision": sum(1 for r in results if r["episode_number"] == target_episode) / len(results),
        "coverage": len(results) / total_segments_in_episode(target_episode),
        "temporal_ordering": is_temporally_ordered(results),
        "only_target_episode": all(r["episode_number"] == target_episode for r in results)
    }
```

**Key metrics:**
- **Episode precision**: % of results from correct episode(s)
- **Coverage**: % of episode segments retrieved (for summary questions)
- **Ordering**: Are results in temporal order when appropriate?
- **Purity**: Are ALL results from target episode?

### Speaker-Specific Questions

```python
def evaluate_speaker_specific(question, results, target_speaker):
    """Metrics for speaker-specific questions"""
    speaker_counts = Counter(r["name"] for r in results)

    return {
        "speaker_precision": speaker_counts[target_speaker] / len(results),
        "speaker_purity": speaker_counts[target_speaker] == len(results),
        "wrong_speaker_count": sum(count for speaker, count in speaker_counts.items() if speaker != target_speaker)
    }
```

**Key metrics:**
- **Speaker precision**: % of results from requested speaker
- **Speaker purity**: Are ALL results from target speaker?
- **Contamination**: How many wrong-speaker segments leaked in?

### Comparative Questions

```python
def evaluate_comparative(question, results, comparison_dimensions):
    """Metrics for comparative questions (e.g., Ricky vs Brendan on topic X)"""
    speaker_a, speaker_b = comparison_dimensions["speakers"]

    counts_a = sum(1 for r in results if r["name"] == speaker_a)
    counts_b = sum(1 for r in results if r["name"] == speaker_b)

    return {
        "balance_ratio": min(counts_a, counts_b) / max(counts_a, counts_b) if max(counts_a, counts_b) > 0 else 0,
        "both_represented": counts_a > 0 and counts_b > 0,
        "balance_score": 1 - abs(counts_a - counts_b) / len(results)
    }
```

**Key metrics:**
- **Balance**: Both sides represented equally (~50/50)?
- **Both present**: Are both comparison dimensions represented?
- **Disagreement detection**: Does it find segments where hosts discuss same topic?

### Temporal Questions

```python
def evaluate_temporal(question, results, date_range):
    """Metrics for temporal/chronological questions"""
    start_date, end_date = date_range

    in_range = [r for r in results if start_date <= r["date_published"] <= end_date]

    return {
        "date_precision": len(in_range) / len(results),
        "temporal_ordering": is_chronologically_ordered(results),
        "time_span_coverage": get_date_span_coverage(results, date_range),
        "all_in_range": len(in_range) == len(results)
    }
```

**Key metrics:**
- **Date accuracy**: % of results in requested date range
- **Temporal ordering**: Are results sorted chronologically?
- **Evolution tracking**: Are multiple time periods represented?
- **Range compliance**: Are ALL results within specified dates?

### Factual Questions

```python
def evaluate_factual(question, results, expected_answer):
    """Metrics for factual lookup questions"""
    return {
        "precision_at_1": check_answer_in_result(results[0], expected_answer) if results else False,
        "precision_at_3": any(check_answer_in_result(r, expected_answer) for r in results[:3]),
        "exact_match_found": any(expected_answer.lower() in r["text"].lower() for r in results),
        "rank_of_answer": next((i for i, r in enumerate(results) if check_answer_in_result(r, expected_answer)), -1)
    }
```

**Key metrics:**
- **Precision@1**: Is answer in top result?
- **Precision@3**: Is answer in top 3 results?
- **Exact match**: Did it find the exact entity/phrase?
- **Rank**: What position is the answer in results?

---

## 6. Regression Testing

Prevent improvements from breaking existing functionality.

### Setup

```python
# regression_tests.py

regression_tests = {
    "general_topical": [
        {
            "question": "What do hosts think about tariffs?",
            "criteria": {
                "min_segments": 5,
                "min_episodes": 3,
                "both_speakers": True,
                "min_avg_similarity": 0.6
            }
        },
        {
            "question": "What are their views on constitutional law?",
            "criteria": {
                "min_segments": 5,
                "keywords_required": ["constitution", "legal", "court"],
                "min_avg_similarity": 0.65
            }
        }
    ],
    "recent_working_well": [
        # Add questions you know currently work well
        # Don't want these to break with new changes
    ]
}

def run_regression_suite(rag_service):
    """Run regression tests to ensure no degradation"""
    results = []

    for category, tests in regression_tests.items():
        for test in tests:
            question = test["question"]
            criteria = test["criteria"]

            # Run retrieval
            segments = rag_service.get_relevant_segments(question)

            # Check criteria
            passed = check_criteria(segments, criteria)

            results.append({
                "category": category,
                "question": question,
                "passed": passed,
                "details": get_failure_details(segments, criteria) if not passed else None
            })

    return results

def check_criteria(segments, criteria):
    """Check if segments meet specified criteria"""
    checks = []

    if "min_segments" in criteria:
        checks.append(len(segments) >= criteria["min_segments"])

    if "min_episodes" in criteria:
        unique_episodes = len(set(s["episode_number"] for s in segments))
        checks.append(unique_episodes >= criteria["min_episodes"])

    if "both_speakers" in criteria and criteria["both_speakers"]:
        speakers = set(s["name"] for s in segments)
        checks.append(len(speakers) >= 2)

    if "min_avg_similarity" in criteria:
        avg_sim = sum(s["similarity"] for s in segments) / len(segments)
        checks.append(avg_sim >= criteria["min_avg_similarity"])

    if "keywords_required" in criteria:
        text = " ".join(s["text"] for s in segments).lower()
        checks.append(all(kw in text for kw in criteria["keywords_required"]))

    return all(checks)
```

### Integration

Run regression suite:
- Before major changes (baseline)
- After each improvement
- In CI/CD pipeline
- Before releases

### Alerts

Set up alerts when regression tests fail:
- Identify which test failed
- Compare to previous baseline
- Investigate root cause before proceeding

---

## 7. Latency Benchmarking

Track performance impact of improvements.

### Implementation

```python
import time
import numpy as np

def benchmark_retrieval(rag_service, question, n_runs=5):
    """Benchmark retrieval latency"""
    timings = {
        "total": [],
        "retrieval": [],
        "formatting": [],
        "llm_first_token": []
    }

    for _ in range(n_runs):
        # Total time
        start_total = time.time()

        # Retrieval time
        start_retrieval = time.time()
        segments = rag_service.get_relevant_segments(question)
        retrieval_time = time.time() - start_retrieval

        # Formatting time
        start_format = time.time()
        formatted = rag_service.format_results(segments)
        format_time = time.time() - start_format

        # First token time (includes LLM call)
        start_llm = time.time()
        stream = rag_service.generate_stream(formatted, question)
        first_chunk = next(stream)
        first_token_time = time.time() - start_llm

        total_time = time.time() - start_total

        timings["total"].append(total_time)
        timings["retrieval"].append(retrieval_time)
        timings["formatting"].append(format_time)
        timings["llm_first_token"].append(first_token_time)

    return {
        metric: {
            "mean": np.mean(times),
            "median": np.median(times),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99),
            "std": np.std(times)
        }
        for metric, times in timings.items()
    }

def compare_latency(baseline_service, improved_service, test_questions):
    """Compare latency between two retrieval strategies"""
    results = []

    for question in test_questions:
        baseline_timing = benchmark_retrieval(baseline_service, question)
        improved_timing = benchmark_retrieval(improved_service, question)

        results.append({
            "question": question,
            "baseline": baseline_timing,
            "improved": improved_timing,
            "delta": {
                metric: improved_timing[metric]["mean"] - baseline_timing[metric]["mean"]
                for metric in baseline_timing.keys()
            }
        })

    return results
```

### Metrics to Track

- **Time to first token (TTFT)**: User-perceived latency
- **Total retrieval time**: Vector search + any filtering
- **Database query time**: SQL execution time
- **LLM generation time**: Streaming response time
- **End-to-end time**: Question → complete answer

### Acceptable Thresholds

Set performance budgets:
- TTFT: < 1 second (ideal), < 2 seconds (acceptable)
- Retrieval: < 500ms (ideal), < 1 second (acceptable)
- Per-improvement overhead: < 100ms (prefer minimal impact)

### Regression Alerts

Flag if:
- Any latency metric increases by >20%
- TTFT exceeds 2 seconds
- Total time exceeds 5 seconds

---

## 8. User Feedback Loop

If this is user-facing, collect real user feedback.

### Implementation

```python
# In your API response
@app.post("/api/chat")
async def chat(request: ChatRequest):
    feedback_id = str(uuid.uuid4())

    # Generate answer
    answer = rag_service.ask_question(request.question)
    segments = rag_service.last_retrieved_segments  # Track what was retrieved

    # Store for feedback linking
    store_feedback_context(feedback_id, {
        "question": request.question,
        "segments": segments,
        "timestamp": datetime.now()
    })

    return {
        "answer": answer,
        "feedback_id": feedback_id,
        "retrieved_segments": segments  # Optional: show user what was retrieved
    }

# Feedback endpoint
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    store_feedback({
        "feedback_id": feedback.feedback_id,
        "rating": feedback.rating,  # "positive" | "negative"
        "comment": feedback.comment,
        "timestamp": datetime.now()
    })

    return {"status": "success"}
```

### Metrics to Track

- **% positive feedback per question type**: Which question types work best?
- **Common complaints in negative feedback**: What's breaking?
- **Questions with consistently low ratings**: Identify problematic queries
- **Before/after comparison**: Did improvement increase satisfaction?

### Analysis

```python
def analyze_feedback_by_question_type(feedback_data):
    """Group feedback by question type and analyze patterns"""

    # Classify questions
    classified = []
    for item in feedback_data:
        question_type = classify_question(item["question"])
        classified.append({
            **item,
            "question_type": question_type
        })

    # Aggregate by type
    by_type = {}
    for qtype in ["episode_scoped", "speaker_specific", "comparative", ...]:
        subset = [f for f in classified if f["question_type"] == qtype]

        if subset:
            by_type[qtype] = {
                "total": len(subset),
                "positive": sum(1 for f in subset if f["rating"] == "positive"),
                "negative": sum(1 for f in subset if f["rating"] == "negative"),
                "satisfaction_rate": sum(1 for f in subset if f["rating"] == "positive") / len(subset),
                "common_complaints": extract_common_themes([f["comment"] for f in subset if f["comment"]])
            }

    return by_type
```

### Continuous Improvement

- Review negative feedback weekly
- Identify patterns in failing questions
- Add failing questions to test suite
- Prioritize improvements based on user pain points

---

## 9. Retrieval Analysis Dashboard

Build simple analysis tools to spot issues in retrieval results.

### Implementation

```python
from collections import Counter
import numpy as np

def analyze_retrieval_results(results):
    """Generate comprehensive analysis of retrieval results"""

    if not results:
        return {"error": "No results to analyze"}

    episodes = [r["episode_number"] for r in results]
    speakers = [r["name"] for r in results]
    similarities = [r["similarity"] for r in results]
    dates = [r["date_published"] for r in results]

    return {
        # Basic counts
        "total_segments": len(results),
        "unique_episodes": len(set(episodes)),
        "unique_speakers": len(set(speakers)),

        # Similarity stats
        "similarity": {
            "mean": np.mean(similarities),
            "median": np.median(similarities),
            "min": min(similarities),
            "max": max(similarities),
            "std": np.std(similarities)
        },

        # Date range
        "date_range": {
            "earliest": min(dates),
            "latest": max(dates),
            "span_days": (max(dates) - min(dates)).days
        },

        # Distribution analysis
        "speaker_distribution": dict(Counter(speakers)),
        "episode_distribution": dict(Counter(episodes)),

        # Diversity metrics
        "episode_diversity": len(set(episodes)) / len(results),  # 0-1, higher = more diverse
        "speaker_diversity": len(set(speakers)) / len(results),

        # Clustering detection
        "top_episode_concentration": max(Counter(episodes).values()) / len(results),  # % in most common episode
        "top_speaker_concentration": max(Counter(speakers).values()) / len(results),

        # Quality flags
        "warnings": generate_warnings(results)
    }

def generate_warnings(results):
    """Flag potential issues in retrieval results"""
    warnings = []

    episodes = [r["episode_number"] for r in results]
    speakers = [r["name"] for r in results]
    similarities = [r["similarity"] for r in results]

    # Check for clustering
    episode_counts = Counter(episodes)
    if max(episode_counts.values()) / len(results) > 0.7:
        warnings.append({
            "type": "episode_clustering",
            "message": f"70%+ results from single episode ({max(episode_counts, key=episode_counts.get)})",
            "severity": "high"
        })

    # Check for speaker imbalance
    speaker_counts = Counter(speakers)
    if len(speaker_counts) > 1:
        min_speaker_pct = min(speaker_counts.values()) / len(results)
        if min_speaker_pct < 0.2:
            warnings.append({
                "type": "speaker_imbalance",
                "message": f"Speaker imbalance: {dict(speaker_counts)}",
                "severity": "medium"
            })

    # Check similarity distribution
    if np.mean(similarities) < 0.5:
        warnings.append({
            "type": "low_similarity",
            "message": f"Low average similarity: {np.mean(similarities):.2f}",
            "severity": "high"
        })

    if len(results) < 3:
        warnings.append({
            "type": "low_result_count",
            "message": f"Only {len(results)} results retrieved",
            "severity": "medium"
        })

    return warnings
```

### Usage

```python
# After each retrieval
question = "What does Ricky think about tariffs?"
results = rag_service.get_relevant_segments(question)

analysis = analyze_retrieval_results(results)
print(json.dumps(analysis, indent=2))

# Output:
{
  "total_segments": 10,
  "unique_episodes": 4,
  "unique_speakers": 2,
  "similarity": {
    "mean": 0.68,
    "median": 0.67,
    "min": 0.58,
    "max": 0.82
  },
  "speaker_distribution": {
    "Ricky Ghoshroy": 6,
    "Brendan Kelly": 4
  },
  "warnings": [
    {
      "type": "speaker_imbalance",
      "message": "Speaker imbalance: {'Ricky Ghoshroy': 6, 'Brendan Kelly': 4}",
      "severity": "medium"
    }
  ]
}
```

### Spotting Issues

Use this analysis to identify:
- **All results from 1-2 episodes?** → Need diversity-aware ranking
- **All from one speaker?** → Need speaker filtering/balancing
- **Very low similarities?** → Threshold too loose, retrieval quality poor
- **Very few results?** → Threshold too strict, missing relevant content
- **High clustering?** → Need episode diversity constraints

---

## Recommended Practical Approach

> **🎯 START HERE: Use the simple evaluation script from Section 9 first!**
>
> The "Quick Win: Simple Evaluation Script" is now the recommended starting point. Don't jump to the complex phases below until you've used the simple approach and identified specific needs for more sophisticated evaluation.

Given where you are now (just removed query enhancement), here's a practical workflow for evaluating improvements.

### Phase 1: Baseline (Do This First) - USE SIMPLE APPROACH

**Goal:** Establish baseline performance before making improvements.

**Steps:**

1. **Run the simple evaluation script**
   ```bash
   cd server/src/gent_disagreement_chat
   python evaluate_simple.py
   ```

2. **Review results interactively**
   - The script will show you retrieval results for 10 test questions
   - For each question, you'll see: episodes retrieved, speakers, similarity scores, and top 3 segments
   - You provide feedback (y/n/notes) on quality
   - Results are saved to `evaluation_results.json`

3. **Analyze baseline patterns**
   - Which question types work well?
   - Which question types fail?
   - Common issues (wrong speaker, wrong episode, low diversity, etc.)
   - Document these observations

4. **Save the baseline results**
   - Keep the `evaluation_results.json` file
   - Rename it to `baseline_results.json` for comparison later
   - This is your benchmark for measuring improvements

**Time investment:** 30-45 minutes
**Deliverable:** `baseline_results.json` with current performance + notes on what needs improvement

### Phase 2: Iterative Improvements

**Goal:** Make targeted improvements and evaluate each one.

For each improvement (e.g., "add episode filtering for episode-scoped questions"):

**Step 1: Unit Test the Improvement** (15-30 min)
```python
# Test question: "Summarize episode 180"
results = improved_rag_service.get_relevant_segments("Summarize episode 180")

# Assertions
assert all(r["episode_number"] == 180 for r in results), "All results should be from episode 180"
assert len(results) >= 10, "Should retrieve sufficient segments for summary"
assert results[0]["position_in_episode"] < results[-1]["position_in_episode"], "Should be temporally ordered"
```

**Step 2: Side-by-Side Comparison** (30-60 min)
```python
# Run 5-10 relevant test questions
test_questions = [
    "Summarize episode 180",
    "What did they discuss in episode 150?",
    "Tell me about the latest episode"
]

for question in test_questions:
    baseline_results = baseline_service.get_relevant_segments(question)
    improved_results = improved_service.get_relevant_segments(question)

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"\nBaseline: {len(baseline_results)} segments from episodes {set(r['episode_number'] for r in baseline_results)}")
    print(f"Improved: {len(improved_results)} segments from episodes {set(r['episode_number'] for r in improved_results)}")

    # Manual review: which is better?
```

**Step 3: Full Test Suite** (1-2 hours)
```bash
# Run all test questions with improved system
python -m evaluation.evaluation_runner evaluate --baseline-id <baseline_id>

# Check:
# 1. Did target question type improve?
# 2. Did other question types stay the same or improve?
# 3. Any regressions?
```

**Step 4: Latency Check** (15-30 min)
```python
benchmark_results = compare_latency(baseline_service, improved_service, test_questions)

# Ensure acceptable overhead (<100ms ideal)
for result in benchmark_results:
    delta = result["delta"]["total"]["mean"]
    assert delta < 0.1, f"Too much overhead: {delta:.3f}s"
```

**Step 5: Decision Point**
- **If improvement is significant + no regressions + acceptable latency** → Keep it, make new baseline
- **If improvement is marginal or has regressions** → Iterate or abandon
- **If latency is unacceptable** → Optimize or consider trade-offs

**Repeat for each improvement.**

### Phase 3: Quantitative Evaluation

**Goal:** Generate comprehensive metrics once you have several improvements.

**Steps:**

1. **Run full evaluation suite**
   ```bash
   python -m evaluation.evaluation_runner evaluate --baseline-id <original_baseline>
   ```

2. **Generate comparison report**
   ```bash
   python -m evaluation.evaluation_runner report \
       --baseline-id <original_baseline> \
       --improvement-id <current_version>
   ```

3. **Analyze results**
   - Overall improvement percentage
   - Per-question-type breakdown
   - Latency impact
   - User satisfaction (if applicable)

4. **Track trends over time**
   - Query evaluation database for historical comparisons
   - Plot improvement trajectory
   - Identify which improvements had biggest impact

**Time investment:** 1-2 hours
**Deliverable:** Comprehensive evaluation report

---

## 🎯 Quick Win: Simple Evaluation Script (RECOMMENDED START)

> **This is the current recommended approach!**
>
> The script described here has been created as `server/src/gent_disagreement_chat/evaluate_simple.py` and is ready to use.

This basic evaluation script provides immediate, practical feedback on retrieval quality:

```python
# evaluate_simple.py
"""Simple evaluation script for quick feedback on retrieval improvements"""

from collections import Counter
from gent_disagreement_chat.core import RAGService

# Test questions covering different types
TEST_QUESTIONS = [
    ("Summarize episode 180", "episode_scoped"),
    ("What did they discuss in the latest episode?", "episode_scoped"),
    ("What does Ricky think about tariffs?", "speaker_specific"),
    ("What are Brendan's views on the Supreme Court?", "speaker_specific"),
    ("Do the hosts disagree on trade?", "comparative"),
    ("Compare their views on Trump vs Biden", "comparative"),
    ("What have they discussed in the last 3 months?", "temporal"),
    ("Which episode discussed the Chevron case?", "factual"),
    ("What do the hosts think about tariffs?", "topical_broad"),
    ("What are their overall views on constitutional law?", "topical_broad"),
]

def evaluate():
    """Run simple evaluation and print results"""
    rag_service = RAGService()

    results = []

    for question, category in TEST_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        # Get retrieval results
        segments = rag_service.vector_search.find_relevant_above_adaptive_threshold(
            question,
            min_docs=3,
            max_docs=10,
            similarity_threshold=0.6
        )

        # Analyze results
        episodes = [s["episode_number"] for s in segments]
        speakers = [s["name"] for s in segments]
        similarities = [s["similarity"] for s in segments]

        print(f"\nRetrieved: {len(segments)} segments")
        print(f"Episodes: {dict(Counter(episodes))}")
        print(f"Unique episodes: {len(set(episodes))}")
        print(f"Speakers: {dict(Counter(speakers))}")
        print(f"Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        print(f"Similarity mean: {sum(similarities)/len(similarities):.3f}")

        # Show top 3 segments
        print(f"\nTop 3 segments:")
        for i, segment in enumerate(segments[:3], 1):
            print(f"\n{i}. Episode {segment['episode_number']} - {segment['name']} (sim: {segment['similarity']:.3f})")
            print(f"   {segment['text'][:150]}...")

        # Manual evaluation
        print(f"\n{'─'*80}")
        print("Is this retrieval good? (y/n/notes): ", end="")
        feedback = input()

        results.append({
            "question": question,
            "category": category,
            "num_segments": len(segments),
            "num_episodes": len(set(episodes)),
            "speaker_balance": dict(Counter(speakers)),
            "avg_similarity": sum(similarities) / len(similarities),
            "feedback": feedback
        })

    # Summary
    print(f"\n\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    for category, items in by_category.items():
        positive = sum(1 for i in items if i["feedback"].lower().startswith('y'))
        total = len(items)

        print(f"\n{category}:")
        print(f"  Success rate: {positive}/{total}")
        print(f"  Avg segments: {sum(i['num_segments'] for i in items) / total:.1f}")
        print(f"  Avg episodes: {sum(i['num_episodes'] for i in items) / total:.1f}")
        print(f"  Avg similarity: {sum(i['avg_similarity'] for i in items) / total:.3f}")

    # Save results
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    evaluate()
```

### Usage

The script is ready to use. Simply run:

```bash
cd server/src/gent_disagreement_chat
python evaluate_simple.py
```

This gives you immediate visibility into:
- What's being retrieved for each question type
- Episode/speaker distributions
- Similarity scores
- Quick manual quality assessment

**Time investment:** 30-45 minutes to run
**Deliverable:** `evaluation_results.json` with feedback on current system performance

### How to Use Results

After running the script:

1. **Review the JSON output** - Compare results over time to see if changes improve retrieval
2. **Identify patterns** - Which question types consistently fail? Which succeed?
3. **Save baselines** - Before making changes, save a copy as `baseline_results.json`
4. **Compare improvements** - Run again after changes and compare the JSON files
5. **Iterate** - Focus improvements on the question types that are failing

---

## Summary

### Recommended Tools by Development Stage

> **🎯 Current Status: Early Development**
>
> **You should use:** Simple evaluation script (`evaluate_simple.py`)
>
> Only move to more complex approaches when you have a clear, demonstrated need.

**Early development (exploring improvements) - START HERE:**
- ✅ **Simple evaluation script** (`evaluate_simple.py`) - **IMPLEMENTED & RECOMMENDED**
- ✅ Side-by-side comparison (manual)
- ✅ Retrieval analysis dashboard

**Mid development (validating improvements) - ADD ONLY IF NEEDED:**
- ⚪ Question type-specific test suite
- ⚪ Regression testing
- ⚪ Latency benchmarking

**Mature development (optimizing at scale) - FUTURE CONSIDERATION:**
- ⚪ Automated evaluation framework
- ⚪ LLM-as-judge evaluation
- ⚪ User feedback loops
- ⚪ A/B testing

### Key Principles

1. **Start simple**: Manual evaluation of 10-20 questions gives quick signal
2. **Automate gradually**: Build automation as patterns emerge
3. **Test each improvement**: Don't batch changes without measuring each
4. **Track regressions**: Ensure new improvements don't break existing functionality
5. **Measure latency**: Speed matters - track performance impact
6. **Use multiple metrics**: No single metric tells the full story
7. **Compare to baseline**: Always have a reference point
8. **Question type matters**: Different questions need different evaluation approaches

### Success Metrics

You're making good progress if:
- ✅ Target question types show measurable improvement
- ✅ No regressions in other question types
- ✅ Latency overhead is acceptable (<100ms per improvement)
- ✅ User satisfaction increases (if measured)
- ✅ Failure modes are reduced or eliminated

---

## Next Steps

> **🚀 Ready to start evaluating? Here's what to do:**

1. **Run the simple evaluation script** to establish baseline
   ```bash
   cd server/src/gent_disagreement_chat
   python evaluate_simple.py
   ```

2. **Save the baseline results**
   ```bash
   cp evaluation_results.json baseline_results.json
   ```

3. **Analyze the results** - Which question types work? Which fail?

4. **Pick first improvement** from `QUESTION_TYPES_AND_RETRIEVAL_STRATEGIES.md` based on what's failing

5. **Make the improvement** in your code

6. **Run evaluation again** and compare
   ```bash
   python evaluate_simple.py
   # Compare evaluation_results.json with baseline_results.json
   ```

7. **Iterate** - Keep what works, discard what doesn't

The simple evaluation script is ready to use. Start there, and only add more sophisticated evaluation approaches when you have a clear need for them.
