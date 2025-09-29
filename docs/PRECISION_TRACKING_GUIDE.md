# RAG Precision Tracking and Measurement Guide

This guide explains how to track current precision and quantify improvements after implementing the RAG optimization recommendations.

## Quick Start

### 1. Create Baseline (5 minutes)
```bash
cd server/src/gent_disagreement_chat
python -m evaluation.evaluation_runner baseline --questions 20
```

### 2. Implement Improvements
Follow the recommendations in `RAG_OPTIMIZATION_RECOMMENDATIONS.md`

### 3. Measure Improvements (5 minutes)
```bash
python -m evaluation.evaluation_runner evaluate --baseline-id <your_baseline_id>
```

### 4. Generate Report (2 minutes)
```bash
python -m evaluation.evaluation_runner report --baseline-id <baseline_id> --improvement-id <improvement_id>
```

## Evaluation Framework Overview

The evaluation system provides three levels of precision tracking:

### **Level 1: Automated Metrics (No Ground Truth Required)**
- **Similarity-based metrics**: Average cosine similarity, min/max scores
- **Diversity metrics**: Content diversity, episode diversity, speaker balance
- **Coverage metrics**: Question-answer overlap, semantic relevance
- **Coherence metrics**: Contextual and temporal consistency

### **Level 2: Comparative Analysis**
- **Before/after comparison**: Automated comparison between baseline and improved systems
- **A/B testing**: Side-by-side evaluation with statistical significance
- **Regression detection**: Identify any performance decreases

### **Level 3: Ground Truth Validation (Highest Accuracy)**
- **Human-validated datasets**: Precisely labeled relevant/irrelevant segments
- **Standard IR metrics**: Precision@K, Recall@K, NDCG, MRR
- **Answer quality assessment**: Relevance, completeness, accuracy

## Detailed Usage Instructions

### Creating Baseline Evaluation

```bash
# Create baseline with default questions
python -m evaluation.evaluation_runner baseline

# Create baseline with custom question count
python -m evaluation.evaluation_runner baseline --questions 50

# Output example:
# 🎯 Baseline created: baseline_20250122_143022
# Average Precision@5: 0.654
```

**What this measures:**
- Current retrieval similarity scores
- Diversity of retrieved content
- Coverage of question topics
- Speaker and episode distribution

### Implementing Improvements

Before measuring improvements, implement the recommendations:

1. **Quick wins** (implement first):
   - Dynamic threshold filtering (`rag_service.py:29`)
   - Enhanced context formatting (`rag_service.py:170`)
   - Basic query preprocessing

2. **Core improvements**:
   - Hybrid search with BM25
   - Two-stage reranking
   - Contextual embeddings

3. **Advanced features**:
   - Semantic chunking
   - Ensemble embeddings
   - Fine-tuned models

### Measuring Improvements

```bash
# Evaluate improved system against baseline
python -m evaluation.evaluation_runner evaluate --baseline-id baseline_20250122_143022

# Output example:
# 🚀 Improvement evaluation: improved_20250122_150322
# === RAG Improvement Report ===
# Precision@5 Improvement: +28.5%
# Relevance Score Improvement: +15.2%
```

### Single Question Comparison

```bash
# Quick comparison for a single question
python -m evaluation.evaluation_runner compare --question "What are the hosts' views on technology?"

# Output example:
# 📈 IMPROVEMENTS:
#   • avg_similarity: +15.3%
#   • content_diversity: +22.1%
#   • question_coverage: +18.7%
# 🎯 OVERALL IMPROVEMENT: 18.7%
```

## Creating Ground Truth Datasets (For Precise Measurement)

### Step 1: Generate Dataset
```bash
python -m evaluation.evaluation_runner ground-truth --questions 100
```

This creates:
- `ground_truth_dataset_YYYYMMDD_HHMMSS.json`
- `ground_truth_evaluation_YYYYMMDD_HHMMSS.html`

### Step 2: Human Evaluation
1. Open the generated HTML file in a browser
2. For each question, mark retrieved segments as "Relevant" or "Irrelevant"
3. Rate question difficulty (Easy/Medium/Hard)
4. Export results as JSON

### Step 3: Validate Dataset
```bash
python -m evaluation.evaluation_runner validate \
  --dataset ground_truth_dataset_20250122.json \
  --evaluation ground_truth_evaluation.json
```

### Step 4: Precise Measurement
```bash
python -m evaluation.evaluation_runner report \
  --baseline-id baseline_20250122_143022 \
  --improvement-id improved_20250122_150322 \
  --dataset validated_ground_truth_20250122.json
```

## Understanding the Metrics

### Automated Metrics (Level 1)

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `avg_similarity` | 0-1 | Average cosine similarity of retrieved segments |
| `content_diversity` | 0-1 | How diverse the retrieved content is (1 = very diverse) |
| `question_coverage` | 0-1 | How well segments cover the question (1 = complete coverage) |
| `speaker_diversity` | 0-1 | Distribution across different speakers |
| `temporal_span` | 0-N | Range of episodes covered |
| `semantic_relevance_mean` | 0-1 | Average semantic similarity to question |

### Standard IR Metrics (Level 3)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision@K** | `relevant_retrieved@K / K` | Fraction of retrieved documents that are relevant |
| **Recall@K** | `relevant_retrieved@K / total_relevant` | Fraction of relevant documents that are retrieved |
| **MRR** | `1 / rank_of_first_relevant` | Mean Reciprocal Rank |
| **NDCG@K** | Normalized DCG | Ranking quality considering position |

### Expected Baseline Scores

Based on typical RAG systems:

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Precision@5 | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |
| Avg Similarity | <0.5 | 0.5-0.7 | 0.7-0.85 | >0.85 |
| Content Diversity | <0.3 | 0.3-0.5 | 0.5-0.7 | >0.7 |
| Question Coverage | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |

## Tracking Improvements Over Time

### Continuous Monitoring
```bash
# Create weekly baselines
python -m evaluation.evaluation_runner baseline --questions 20
# Store baseline_id in tracking sheet

# Monthly detailed evaluation
python -m evaluation.evaluation_runner baseline --questions 50
python -m evaluation.evaluation_runner ground-truth --questions 30
```

### Performance Dashboard
The system generates JSON reports that can be imported into dashboards:

```json
{
  "timestamp": "2025-01-22T15:30:22",
  "metrics": {
    "precision_at_5": 0.68,
    "recall_at_10": 0.75,
    "avg_similarity": 0.72,
    "improvement_vs_baseline": "+15.3%"
  }
}
```

## Expected Improvement Ranges

Based on the optimization recommendations:

### **Phase 1 Improvements (Quick Wins)**
- **Precision@5**: +10-20%
- **Relevance Score**: +5-15%
- **Content Diversity**: +15-25%

### **Phase 2 Improvements (Core Optimizations)**
- **Precision@5**: +25-40% (cumulative)
- **Recall@10**: +20-35%
- **Question Coverage**: +30-50%

### **Phase 3 Improvements (Advanced Features)**
- **Precision@5**: +35-50% (cumulative)
- **NDCG@10**: +25-40%
- **Overall User Satisfaction**: +20-35%

## Common Issues and Troubleshooting

### Low Baseline Scores
- **avg_similarity < 0.5**: Check embedding model and database setup
- **content_diversity < 0.3**: Too many similar segments, improve chunking
- **question_coverage < 0.4**: Questions may be too specific or outside domain

### Inconsistent Improvements
- **High variance in metrics**: Use more test questions (50-100)
- **Negative improvements**: Check for regressions in core functionality
- **Plateau effects**: May need advanced techniques (reranking, fine-tuning)

### Implementation Validation
```bash
# Quick sanity check
python -m evaluation.evaluation_runner compare \
  --question "What do the hosts think about AI?"

# Should show meaningful differences if improvements are working
```

## Integration with Development Workflow

### Pre-Deployment Checklist
```bash
# 1. Run baseline
baseline_id=$(python -m evaluation.evaluation_runner baseline --questions 20)

# 2. Implement changes
# ... your improvements ...

# 3. Validate improvements
python -m evaluation.evaluation_runner evaluate --baseline-id $baseline_id

# 4. Generate report
python -m evaluation.evaluation_runner report --baseline-id $baseline_id --improvement-id $improvement_id
```

### Regression Testing
```bash
# Monthly regression check
python -m evaluation.evaluation_runner compare \
  --question "Standard test question"
# Compare with historical results
```

## File Locations

```
server/src/gent_disagreement_chat/
├── evaluation/
│   ├── __init__.py
│   ├── precision_tracker.py      # Main tracking logic
│   ├── ground_truth_generator.py # Dataset creation
│   ├── automated_metrics.py      # Automated evaluation
│   └── evaluation_runner.py      # CLI interface
├── evaluation_metrics.db         # SQLite database (auto-created)
├── ground_truth_*.json          # Generated datasets
├── ground_truth_*.html          # Human evaluation interfaces
└── precision_report_*.json      # Generated reports
```

## Next Steps

1. **Start with baseline**: Create your current system baseline
2. **Implement quick wins**: Focus on dynamic thresholds and context formatting
3. **Measure incrementally**: Test each improvement separately
4. **Create ground truth**: Develop precise measurement capability
5. **Monitor continuously**: Track performance over time

This framework gives you quantitative evidence of RAG improvements and helps prioritize optimization efforts based on measured impact.