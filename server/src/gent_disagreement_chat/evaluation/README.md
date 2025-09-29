# RAG Evaluation System

This directory contains a comprehensive evaluation framework for the RAG (Retrieval-Augmented Generation) system, specifically optimized for podcast content analysis. The system provides both automated metrics and human-validated ground truth evaluation capabilities.

## Overview

The evaluation system consists of four main components:

1. **`evaluation_runner.py`** - Main orchestrator for all evaluation workflows
2. **`automated_metrics.py`** - Automated quality metrics without requiring ground truth
3. **`ground_truth_generator.py`** - Creates human-validated datasets for precise evaluation
4. **`precision_tracker.py`** - Tracks retrieval precision over time with detailed metrics

## Quick Start

### 1. Create a Baseline Evaluation

Start by evaluating your current RAG system to establish a baseline:

```bash
cd server/src/gent_disagreement_chat/evaluation
python evaluation_runner.py baseline --questions 20
```

This will:
- Run 20 test questions through your current RAG system
- Calculate automated metrics
- Store results in SQLite database
- Provide a baseline ID for future comparisons

### 2. Run Automated Analysis

For immediate feedback on specific queries:

```python
from evaluation.automated_metrics import AutomatedMetrics

# Initialize metrics calculator
metrics = AutomatedMetrics()

# Test a specific question
question = "What did the hosts say about constitutional law?"
retrieved_segments = rag_service.vector_search.find_most_similar(question, limit=10)

# Generate comprehensive report
report = metrics.generate_automated_report(question, retrieved_segments)
print(f"Retrieval Quality: {report['evaluation_summary']['retrieval_quality']}")
print(f"Content Diversity: {report['evaluation_summary']['content_diversity']}")
```

### 3. Compare Two RAG Systems

To compare baseline vs improved systems:

```bash
python evaluation_runner.py compare --question "How do the hosts analyze executive power?"
```

## Detailed Usage

### Automated Metrics (`automated_metrics.py`)

The automated metrics system provides immediate feedback without requiring human evaluation. It's optimized for podcast content with specific features:

#### Key Metrics

- **Similarity Analysis**: Mean, min, max, and standard deviation of similarity scores
- **Content Diversity**: Semantic diversity using sentence transformers
- **Episode Diversity**: Spread across different episodes
- **Speaker Analysis**: Host vs guest content distribution
- **Legal Content Scoring**: Identifies constitutional/legal discussions
- **Temporal Coherence**: Time-based clustering analysis
- **Question Coverage**: How well retrieved content answers the question

#### Usage Example

```python
from evaluation.automated_metrics import AutomatedMetrics

metrics = AutomatedMetrics()

# Evaluate retrieval quality
question = "What are the constitutional implications of executive orders?"
segments = rag_service.vector_search.find_most_similar(question, limit=10)

# Get detailed metrics
quality_metrics = metrics.evaluate_retrieval_quality(question, segments)

# Key metrics to examine:
print(f"Average Similarity: {quality_metrics['avg_similarity']:.3f}")
print(f"Content Diversity: {quality_metrics['content_diversity']:.3f}")
print(f"Speaker Diversity: {quality_metrics['speaker_diversity']:.3f}")
print(f"Legal Content Score: {quality_metrics['legal_content_score']:.3f}")
```

#### Podcast-Specific Features

- **Host vs Guest Analysis**: Identifies segments from Ricky Ghoshroy and Brendan Kelly vs guest experts
- **Legal Content Detection**: Recognizes constitutional law, Supreme Court, and legal analysis discussions
- **Expert Authority Scoring**: Weights segments based on guest expertise indicators
- **Long-form Content Optimization**: Adjusted thresholds for longer podcast segments

### Ground Truth Generation (`ground_truth_generator.py`)

For precise evaluation, create human-validated datasets:

#### Generate Ground Truth Dataset

```bash
python evaluation_runner.py ground-truth --questions 50
```

This creates:
- **JSON dataset** with 50 diverse questions
- **HTML evaluation interface** for human review
- **Question categories**: factual, analytical, comparative, opinion, chronological

#### Question Types Generated

1. **Factual**: "What did Professor Jack Bierman say about Supreme Court textualism?"
2. **Analytical**: "How do the hosts analyze the constitutional implications of executive power?"
3. **Comparative**: "What are Ricky and Brendan's different perspectives on federal authority?"
4. **Opinion**: "What are the hosts' personal views on Court politicization?"
5. **Chronological**: "How have the hosts' discussions evolved over time?"

#### Human Validation Workflow

1. Run ground truth generation
2. Open the generated HTML file in a browser
3. Mark segments as relevant/irrelevant for each question
4. Rate question difficulty
5. Export evaluation results as JSON
6. Validate the dataset:

```bash
python evaluation_runner.py validate --dataset ground_truth_dataset_20231201_143022.json --evaluation ground_truth_evaluation.json
```

### Precision Tracking (`precision_tracker.py`)

Track system performance over time with detailed metrics:

#### Create Baseline

```python
from evaluation.precision_tracker import PrecisionTracker

tracker = PrecisionTracker()
baseline_id = tracker.create_baseline_evaluation(rag_service, test_questions=None)
```

#### Evaluate Improvements

```python
# After implementing improvements
improved_id = tracker.evaluate_improved_system(improved_rag_service, baseline_id)
```

#### Calculate Precise Metrics

With validated ground truth:

```python
metrics = tracker.calculate_precision_with_ground_truth(rag_service, "gt_20231201_143022")
print(f"Precision@5: {metrics['precision_at_5']:.3f}")
print(f"Recall@10: {metrics['recall_at_10']:.3f}")
print(f"NDCG@5: {metrics['ndcg_at_5']:.3f}")
```

### Main Evaluation Runner (`evaluation_runner.py`)

The main orchestrator provides a command-line interface for all evaluation tasks:

#### Available Commands

```bash
# Create baseline evaluation
python evaluation_runner.py baseline --questions 20

# Generate ground truth dataset
python evaluation_runner.py ground-truth --questions 50

# Validate ground truth with human evaluation
python evaluation_runner.py validate --dataset dataset.json --evaluation evaluation.json

# Evaluate improved system
python evaluation_runner.py evaluate --baseline-id baseline_20231201_143022

# Compare systems on single question
python evaluation_runner.py compare --question "Your test question here"

# Generate comprehensive report
python evaluation_runner.py report --baseline-id baseline_123 --improvement-id improved_456
```

## Evaluation Workflow

### Standard Evaluation Process

1. **Establish Baseline**
   ```bash
   poetry run python src/gent_disagreement_chat/evaluation evaluation_runner.py baseline --questions 20

   python evaluation_runner.py baseline --questions 20
   ```

2. **Create Ground Truth** (for precise evaluation)
   ```bash
   python evaluation_runner.py ground-truth --questions 50
   ```

3. **Complete Human Evaluation** (using generated HTML interface)

4. **Validate Ground Truth**
   ```bash
   python evaluation_runner.py validate --dataset dataset.json --evaluation evaluation.json
   ```

5. **Implement RAG Improvements** (hybrid search, reranking, etc.)

6. **Evaluate Improvements**
   ```bash
   python evaluation_runner.py evaluate --baseline-id your_baseline_id
   ```

7. **Generate Reports**
   ```bash
   python evaluation_runner.py report --baseline-id baseline_123 --improvement-id improved_456
   ```

### Quick Iteration Workflow

For rapid development cycles:

1. **Use Automated Metrics** for immediate feedback
2. **Compare specific questions** with the compare command
3. **Track key metrics** like similarity, diversity, and coverage
4. **Validate with ground truth** for final evaluation

## Key Metrics Explained

### Precision Metrics
- **Precision@k**: Fraction of top-k results that are relevant
- **Recall@k**: Fraction of relevant documents found in top-k results
- **MRR**: Mean Reciprocal Rank of first relevant result
- **NDCG**: Normalized Discounted Cumulative Gain (position-weighted)

### Automated Metrics
- **Similarity Scores**: Cosine similarity between query and results
- **Content Diversity**: Semantic variety in retrieved segments
- **Speaker Diversity**: Distribution across hosts and guests
- **Temporal Span**: Range of episodes covered
- **Legal Content Score**: Presence of constitutional/legal analysis

### Podcast-Specific Metrics
- **Host Ratio**: Percentage of segments from show hosts
- **Guest Authority**: Expertise indicators in guest segments
- **Episode Diversity**: Spread across different episodes
- **Long-form Score**: Content complexity indicators

## Configuration and Customization

### Database Configuration

The system uses SQLite for storing evaluation metrics. Default location: `evaluation_metrics.db`

To use a custom database:

```python
tracker = PrecisionTracker("custom_evaluation.db")
```

### Custom Test Questions

Override default questions:

```python
custom_questions = [
    "Your custom question 1",
    "Your custom question 2",
    # ...
]

baseline_id = tracker.create_baseline_evaluation(rag_service, custom_questions)
```

### Evaluation Parameters

Customize evaluation parameters:

```python
# Adjust similarity thresholds for your content
metrics = AutomatedMetrics()
# Modify internal thresholds in automated_metrics.py if needed

# Adjust retrieval limits
segments = rag_service.vector_search.find_most_similar(question, limit=15)
```

## Output Files

The evaluation system generates several types of output files:

- **`baseline_metrics_TIMESTAMP.json`**: Detailed baseline metrics
- **`ground_truth_dataset_TIMESTAMP.json`**: Generated question dataset
- **`ground_truth_evaluation_TIMESTAMP.html`**: Human evaluation interface
- **`validated_ground_truth_TIMESTAMP.json`**: Human-validated dataset
- **`precision_report_TIMESTAMP.json`**: Comprehensive evaluation report

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory and have all dependencies installed
2. **Database Errors**: Check file permissions for SQLite database creation
3. **Model Loading**: The sentence transformer model requires internet connection on first use
4. **Memory Issues**: Large datasets may require adjusting batch sizes

### Dependencies

Required packages:
- `numpy`
- `sentence-transformers`
- `scikit-learn`
- `sqlite3` (built-in)

Install with:
```bash
pip install numpy sentence-transformers scikit-learn
```

## Advanced Usage

### Custom Metrics

Add your own evaluation metrics by extending the `AutomatedMetrics` class:

```python
class CustomMetrics(AutomatedMetrics):
    def custom_metric(self, question, segments):
        # Your custom logic here
        return score
```

### Integration with RAG Service

The evaluation system expects your RAG service to have a `vector_search.find_most_similar()` method that returns segments with `similarity`, `text`, `episode_number`, and `speaker` fields.

### Batch Evaluation

For large-scale evaluation:

```python
questions = load_many_questions()
results = []

for question in questions:
    segments = rag_service.vector_search.find_most_similar(question)
    metrics = automated_metrics.evaluate_retrieval_quality(question, segments)
    results.append(metrics)

# Aggregate results
avg_similarity = np.mean([r['avg_similarity'] for r in results])
```

## Next Steps

1. Start with baseline evaluation to understand current performance
2. Use automated metrics for quick iteration
3. Create ground truth dataset for precise measurement
4. Implement improvements based on metric insights
5. Track progress over time with precision tracking

For questions or issues, refer to the individual Python files for detailed implementation notes and comments.