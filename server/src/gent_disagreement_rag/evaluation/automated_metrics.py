import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime


class AutomatedMetrics:
    """
    Implements automated metrics for RAG evaluation without requiring ground truth.
    Uses various heuristics and embedding-based approaches.
    """

    def __init__(self):
        # Load semantic similarity model for automated evaluation
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Could not load semantic model: {e}")
            self.semantic_model = None

    def evaluate_retrieval_quality(self, question: str, retrieved_segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate retrieval quality using automated metrics

        Args:
            question: The input question
            retrieved_segments: List of retrieved segments with metadata

        Returns:
            metrics: Dictionary of automated quality metrics
        """
        metrics = {}

        # Basic similarity metrics
        similarities = [seg.get('similarity', 0) for seg in retrieved_segments]
        if similarities:
            metrics['avg_similarity'] = np.mean(similarities)
            metrics['min_similarity'] = np.min(similarities)
            metrics['max_similarity'] = np.max(similarities)
            metrics['similarity_std'] = np.std(similarities)
            metrics['similarity_range'] = metrics['max_similarity'] - metrics['min_similarity']

        # Diversity metrics
        metrics.update(self._calculate_diversity_metrics(retrieved_segments))

        # Coherence metrics
        metrics.update(self._calculate_coherence_metrics(retrieved_segments))

        # Coverage metrics
        metrics.update(self._calculate_coverage_metrics(question, retrieved_segments))

        # Semantic relevance (if model available)
        if self.semantic_model:
            metrics.update(self._calculate_semantic_relevance(question, retrieved_segments))

        # Temporal distribution
        metrics.update(self._calculate_temporal_metrics(retrieved_segments))

        # Speaker diversity
        metrics.update(self._calculate_speaker_metrics(retrieved_segments))

        return metrics

    def _calculate_diversity_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate diversity metrics for retrieved segments"""
        if len(segments) < 2:
            return {'content_diversity': 0.0, 'episode_diversity': 0.0}

        # Content diversity using text similarity
        texts = [seg.get('text', '') for seg in segments]
        if self.semantic_model:
            embeddings = self.semantic_model.encode(texts)
            similarity_matrix = cosine_similarity(embeddings)

            # Calculate average pairwise similarity (lower = more diverse)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            avg_similarity = np.mean(upper_triangle)
            content_diversity = 1.0 - avg_similarity
        else:
            # Fallback: text overlap diversity
            content_diversity = self._calculate_text_overlap_diversity(texts)

        # Episode diversity
        episodes = [seg.get('episode_number', 0) for seg in segments]
        unique_episodes = len(set(episodes))
        episode_diversity = unique_episodes / len(segments) if segments else 0.0

        return {
            'content_diversity': content_diversity,
            'episode_diversity': episode_diversity,
            'unique_episodes': unique_episodes
        }

    def _calculate_coherence_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate coherence metrics for retrieved segments"""
        if len(segments) < 2:
            return {'contextual_coherence': 1.0, 'temporal_coherence': 1.0}

        # Contextual coherence: how well segments connect thematically
        if self.semantic_model:
            texts = [seg.get('text', '') for seg in segments]
            embeddings = self.semantic_model.encode(texts)

            # Calculate sequential coherence (adjacent segments)
            sequential_similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                sequential_similarities.append(sim)

            contextual_coherence = np.mean(sequential_similarities) if sequential_similarities else 0.0
        else:
            contextual_coherence = 0.5  # neutral score when model unavailable

        # Temporal coherence: segments from similar time periods
        timestamps = []
        for seg in segments:
            ep_num = seg.get('episode_number', 0)
            if ep_num > 0:
                timestamps.append(ep_num)

        if len(timestamps) > 1:
            timestamp_std = np.std(timestamps)
            max_possible_std = np.std(range(1, max(timestamps) + 1)) if max(timestamps) > 1 else 1
            temporal_coherence = 1.0 - (timestamp_std / max_possible_std)
        else:
            temporal_coherence = 1.0

        return {
            'contextual_coherence': contextual_coherence,
            'temporal_coherence': temporal_coherence
        }

    def _calculate_coverage_metrics(self, question: str, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate how well segments cover the question"""
        if not segments:
            return {'question_coverage': 0.0, 'keyword_overlap': 0.0}

        # Simple keyword overlap
        question_words = set(question.lower().split())
        segment_texts = ' '.join([seg.get('text', '') for seg in segments]).lower()
        segment_words = set(segment_texts.split())

        keyword_overlap = len(question_words & segment_words) / len(question_words) if question_words else 0.0

        # Semantic coverage (if model available)
        if self.semantic_model:
            question_embedding = self.semantic_model.encode([question])
            combined_text = ' '.join([seg.get('text', '') for seg in segments])
            combined_embedding = self.semantic_model.encode([combined_text])

            question_coverage = cosine_similarity(question_embedding, combined_embedding)[0][0]
        else:
            question_coverage = keyword_overlap

        return {
            'question_coverage': question_coverage,
            'keyword_overlap': keyword_overlap
        }

    def _calculate_semantic_relevance(self, question: str, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate semantic relevance using sentence transformers"""
        if not self.semantic_model or not segments:
            return {'semantic_relevance_mean': 0.0, 'semantic_relevance_max': 0.0}

        question_embedding = self.semantic_model.encode([question])
        segment_texts = [seg.get('text', '') for seg in segments]
        segment_embeddings = self.semantic_model.encode(segment_texts)

        # Calculate similarities
        similarities = cosine_similarity(question_embedding, segment_embeddings)[0]

        return {
            'semantic_relevance_mean': np.mean(similarities),
            'semantic_relevance_max': np.max(similarities),
            'semantic_relevance_min': np.min(similarities),
            'semantic_relevance_std': np.std(similarities)
        }

    def _calculate_temporal_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate temporal distribution metrics"""
        episodes = [seg.get('episode_number', 0) for seg in segments if seg.get('episode_number', 0) > 0]

        if not episodes:
            return {'temporal_span': 0.0, 'recency_bias': 0.0}

        temporal_span = max(episodes) - min(episodes) if len(episodes) > 1 else 0
        max_episode = max(episodes)

        # Recency bias: preference for newer episodes
        recency_scores = [ep / max_episode for ep in episodes]
        recency_bias = np.mean(recency_scores)

        return {
            'temporal_span': temporal_span,
            'recency_bias': recency_bias,
            'episode_range': f"{min(episodes)}-{max(episodes)}"
        }

    def _calculate_speaker_metrics(self, segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate speaker diversity metrics"""
        speakers = [seg.get('speaker', 'unknown') for seg in segments]
        unique_speakers = len(set(speakers))
        speaker_diversity = unique_speakers / len(segments) if segments else 0.0

        # Calculate speaker balance (how evenly distributed)
        from collections import Counter
        speaker_counts = Counter(speakers)
        if len(speaker_counts) > 1:
            count_values = list(speaker_counts.values())
            speaker_balance = 1.0 - (np.std(count_values) / np.mean(count_values))
        else:
            speaker_balance = 1.0

        return {
            'speaker_diversity': speaker_diversity,
            'unique_speakers': unique_speakers,
            'speaker_balance': speaker_balance
        }

    def _calculate_text_overlap_diversity(self, texts: List[str]) -> float:
        """Calculate diversity using text overlap (fallback method)"""
        if len(texts) < 2:
            return 0.0

        # Calculate average Jaccard similarity between text pairs
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())

                if not words_i and not words_j:
                    jaccard = 1.0
                elif not words_i or not words_j:
                    jaccard = 0.0
                else:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    jaccard = intersection / union

                similarities.append(jaccard)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity

    def calculate_answer_quality_metrics(self, question: str, answer: str,
                                       retrieved_segments: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics for answer quality based on retrieved context

        Args:
            question: Original question
            answer: Generated answer
            retrieved_segments: Segments used for generation

        Returns:
            metrics: Answer quality metrics
        """
        metrics = {}

        # Answer-question relevance
        if self.semantic_model:
            q_embedding = self.semantic_model.encode([question])
            a_embedding = self.semantic_model.encode([answer])
            answer_relevance = cosine_similarity(q_embedding, a_embedding)[0][0]
            metrics['answer_question_relevance'] = answer_relevance

        # Context utilization
        segment_texts = [seg.get('text', '') for seg in retrieved_segments]
        if segment_texts and self.semantic_model:
            combined_context = ' '.join(segment_texts)
            context_embedding = self.semantic_model.encode([combined_context])
            a_embedding = self.semantic_model.encode([answer])
            context_utilization = cosine_similarity(context_embedding, a_embedding)[0][0]
            metrics['context_utilization'] = context_utilization

        # Answer completeness (length-based heuristic)
        answer_length = len(answer.split())
        context_length = sum(len(text.split()) for text in segment_texts)

        if context_length > 0:
            length_ratio = answer_length / context_length
            # Normalize to 0-1 range (assuming optimal ratio is around 0.3)
            completeness_score = min(1.0, length_ratio / 0.3)
            metrics['answer_completeness'] = completeness_score

        # Specificity (presence of concrete details)
        specificity_indicators = ['number', 'date', 'name', 'specific', 'exactly', 'precisely']
        specificity_count = sum(1 for indicator in specificity_indicators
                              if indicator in answer.lower())
        metrics['answer_specificity'] = min(1.0, specificity_count / 3)

        return metrics

    def generate_automated_report(self, question: str, retrieved_segments: List[Dict[str, Any]],
                                answer: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive automated evaluation report

        Args:
            question: Input question
            retrieved_segments: Retrieved segments
            answer: Generated answer (optional)

        Returns:
            report: Comprehensive evaluation report
        """
        retrieval_metrics = self.evaluate_retrieval_quality(question, retrieved_segments)

        report = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'retrieval_metrics': retrieval_metrics,
            'segments_count': len(retrieved_segments),
            'evaluation_summary': self._generate_summary(retrieval_metrics)
        }

        if answer:
            answer_metrics = self.calculate_answer_quality_metrics(
                question, answer, retrieved_segments
            )
            report['answer_metrics'] = answer_metrics
            report['evaluation_summary']['answer_quality'] = self._summarize_answer_quality(answer_metrics)

        return report

    def _generate_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable summary of metrics"""
        summary = {}

        # Retrieval quality
        avg_sim = metrics.get('avg_similarity', 0)
        if avg_sim >= 0.8:
            summary['retrieval_quality'] = 'Excellent'
        elif avg_sim >= 0.6:
            summary['retrieval_quality'] = 'Good'
        elif avg_sim >= 0.4:
            summary['retrieval_quality'] = 'Fair'
        else:
            summary['retrieval_quality'] = 'Poor'

        # Diversity
        diversity = metrics.get('content_diversity', 0)
        if diversity >= 0.7:
            summary['content_diversity'] = 'High'
        elif diversity >= 0.4:
            summary['content_diversity'] = 'Medium'
        else:
            summary['content_diversity'] = 'Low'

        # Coverage
        coverage = metrics.get('question_coverage', 0)
        if coverage >= 0.8:
            summary['question_coverage'] = 'Comprehensive'
        elif coverage >= 0.6:
            summary['question_coverage'] = 'Good'
        elif coverage >= 0.4:
            summary['question_coverage'] = 'Partial'
        else:
            summary['question_coverage'] = 'Limited'

        return summary

    def _summarize_answer_quality(self, metrics: Dict[str, float]) -> str:
        """Summarize answer quality metrics"""
        relevance = metrics.get('answer_question_relevance', 0)
        utilization = metrics.get('context_utilization', 0)
        completeness = metrics.get('answer_completeness', 0)

        avg_score = (relevance + utilization + completeness) / 3

        if avg_score >= 0.8:
            return 'Excellent'
        elif avg_score >= 0.6:
            return 'Good'
        elif avg_score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'

    def compare_retrieval_systems(self, question: str,
                                baseline_segments: List[Dict[str, Any]],
                                improved_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two retrieval systems using automated metrics

        Args:
            question: Test question
            baseline_segments: Segments from baseline system
            improved_segments: Segments from improved system

        Returns:
            comparison: Detailed comparison report
        """
        baseline_metrics = self.evaluate_retrieval_quality(question, baseline_segments)
        improved_metrics = self.evaluate_retrieval_quality(question, improved_segments)

        comparison = {
            'question': question,
            'baseline_metrics': baseline_metrics,
            'improved_metrics': improved_metrics,
            'improvements': {},
            'regressions': {},
            'overall_improvement': 0.0
        }

        # Calculate improvements/regressions
        key_metrics = ['avg_similarity', 'content_diversity', 'question_coverage',
                      'semantic_relevance_mean', 'speaker_diversity']

        improvements = 0
        total_metrics = 0

        for metric in key_metrics:
            if metric in baseline_metrics and metric in improved_metrics:
                baseline_val = baseline_metrics[metric]
                improved_val = improved_metrics[metric]
                change = improved_val - baseline_val
                change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0

                if change > 0:
                    comparison['improvements'][metric] = {
                        'absolute_change': change,
                        'percentage_change': change_pct
                    }
                    improvements += 1
                elif change < 0:
                    comparison['regressions'][metric] = {
                        'absolute_change': change,
                        'percentage_change': change_pct
                    }

                total_metrics += 1

        comparison['overall_improvement'] = improvements / total_metrics if total_metrics > 0 else 0.0

        return comparison