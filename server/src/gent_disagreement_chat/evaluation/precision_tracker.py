import json
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
import sqlite3
import os
from pathlib import Path


class PrecisionTracker:
    """
    Tracks and evaluates RAG retrieval precision over time.
    Supports both automated metrics and human evaluation.
    """

    def __init__(self, evaluation_db_path="evaluation_metrics.db"):
        self.db_path = evaluation_db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for storing evaluation metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for tracking metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                configuration TEXT,
                notes TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS precision_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                query_id TEXT NOT NULL,
                question TEXT NOT NULL,
                retrieved_segments INTEGER NOT NULL,
                relevant_segments INTEGER NOT NULL,
                precision_at_k REAL NOT NULL,
                recall_at_k REAL NOT NULL,
                mrr REAL,
                ndcg REAL,
                human_relevance_score REAL,
                automated_relevance_score REAL,
                FOREIGN KEY (run_id) REFERENCES evaluation_runs (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ground_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT UNIQUE NOT NULL,
                question TEXT NOT NULL,
                relevant_episode_segments TEXT NOT NULL,  -- JSON array
                irrelevant_episode_segments TEXT,         -- JSON array
                difficulty_level TEXT DEFAULT 'medium',   -- easy, medium, hard
                category TEXT,                             -- factual, analytical, comparative
                created_by TEXT DEFAULT 'system',
                created_at TEXT NOT NULL,
                validated BOOLEAN DEFAULT FALSE
            )
        """)

        conn.commit()
        conn.close()

    def create_baseline_evaluation(self, rag_service, test_questions: List[str] = None) -> str:
        """
        Create baseline evaluation with current RAG implementation

        Args:
            rag_service: Current RAG service instance
            test_questions: List of test questions, if None uses default set

        Returns:
            run_id: Unique identifier for this evaluation run
        """
        if test_questions is None:
            test_questions = self.get_default_test_questions()

        # Create new evaluation run
        run_id = self._create_evaluation_run("baseline", "Current implementation")

        results = []
        for i, question in enumerate(test_questions):
            query_id = f"baseline_q_{i+1}"

            # Get current retrieval results
            search_results = rag_service.vector_search.find_most_similar(question, limit=10)

            # Calculate metrics (initial assessment without ground truth)
            metrics = self._calculate_initial_metrics(query_id, question, search_results)

            # Store in database
            self._store_precision_metrics(run_id, query_id, question, metrics)
            results.append(metrics)

        print(f"Baseline evaluation complete. Run ID: {run_id}")
        print(f"Average Precision@5: {np.mean([r['precision_at_5'] for r in results]):.3f}")

        return run_id

    def evaluate_improved_system(self, improved_rag_service, baseline_run_id: str) -> str:
        """
        Evaluate improved RAG system against baseline

        Args:
            improved_rag_service: New RAG service implementation
            baseline_run_id: ID of baseline run to compare against

        Returns:
            run_id: Unique identifier for this evaluation run
        """
        # Get baseline questions
        baseline_questions = self._get_baseline_questions(baseline_run_id)

        # Create new evaluation run
        run_id = self._create_evaluation_run("improved", "Post-optimization implementation")

        results = []
        for question_data in baseline_questions:
            query_id = question_data['query_id'].replace('baseline_', 'improved_')
            question = question_data['question']

            # Get improved retrieval results
            search_results = improved_rag_service.vector_search.find_most_similar(question, limit=10)

            # Calculate metrics
            metrics = self._calculate_initial_metrics(query_id, question, search_results)

            # Store in database
            self._store_precision_metrics(run_id, query_id, question, metrics)
            results.append(metrics)

        # Generate comparison report
        self._generate_comparison_report(baseline_run_id, run_id)

        return run_id

    def create_ground_truth_dataset(self, rag_service, num_questions: int = 50) -> str:
        """
        Generate ground truth dataset for more accurate evaluation

        Args:
            rag_service: RAG service to use for generating test cases
            num_questions: Number of questions to generate

        Returns:
            dataset_id: Identifier for the created dataset
        """
        questions = self._generate_diverse_questions(rag_service, num_questions)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dataset_id = f"gt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for i, question_data in enumerate(questions):
            query_id = f"{dataset_id}_q_{i+1}"

            # Get retrieval results to help with ground truth creation
            search_results = rag_service.vector_search.find_most_similar(
                question_data['question'], limit=20
            )

            # Store in ground truth table (without validation initially)
            cursor.execute("""
                INSERT INTO ground_truth
                (query_id, question, relevant_episode_segments, category, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query_id,
                question_data['question'],
                json.dumps([]),  # To be filled by human evaluation
                question_data['category'],
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

        print(f"Ground truth dataset created: {dataset_id}")
        print(f"Generated {len(questions)} questions for evaluation")
        print("Next step: Run human evaluation to validate ground truth")

        return dataset_id

    def calculate_precision_with_ground_truth(self, rag_service, ground_truth_dataset_id: str) -> Dict[str, float]:
        """
        Calculate precise metrics using validated ground truth

        Args:
            rag_service: RAG service to evaluate
            ground_truth_dataset_id: ID of ground truth dataset

        Returns:
            metrics: Dictionary of calculated metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get validated ground truth questions
        cursor.execute("""
            SELECT query_id, question, relevant_episode_segments
            FROM ground_truth
            WHERE query_id LIKE ? AND validated = TRUE
        """, (f"{ground_truth_dataset_id}%",))

        ground_truth_data = cursor.fetchall()
        conn.close()

        if not ground_truth_data:
            raise ValueError(f"No validated ground truth found for dataset {ground_truth_dataset_id}")

        all_metrics = []

        for query_id, question, relevant_segments_json in ground_truth_data:
            relevant_segments = json.loads(relevant_segments_json)

            # Get retrieval results
            search_results = rag_service.vector_search.find_most_similar(question, limit=10)

            # Calculate precise metrics
            metrics = self._calculate_precise_metrics(search_results, relevant_segments)
            all_metrics.append(metrics)

        # Aggregate metrics
        aggregated = {
            'precision_at_1': np.mean([m['precision_at_1'] for m in all_metrics]),
            'precision_at_3': np.mean([m['precision_at_3'] for m in all_metrics]),
            'precision_at_5': np.mean([m['precision_at_5'] for m in all_metrics]),
            'recall_at_5': np.mean([m['recall_at_5'] for m in all_metrics]),
            'recall_at_10': np.mean([m['recall_at_10'] for m in all_metrics]),
            'mrr': np.mean([m['mrr'] for m in all_metrics]),
            'ndcg_at_5': np.mean([m['ndcg_at_5'] for m in all_metrics]),
            'ndcg_at_10': np.mean([m['ndcg_at_10'] for m in all_metrics])
        }

        return aggregated

    def _calculate_initial_metrics(self, query_id: str, question: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Calculate basic metrics without ground truth"""
        metrics = {
            'query_id': query_id,
            'question': question,
            'retrieved_segments': len(search_results),
            'avg_similarity': np.mean([r.get('similarity', 0) for r in search_results]) if search_results else 0,
            'min_similarity': min([r.get('similarity', 0) for r in search_results]) if search_results else 0,
            'similarity_std': np.std([r.get('similarity', 0) for r in search_results]) if search_results else 0,
            # Initial estimates (to be refined with ground truth)
            'precision_at_3': self._estimate_precision(search_results[:3]),
            'precision_at_5': self._estimate_precision(search_results[:5]),
            'estimated_relevance': self._estimate_overall_relevance(search_results)
        }
        return metrics

    def _calculate_precise_metrics(self, search_results: List[Dict], relevant_segments: List[str]) -> Dict[str, float]:
        """Calculate precise metrics using ground truth"""
        retrieved_ids = [self._get_segment_id(result) for result in search_results]
        relevant_set = set(relevant_segments)

        # Calculate precision at different k values
        def precision_at_k(k):
            if k > len(retrieved_ids):
                k = len(retrieved_ids)
            if k == 0:
                return 0.0
            return len(set(retrieved_ids[:k]) & relevant_set) / k

        # Calculate recall at different k values
        def recall_at_k(k):
            if len(relevant_set) == 0:
                return 1.0 if k == 0 else 0.0
            if k > len(retrieved_ids):
                k = len(retrieved_ids)
            return len(set(retrieved_ids[:k]) & relevant_set) / len(relevant_set)

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        def ndcg_at_k(k):
            if k > len(retrieved_ids):
                k = len(retrieved_ids)

            dcg = 0.0
            for i in range(k):
                if retrieved_ids[i] in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)

            # Ideal DCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))

            return dcg / idcg if idcg > 0 else 0.0

        return {
            'precision_at_1': precision_at_k(1),
            'precision_at_3': precision_at_k(3),
            'precision_at_5': precision_at_k(5),
            'recall_at_5': recall_at_k(5),
            'recall_at_10': recall_at_k(10),
            'mrr': mrr,
            'ndcg_at_5': ndcg_at_k(5),
            'ndcg_at_10': ndcg_at_k(10)
        }

    def _estimate_precision(self, results: List[Dict]) -> float:
        """Estimate precision based on similarity scores (adjusted for podcast content)"""
        if not results:
            return 0.0

        # Adjusted thresholds for long-form podcast content
        # Podcast segments are longer and more complex, so lower thresholds may be appropriate
        high_similarity_count = sum(1 for r in results if r.get('similarity', 0) > 0.6)
        medium_similarity_count = sum(1 for r in results if r.get('similarity', 0) > 0.4)

        # Weighted scoring: high similarity gets full credit, medium gets partial
        weighted_score = (high_similarity_count + 0.5 * medium_similarity_count) / len(results)
        return weighted_score

    def _estimate_overall_relevance(self, results: List[Dict]) -> float:
        """Estimate overall relevance of results (optimized for podcast content)"""
        if not results:
            return 0.0

        # Weighted by position and similarity, with adjustments for long-form content
        relevance_score = 0.0
        for i, result in enumerate(results):
            # Less steep position decay for podcast content since longer segments
            # may contain relevant information even if not perfectly similar
            position_weight = 1.0 / (i + 1) ** 0.7  # Gentler decay than 1.0

            similarity = result.get('similarity', 0)

            # Boost score for podcast-specific indicators
            text = result.get('text', '').lower()
            has_analysis = any(term in text for term in ['analyze', 'perspective', 'framework', 'implications'])
            has_expertise = any(term in text for term in ['professor', 'expert', 'research', 'study'])

            content_boost = 1.0
            if has_analysis:
                content_boost += 0.1
            if has_expertise:
                content_boost += 0.1

            relevance_score += position_weight * similarity * content_boost

        normalization_factor = sum(1.0 / (i + 1) ** 0.7 for i in range(len(results)))
        return relevance_score / normalization_factor

    def _get_segment_id(self, result: Dict) -> str:
        """Generate unique segment ID for comparison"""
        return f"ep{result.get('episode_number', 0)}_seg_{hash(result.get('text', ''))}"

    def _create_evaluation_run(self, version: str, notes: str) -> str:
        """Create new evaluation run record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        run_id = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cursor.execute("""
            INSERT INTO evaluation_runs (timestamp, model_version, notes)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), run_id, notes))

        conn.commit()
        conn.close()

        return run_id

    def _store_precision_metrics(self, run_id: str, query_id: str, question: str, metrics: Dict):
        """Store calculated metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO precision_metrics
            (run_id, query_id, question, retrieved_segments, relevant_segments,
             precision_at_k, recall_at_k, automated_relevance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            query_id,
            question,
            metrics.get('retrieved_segments', 0),
            0,  # Will be updated with ground truth
            metrics.get('precision_at_5', 0),
            0,  # Will be calculated with ground truth
            metrics.get('estimated_relevance', 0)
        ))

        conn.commit()
        conn.close()

    def _get_baseline_questions(self, baseline_run_id: str) -> List[Dict]:
        """Get questions from baseline run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query_id, question
            FROM precision_metrics
            WHERE run_id = ?
        """, (baseline_run_id,))

        results = [{'query_id': row[0], 'question': row[1]} for row in cursor.fetchall()]
        conn.close()

        return results

    def _generate_comparison_report(self, baseline_run_id: str, improved_run_id: str):
        """Generate comparison report between baseline and improved systems"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get baseline metrics
        cursor.execute("""
            SELECT AVG(precision_at_k), AVG(automated_relevance_score)
            FROM precision_metrics
            WHERE run_id = ?
        """, (baseline_run_id,))
        baseline_avg = cursor.fetchone()

        # Get improved metrics
        cursor.execute("""
            SELECT AVG(precision_at_k), AVG(automated_relevance_score)
            FROM precision_metrics
            WHERE run_id = ?
        """, (improved_run_id,))
        improved_avg = cursor.fetchone()

        conn.close()

        if baseline_avg and improved_avg:
            precision_improvement = ((improved_avg[0] - baseline_avg[0]) / baseline_avg[0]) * 100
            relevance_improvement = ((improved_avg[1] - baseline_avg[1]) / baseline_avg[1]) * 100

            print(f"\n=== RAG Improvement Report ===")
            print(f"Baseline Run: {baseline_run_id}")
            print(f"Improved Run: {improved_run_id}")
            print(f"Precision@5 Improvement: {precision_improvement:+.1f}%")
            print(f"Relevance Score Improvement: {relevance_improvement:+.1f}%")
            print(f"================================\n")

    def _generate_diverse_questions(self, rag_service, num_questions: int) -> List[Dict]:
        """Generate diverse test questions for evaluation"""
        # This would ideally use episode metadata to generate realistic questions
        categories = ['factual', 'analytical', 'comparative', 'opinion']

        questions = []
        for i in range(num_questions):
            category = categories[i % len(categories)]

            # Generate category-specific questions
            if category == 'factual':
                question = f"What did the hosts say about specific topic {i}?"
            elif category == 'analytical':
                question = f"How do the hosts analyze the relationship between concept A and B in episode discussions?"
            elif category == 'comparative':
                question = f"What are the different perspectives the hosts have shared on controversial topic {i}?"
            else:  # opinion
                question = f"What are the hosts' personal opinions on current event {i}?"

            questions.append({
                'question': question,
                'category': category
            })

        return questions

    def get_default_test_questions(self) -> List[str]:
        """Default set of test questions for podcast evaluation"""
        return [
            "What did Professor Jack Bierman say about Supreme Court textualism and originalism?",
            "How do the hosts analyze the constitutional implications of executive power expansion?",
            "What are Ricky and Brendan's different perspectives on federal versus state authority?",
            "What economic impacts of tariffs did the hosts discuss with their economist guest?",
            "How do the hosts evaluate the effectiveness of current immigration policies?",
            "What legal precedents do the hosts reference when discussing voting rights?",
            "What are the hosts' views on the politicization of the Supreme Court?",
            "How do the hosts connect current political events to historical patterns?",
            "What constitutional principles do the hosts invoke when analyzing government actions?",
            "What specific Supreme Court cases do the hosts analyze in recent episodes?",
            "How do the hosts discuss the balance between security and civil liberties?",
            "What are the hosts' perspectives on gerrymandering and electoral integrity?",
            "How do guest experts contribute to the hosts' analysis of complex legal issues?",
            "What foreign policy implications do the hosts identify in domestic political decisions?",
            "How do the hosts evaluate the role of federal agencies in policy implementation?"
        ]

    def export_metrics_report(self, run_id: str, output_path: str):
        """Export detailed metrics report to file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM precision_metrics WHERE run_id = ?
        """, (run_id,))

        metrics_data = cursor.fetchall()
        conn.close()

        # Create detailed report
        report = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': [dict(zip([col[0] for col in cursor.description], row)) for row in metrics_data],
            'summary': {
                'total_questions': len(metrics_data),
                'avg_precision': np.mean([row[6] for row in metrics_data]),
                'avg_relevance': np.mean([row[9] for row in metrics_data if row[9] is not None])
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Metrics report exported to: {output_path}")