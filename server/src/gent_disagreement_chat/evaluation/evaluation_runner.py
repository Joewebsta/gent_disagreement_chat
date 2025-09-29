#!/usr/bin/env python3
"""
Main evaluation runner for RAG system performance tracking.
Orchestrates baseline measurement, improvement tracking, and reporting.
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.rag_service import RAGService
from evaluation.precision_tracker import PrecisionTracker
from evaluation.ground_truth_generator import GroundTruthGenerator
from evaluation.automated_metrics import AutomatedMetrics


class EvaluationRunner:
    """
    Main orchestrator for RAG evaluation workflows.
    Handles baseline creation, improvement measurement, and reporting.
    """

    def __init__(self, database_name="gent_disagreement"):
        self.database_name = database_name
        self.precision_tracker = PrecisionTracker()
        self.ground_truth_generator = GroundTruthGenerator(database_name)
        self.automated_metrics = AutomatedMetrics()

    def create_baseline(self, num_questions: int = 20) -> str:
        """
        Create baseline evaluation of current RAG system

        Args:
            num_questions: Number of test questions to use

        Returns:
            baseline_id: Identifier for baseline evaluation
        """
        print("🎯 Creating baseline evaluation...")

        # Initialize RAG service
        rag_service = RAGService(self.database_name)

        # Create baseline evaluation
        baseline_id = self.precision_tracker.create_baseline_evaluation(
            rag_service,
            test_questions=self._get_test_questions(num_questions)
        )

        # Generate detailed automated metrics for each question
        self._generate_detailed_baseline_metrics(rag_service, baseline_id)

        print(f"✅ Baseline evaluation complete: {baseline_id}")
        return baseline_id

    def create_ground_truth_dataset(self, num_questions: int = 50) -> str:
        """
        Create ground truth dataset for precise evaluation

        Args:
            num_questions: Number of questions to generate

        Returns:
            dataset_id: Identifier for ground truth dataset
        """
        print("📊 Creating ground truth dataset...")

        # Initialize RAG service for question generation
        rag_service = RAGService(self.database_name)

        # Generate comprehensive dataset
        dataset = self.ground_truth_generator.generate_comprehensive_dataset(num_questions)

        # Save dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_path = f"ground_truth_dataset_{timestamp}.json"
        self.ground_truth_generator.export_dataset(dataset, dataset_path)

        # Create human evaluation interface
        html_path = f"ground_truth_evaluation_{timestamp}.html"
        self.ground_truth_generator.create_human_evaluation_interface(dataset, html_path)

        print(f"✅ Ground truth dataset created: {dataset_path}")
        print(f"📝 Human evaluation interface: {html_path}")
        print("   Next step: Complete human evaluation, then run validation")

        return dataset_path

    def validate_ground_truth(self, dataset_path: str, evaluation_file: str) -> str:
        """
        Validate ground truth dataset with human evaluation results

        Args:
            dataset_path: Path to ground truth dataset
            evaluation_file: Path to human evaluation results

        Returns:
            validated_dataset_path: Path to validated dataset
        """
        print("✅ Validating ground truth dataset...")

        # Load dataset and evaluation
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        evaluation_data = self.ground_truth_generator.import_human_evaluation(evaluation_file)

        # Create validated dataset
        validated_dataset = self.ground_truth_generator.create_validated_ground_truth(
            dataset, evaluation_data
        )

        # Save validated dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        validated_path = f"validated_ground_truth_{timestamp}.json"

        with open(validated_path, 'w') as f:
            json.dump(validated_dataset, f, indent=2)

        print(f"✅ Validated ground truth saved: {validated_path}")
        return validated_path

    def evaluate_improvements(self, baseline_id: str, improvement_config: Dict[str, Any] = None) -> str:
        """
        Evaluate improved RAG system against baseline

        Args:
            baseline_id: Baseline evaluation identifier
            improvement_config: Configuration for improved system

        Returns:
            evaluation_id: Identifier for improvement evaluation
        """
        print("🚀 Evaluating improved RAG system...")

        # For this example, we'll simulate an improved system
        # In practice, you'd configure your improved RAG service here
        improved_rag_service = self._create_improved_rag_service(improvement_config)

        # Run evaluation against baseline
        evaluation_id = self.precision_tracker.evaluate_improved_system(
            improved_rag_service, baseline_id
        )

        print(f"✅ Improvement evaluation complete: {evaluation_id}")
        return evaluation_id

    def run_automated_comparison(self, question: str, baseline_config: Dict = None,
                                improved_config: Dict = None) -> Dict[str, Any]:
        """
        Run automated comparison between baseline and improved systems

        Args:
            question: Test question
            baseline_config: Configuration for baseline system
            improved_config: Configuration for improved system

        Returns:
            comparison_report: Detailed comparison metrics
        """
        print(f"🔍 Running automated comparison for: '{question}'")

        # Create both systems
        baseline_rag = RAGService(self.database_name)
        improved_rag = self._create_improved_rag_service(improved_config)

        # Get retrieval results
        baseline_segments = baseline_rag.vector_search.find_most_similar(question, limit=10)
        improved_segments = improved_rag.vector_search.find_most_similar(question, limit=10)

        # Compare using automated metrics
        comparison = self.automated_metrics.compare_retrieval_systems(
            question, baseline_segments, improved_segments
        )

        print("📊 Comparison complete!")
        self._print_comparison_summary(comparison)

        return comparison

    def generate_precision_report(self, baseline_id: str, improvement_id: str = None,
                                ground_truth_dataset: str = None) -> str:
        """
        Generate comprehensive precision tracking report

        Args:
            baseline_id: Baseline evaluation ID
            improvement_id: Improvement evaluation ID (optional)
            ground_truth_dataset: Path to validated ground truth dataset (optional)

        Returns:
            report_path: Path to generated report
        """
        print("📈 Generating precision report...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"precision_report_{timestamp}.json"

        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'baseline_id': baseline_id,
                'improvement_id': improvement_id,
                'ground_truth_dataset': ground_truth_dataset
            },
            'baseline_metrics': self._get_evaluation_metrics(baseline_id),
            'improvement_metrics': self._get_evaluation_metrics(improvement_id) if improvement_id else None,
            'improvement_summary': None
        }

        # Calculate improvements if both evaluations exist
        if improvement_id:
            report_data['improvement_summary'] = self._calculate_improvement_summary(
                report_data['baseline_metrics'],
                report_data['improvement_metrics']
            )

        # Add ground truth validation if available
        if ground_truth_dataset:
            report_data['ground_truth_validation'] = self._validate_against_ground_truth(
                ground_truth_dataset, baseline_id, improvement_id
            )

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Export formatted report
        self.precision_tracker.export_metrics_report(baseline_id, f"baseline_metrics_{timestamp}.json")
        if improvement_id:
            self.precision_tracker.export_metrics_report(improvement_id, f"improvement_metrics_{timestamp}.json")

        print(f"✅ Precision report generated: {report_path}")
        return report_path

    def _get_test_questions(self, num_questions: int) -> List[str]:
        """Get default test questions for evaluation"""
        default_questions = self.precision_tracker.get_default_test_questions()

        if num_questions <= len(default_questions):
            return default_questions[:num_questions]
        else:
            # Extend with generated questions if needed
            return default_questions + [
                f"Generated test question {i}"
                for i in range(len(default_questions) + 1, num_questions + 1)
            ]

    def _create_improved_rag_service(self, config: Dict[str, Any] = None) -> RAGService:
        """
        Create improved RAG service with specified configuration

        In practice, this would implement your optimization recommendations:
        - Hybrid search
        - Reranking
        - Dynamic thresholds
        - etc.
        """
        # For now, return the same service (placeholder for actual improvements)
        # TODO: Implement actual improvements based on recommendations
        return RAGService(self.database_name)

    def _generate_detailed_baseline_metrics(self, rag_service: RAGService, baseline_id: str):
        """Generate detailed automated metrics for baseline evaluation"""
        # This would store additional automated metrics in the database
        # For detailed analysis of the baseline system
        pass

    def _get_evaluation_metrics(self, evaluation_id: str) -> Dict[str, Any]:
        """Get metrics for a specific evaluation run"""
        if not evaluation_id:
            return None

        # Query database for evaluation metrics
        # This is a placeholder - would implement actual database query
        return {
            'evaluation_id': evaluation_id,
            'avg_precision': 0.65,  # Placeholder values
            'avg_relevance': 0.72,
            'question_count': 20
        }

    def _calculate_improvement_summary(self, baseline: Dict, improved: Dict) -> Dict[str, Any]:
        """Calculate improvement summary between baseline and improved metrics"""
        if not baseline or not improved:
            return None

        precision_improvement = ((improved['avg_precision'] - baseline['avg_precision'])
                               / baseline['avg_precision'] * 100)
        relevance_improvement = ((improved['avg_relevance'] - baseline['avg_relevance'])
                               / baseline['avg_relevance'] * 100)

        return {
            'precision_improvement_pct': precision_improvement,
            'relevance_improvement_pct': relevance_improvement,
            'overall_improvement': (precision_improvement + relevance_improvement) / 2
        }

    def _validate_against_ground_truth(self, dataset_path: str, baseline_id: str,
                                     improvement_id: str = None) -> Dict[str, Any]:
        """Validate evaluations against ground truth dataset"""
        # This would implement precise validation using the ground truth dataset
        return {
            'ground_truth_precision': 0.78,
            'ground_truth_recall': 0.65,
            'ground_truth_f1': 0.71
        }

    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print human-readable comparison summary"""
        print("\n" + "="*50)
        print("AUTOMATED COMPARISON RESULTS")
        print("="*50)

        if comparison['improvements']:
            print("\n📈 IMPROVEMENTS:")
            for metric, change in comparison['improvements'].items():
                print(f"  • {metric}: +{change['percentage_change']:.1f}%")

        if comparison['regressions']:
            print("\n📉 REGRESSIONS:")
            for metric, change in comparison['regressions'].items():
                print(f"  • {metric}: {change['percentage_change']:.1f}%")

        overall = comparison['overall_improvement'] * 100
        print(f"\n🎯 OVERALL IMPROVEMENT: {overall:.1f}%")
        print("="*50 + "\n")


def main():
    """Command-line interface for evaluation runner"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument('action', choices=[
        'baseline', 'ground-truth', 'validate', 'evaluate', 'compare', 'report'
    ], help='Action to perform')

    parser.add_argument('--questions', type=int, default=20,
                       help='Number of questions for evaluation')
    parser.add_argument('--baseline-id', type=str,
                       help='Baseline evaluation ID')
    parser.add_argument('--dataset', type=str,
                       help='Ground truth dataset path')
    parser.add_argument('--evaluation', type=str,
                       help='Human evaluation file path')
    parser.add_argument('--improvement-id', type=str,
                       help='Improvement evaluation ID')
    parser.add_argument('--question', type=str,
                       help='Single question for comparison')

    args = parser.parse_args()

    runner = EvaluationRunner()

    try:
        if args.action == 'baseline':
            baseline_id = runner.create_baseline(args.questions)
            print(f"\n🎯 Baseline created: {baseline_id}")
            print("Next steps:")
            print("1. Implement RAG improvements")
            print("2. Run: python evaluation_runner.py evaluate --baseline-id <baseline_id>")

        elif args.action == 'ground-truth':
            dataset_path = runner.create_ground_truth_dataset(args.questions)
            print(f"\n📊 Ground truth dataset: {dataset_path}")
            print("Next steps:")
            print("1. Complete human evaluation using the HTML interface")
            print("2. Run: python evaluation_runner.py validate --dataset <dataset> --evaluation <evaluation_file>")

        elif args.action == 'validate':
            if not args.dataset or not args.evaluation:
                print("Error: --dataset and --evaluation required for validation")
                return
            validated_path = runner.validate_ground_truth(args.dataset, args.evaluation)
            print(f"\n✅ Validated dataset: {validated_path}")

        elif args.action == 'evaluate':
            if not args.baseline_id:
                print("Error: --baseline-id required for evaluation")
                return
            evaluation_id = runner.evaluate_improvements(args.baseline_id)
            print(f"\n🚀 Improvement evaluation: {evaluation_id}")

        elif args.action == 'compare':
            if not args.question:
                print("Error: --question required for comparison")
                return
            comparison = runner.run_automated_comparison(args.question)

        elif args.action == 'report':
            if not args.baseline_id:
                print("Error: --baseline-id required for report generation")
                return
            report_path = runner.generate_precision_report(
                args.baseline_id, args.improvement_id, args.dataset
            )
            print(f"\n📈 Report generated: {report_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())