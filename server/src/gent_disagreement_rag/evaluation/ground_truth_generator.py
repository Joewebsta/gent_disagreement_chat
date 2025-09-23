import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
import sqlite3
from ..core.database_manager import DatabaseManager


class GroundTruthGenerator:
    """
    Generates ground truth datasets for RAG evaluation.
    Creates realistic questions and identifies relevant segments.
    """

    def __init__(self, database_name="gent_disagreement"):
        self.db_manager = DatabaseManager(database=database_name)

    def generate_comprehensive_dataset(self, num_questions: int = 100) -> Dict[str, Any]:
        """
        Generate a comprehensive ground truth dataset with diverse question types

        Args:
            num_questions: Total number of questions to generate

        Returns:
            dataset: Dictionary containing questions and metadata
        """
        # Get episode metadata for context
        episodes = self._get_episode_metadata()
        speakers = self._get_unique_speakers()
        topics = self._extract_common_topics()

        # Question distribution
        question_types = {
            'factual': 0.3,
            'analytical': 0.25,
            'comparative': 0.2,
            'opinion': 0.15,
            'chronological': 0.1
        }

        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_questions': num_questions,
                'episode_count': len(episodes),
                'speaker_count': len(speakers),
                'question_distribution': question_types
            },
            'questions': []
        }

        for i in range(num_questions):
            # Select question type based on distribution
            question_type = self._select_question_type(question_types)

            # Generate question based on type
            question_data = self._generate_question_by_type(
                question_type, episodes, speakers, topics, i
            )

            # Find relevant segments for this question
            relevant_segments = self._find_relevant_segments(question_data['question'])

            question_entry = {
                'id': f"gt_q_{i+1:03d}",
                'question': question_data['question'],
                'type': question_type,
                'difficulty': question_data['difficulty'],
                'expected_answer_type': question_data['answer_type'],
                'relevant_segments': relevant_segments,
                'metadata': question_data.get('metadata', {})
            }

            dataset['questions'].append(question_entry)

        return dataset

    def create_human_evaluation_interface(self, dataset: Dict[str, Any], output_path: str):
        """
        Create HTML interface for human evaluation of ground truth

        Args:
            dataset: Generated dataset
            output_path: Path to save HTML file
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Ground Truth Evaluation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .question-block { border: 1px solid #ddd; margin: 20px 0; padding: 20px; }
                .segment { border-left: 3px solid #ccc; margin: 10px 0; padding: 10px; }
                .relevant { border-left-color: #4CAF50; background-color: #f9fff9; }
                .irrelevant { border-left-color: #f44336; background-color: #fff9f9; }
                .rating { margin: 10px 0; }
                button { margin: 5px; padding: 8px 16px; }
                .save-btn { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                .metadata { font-size: 0.9em; color: #666; margin-bottom: 10px; }
            </style>
            <script>
                let evaluationData = {};

                function markRelevant(questionId, segmentId, isRelevant) {
                    if (!evaluationData[questionId]) {
                        evaluationData[questionId] = { relevant: [], irrelevant: [] };
                    }

                    // Remove from both arrays first
                    evaluationData[questionId].relevant = evaluationData[questionId].relevant.filter(id => id !== segmentId);
                    evaluationData[questionId].irrelevant = evaluationData[questionId].irrelevant.filter(id => id !== segmentId);

                    // Add to appropriate array
                    if (isRelevant) {
                        evaluationData[questionId].relevant.push(segmentId);
                    } else {
                        evaluationData[questionId].irrelevant.push(segmentId);
                    }

                    // Update UI
                    updateSegmentDisplay(questionId, segmentId, isRelevant);
                }

                function updateSegmentDisplay(questionId, segmentId, isRelevant) {
                    const segment = document.getElementById(`segment_${questionId}_${segmentId}`);
                    segment.className = `segment ${isRelevant ? 'relevant' : 'irrelevant'}`;
                }

                function rateQuestion(questionId, difficulty) {
                    if (!evaluationData[questionId]) {
                        evaluationData[questionId] = { relevant: [], irrelevant: [] };
                    }
                    evaluationData[questionId].difficulty = difficulty;
                }

                function exportEvaluation() {
                    const dataStr = JSON.stringify(evaluationData, null, 2);
                    const dataBlob = new Blob([dataStr], {type: 'application/json'});
                    const url = URL.createObjectURL(dataBlob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'ground_truth_evaluation.json';
                    link.click();
                }

                function saveProgress() {
                    localStorage.setItem('rag_evaluation_progress', JSON.stringify(evaluationData));
                    alert('Progress saved locally!');
                }

                function loadProgress() {
                    const saved = localStorage.getItem('rag_evaluation_progress');
                    if (saved) {
                        evaluationData = JSON.parse(saved);
                        alert('Progress loaded!');
                        // Update UI to reflect loaded data
                        // This would require more complex state management
                    }
                }
            </script>
        </head>
        <body>
            <h1>RAG Ground Truth Evaluation</h1>
            <div class="metadata">
                <p><strong>Dataset:</strong> {total_questions} questions | Created: {created_at}</p>
                <button onclick="loadProgress()">Load Progress</button>
                <button onclick="saveProgress()">Save Progress</button>
                <button onclick="exportEvaluation()" class="save-btn">Export Results</button>
            </div>

            {questions_html}

            <div style="margin-top: 40px;">
                <button onclick="exportEvaluation()" class="save-btn">Export Final Results</button>
            </div>
        </body>
        </html>
        """

        questions_html = ""
        for question in dataset['questions']:
            questions_html += self._generate_question_html(question)

        final_html = html_template.format(
            total_questions=dataset['metadata']['total_questions'],
            created_at=dataset['metadata']['created_at'],
            questions_html=questions_html
        )

        with open(output_path, 'w') as f:
            f.write(final_html)

        print(f"Human evaluation interface created: {output_path}")

    def _generate_question_html(self, question: Dict[str, Any]) -> str:
        """Generate HTML for a single question evaluation"""
        segments_html = ""
        for i, segment in enumerate(question['relevant_segments']):
            segment_id = f"seg_{i}"
            segments_html += f"""
            <div id="segment_{question['id']}_{segment_id}" class="segment">
                <p><strong>Episode {segment['episode_number']}</strong> - {segment['speaker']}</p>
                <p>{segment['text']}</p>
                <p><small>Similarity: {segment['similarity']:.3f}</small></p>
                <button onclick="markRelevant('{question['id']}', '{segment_id}', true)">Relevant</button>
                <button onclick="markRelevant('{question['id']}', '{segment_id}', false)">Irrelevant</button>
            </div>
            """

        return f"""
        <div class="question-block">
            <h3>Question {question['id']}</h3>
            <div class="metadata">
                Type: {question['type']} | Difficulty: {question['difficulty']} | Expected: {question['expected_answer_type']}
            </div>
            <p><strong>Question:</strong> {question['question']}</p>

            <div class="rating">
                <strong>Rate Difficulty:</strong>
                <button onclick="rateQuestion('{question['id']}', 'easy')">Easy</button>
                <button onclick="rateQuestion('{question['id']}', 'medium')">Medium</button>
                <button onclick="rateQuestion('{question['id']}', 'hard')">Hard</button>
            </div>

            <h4>Retrieved Segments (mark as relevant/irrelevant):</h4>
            {segments_html}
        </div>
        """

    def _get_episode_metadata(self) -> List[Dict[str, Any]]:
        """Get all episode metadata"""
        query = """
            SELECT id, episode_number, title, date_published,
                   description, duration_minutes
            FROM episodes
            ORDER BY episode_number
        """
        return self.db_manager.execute_query(query)

    def _get_unique_speakers(self) -> List[str]:
        """Get list of unique speakers"""
        query = """
            SELECT DISTINCT speaker
            FROM transcript_segments
            WHERE speaker IS NOT NULL AND speaker != ''
            ORDER BY speaker
        """
        results = self.db_manager.execute_query(query)
        return [row['speaker'] for row in results]

    def _extract_common_topics(self) -> List[str]:
        """Extract common topics from transcript text (simplified approach)"""
        # This is a simplified version - in practice, you'd use NLP techniques
        common_topics = [
            "politics", "technology", "culture", "society", "economics",
            "philosophy", "current events", "personal experiences", "books",
            "media", "relationships", "career", "education", "health"
        ]
        return common_topics

    def _select_question_type(self, distributions: Dict[str, float]) -> str:
        """Select question type based on probability distribution"""
        rand = random.random()
        cumulative = 0.0

        for q_type, prob in distributions.items():
            cumulative += prob
            if rand <= cumulative:
                return q_type

        return list(distributions.keys())[-1]  # fallback

    def _generate_question_by_type(self, question_type: str, episodes: List[Dict],
                                  speakers: List[str], topics: List[str], index: int) -> Dict[str, Any]:
        """Generate question based on type"""
        if question_type == 'factual':
            return self._generate_factual_question(episodes, speakers, topics, index)
        elif question_type == 'analytical':
            return self._generate_analytical_question(topics, index)
        elif question_type == 'comparative':
            return self._generate_comparative_question(speakers, topics, index)
        elif question_type == 'opinion':
            return self._generate_opinion_question(speakers, topics, index)
        elif question_type == 'chronological':
            return self._generate_chronological_question(episodes, index)
        else:
            return self._generate_factual_question(episodes, speakers, topics, index)

    def _generate_factual_question(self, episodes: List[Dict], speakers: List[str],
                                  topics: List[str], index: int) -> Dict[str, Any]:
        """Generate factual questions"""
        templates = [
            f"What did {random.choice(speakers)} say about {random.choice(topics)}?",
            f"What specific examples did the hosts give when discussing {random.choice(topics)}?",
            f"What facts or statistics were mentioned in episode {random.choice(episodes)['episode_number']}?",
            f"What book/movie/article recommendations did {random.choice(speakers)} make?",
            f"What personal anecdotes did the hosts share about {random.choice(topics)}?"
        ]

        return {
            'question': random.choice(templates),
            'difficulty': 'easy',
            'answer_type': 'specific_fact',
            'metadata': {'requires_exact_match': True}
        }

    def _generate_analytical_question(self, topics: List[str], index: int) -> Dict[str, Any]:
        """Generate analytical questions"""
        topic1, topic2 = random.sample(topics, 2)
        templates = [
            f"How do the hosts analyze the relationship between {topic1} and {topic2}?",
            f"What underlying patterns do the hosts identify in {topic1} discussions?",
            f"How do the hosts break down complex {topic1} issues?",
            f"What framework do the hosts use to understand {topic1}?",
            f"What are the root causes the hosts identify for {topic1} problems?"
        ]

        return {
            'question': random.choice(templates),
            'difficulty': 'medium',
            'answer_type': 'analytical_framework',
            'metadata': {'requires_synthesis': True}
        }

    def _generate_comparative_question(self, speakers: List[str], topics: List[str], index: int) -> Dict[str, Any]:
        """Generate comparative questions"""
        if len(speakers) >= 2:
            speaker1, speaker2 = random.sample(speakers, 2)
            templates = [
                f"How do {speaker1} and {speaker2} differ in their views on {random.choice(topics)}?",
                f"What are the contrasting perspectives between {speaker1} and {speaker2} regarding {random.choice(topics)}?",
                f"Where do {speaker1} and {speaker2} find common ground on {random.choice(topics)}?",
                f"How have {speaker1}'s and {speaker2}'s views on {random.choice(topics)} evolved over time?",
            ]
        else:
            templates = [
                f"What different perspectives have the hosts presented on {random.choice(topics)}?",
                f"How have the hosts' views on {random.choice(topics)} changed over time?",
                f"What contradictory viewpoints have been discussed about {random.choice(topics)}?"
            ]

        return {
            'question': random.choice(templates),
            'difficulty': 'medium',
            'answer_type': 'comparative_analysis',
            'metadata': {'requires_multiple_sources': True}
        }

    def _generate_opinion_question(self, speakers: List[str], topics: List[str], index: int) -> Dict[str, Any]:
        """Generate opinion-based questions"""
        templates = [
            f"What are {random.choice(speakers)}'s personal opinions on {random.choice(topics)}?",
            f"How do the hosts feel about recent developments in {random.choice(topics)}?",
            f"What controversial stance have the hosts taken on {random.choice(topics)}?",
            f"What are the hosts' predictions about the future of {random.choice(topics)}?",
            f"What personal values do the hosts express when discussing {random.choice(topics)}?"
        ]

        return {
            'question': random.choice(templates),
            'difficulty': 'hard',
            'answer_type': 'subjective_opinion',
            'metadata': {'requires_interpretation': True}
        }

    def _generate_chronological_question(self, episodes: List[Dict], index: int) -> Dict[str, Any]:
        """Generate chronological questions"""
        recent_episodes = sorted(episodes, key=lambda x: x.get('episode_number', 0), reverse=True)[:5]
        older_episodes = sorted(episodes, key=lambda x: x.get('episode_number', 0))[:5]

        templates = [
            f"How have the hosts' discussions evolved from early episodes to recent ones?",
            f"What topics were covered in episode {random.choice(recent_episodes)['episode_number']} versus episode {random.choice(older_episodes)['episode_number']}?",
            f"What recurring themes appear across multiple episodes?",
            f"How have current events influenced the hosts' discussions over time?",
            f"What topics have the hosts revisited in recent episodes?"
        ]

        return {
            'question': random.choice(templates),
            'difficulty': 'hard',
            'answer_type': 'temporal_analysis',
            'metadata': {'requires_episode_comparison': True}
        }

    def _find_relevant_segments(self, question: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Find potentially relevant segments for a question using current RAG system"""
        try:
            # Use existing embedding service to find segments
            from ..core.embedding_service import EmbeddingService
            from ..core.vector_search import VectorSearch

            embedding_service = EmbeddingService(self.db_manager)
            vector_search = VectorSearch()

            # Get top segments
            results = vector_search.find_most_similar(question, limit=limit)

            return results

        except Exception as e:
            print(f"Error finding relevant segments: {e}")
            return []

    def export_dataset(self, dataset: Dict[str, Any], output_path: str):
        """Export dataset to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"Ground truth dataset exported to: {output_path}")

    def import_human_evaluation(self, evaluation_file: str) -> Dict[str, Any]:
        """Import human evaluation results"""
        with open(evaluation_file, 'r') as f:
            evaluation_data = json.load(f)

        print(f"Imported evaluation for {len(evaluation_data)} questions")
        return evaluation_data

    def create_validated_ground_truth(self, dataset: Dict[str, Any],
                                    evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create final validated ground truth from human evaluation"""
        validated_dataset = {
            'metadata': dataset['metadata'].copy(),
            'validation_info': {
                'validated_at': datetime.now().isoformat(),
                'validated_questions': len(evaluation_data)
            },
            'questions': []
        }

        for question in dataset['questions']:
            question_id = question['id']
            if question_id in evaluation_data:
                eval_data = evaluation_data[question_id]

                validated_question = question.copy()
                validated_question['human_validation'] = {
                    'relevant_segments': eval_data.get('relevant', []),
                    'irrelevant_segments': eval_data.get('irrelevant', []),
                    'difficulty_rating': eval_data.get('difficulty', question['difficulty']),
                    'validated': True
                }

                validated_dataset['questions'].append(validated_question)

        return validated_dataset