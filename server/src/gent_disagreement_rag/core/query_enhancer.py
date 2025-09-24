import re
from typing import Dict, List, Any
from openai import OpenAI
import os


class QueryEnhancer:
    """Handles query preprocessing and enhancement for better retrieval"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Podcast-specific vocabulary expansion based on actual transcript analysis
        self.podcast_vocabulary = {
            "hosts": [
                "Ricky Ghoshroy",
                "Ricky",
                "Brendan Kelly",
                "Brendan",
                "hosts",
                "speakers",
                "panelists",
            ],
            "legal_topics": [
                "Supreme Court",
                "SCOTUS",
                "constitutional",
                "originalism",
                "textualism",
                "court cases",
                "judicial",
                "legal reasoning",
            ],
            "economic_topics": [
                "tariffs",
                "trade",
                "Bureau of Labor Statistics",
                "BLS",
                "Federal Reserve",
                "Fed",
                "markets",
                "inflation",
                "labor",
                "economy",
            ],
            "political_topics": [
                "authoritarianism",
                "democracy",
                "soft power",
                "foreign policy",
                "Trump administration",
                "conservative",
                "liberal",
            ],
            "guest_experts": [
                "Professor",
                "expert",
                "analyst",
                "journalist",
                "academic",
            ],
            "general_concepts": [
                "politics",
                "economics",
                "culture",
                "philosophy",
                "governance",
                "policy",
            ],
        }

    def enhance_query(self, question: str) -> Dict[str, Any]:
        """
        Enhance query using multiple strategies

        Args:
            question: Original user question

        Returns:
            Enhanced query data with multiple variations
        """
        return {
            "original": question,
            "hyde": self._generate_hypothetical_answer(question),
            "expanded": self._expand_query_keywords(question),
            "intent": self._classify_query_intent(question),
            "cleaned": self._clean_and_normalize(question),
        }

    def _generate_hypothetical_answer(self, question: str) -> str:
        """Generate hypothetical answer for HyDE approach"""
        try:
            # Classify the query intent to tailor the HyDE prompt
            intent = self._classify_query_intent(question)

            base_prompt = f'Based on the question "{question}", generate a hypothetical answer that might appear in the "A Gentleman\'s Disagreement" podcast transcript.'

            if intent == "legal_analysis":
                specific_prompt = "Focus on constitutional law, Supreme Court cases, legal reasoning, and judicial interpretation. Include analytical discussion typical of legal experts."
            elif intent == "economic_policy":
                specific_prompt = "Focus on economic analysis, policy implications, market trends, and data-driven discussion typical of economic experts."
            elif intent == "historical_comparison":
                specific_prompt = "Focus on historical parallels, past precedents, and comparative analysis across different eras."
            elif intent == "expert_opinion":
                specific_prompt = "Focus on academic expertise, scholarly analysis, and expert commentary from professors or journalists."
            else:
                specific_prompt = "Focus on thoughtful political commentary, nuanced debate, and analytical discussion between the hosts."

            prompt = f"{base_prompt} {specific_prompt} Keep it concise (1-2 sentences) and match the intellectual, conversational tone of the podcast."

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating hypothetical answer: {e}")
            return question  # Fallback to original

    def _expand_query_keywords(self, question: str) -> str:
        """Expand query with relevant keywords and synonyms"""
        expanded_terms = []

        # Add original terms
        expanded_terms.extend(question.split())

        # Add podcast-specific vocabulary if relevant
        question_lower = question.lower()
        for _category, terms in self.podcast_vocabulary.items():
            if any(term in question_lower for term in terms):
                expanded_terms.extend([t for t in terms if t not in question_lower])

        # Add common synonyms
        expanded_terms.extend(self._get_synonyms(question))

        return " ".join(list(set(expanded_terms)))  # Remove duplicates

    def _classify_query_intent(self, question: str) -> str:
        """Classify the intent of the query based on podcast content patterns"""
        question_lower = question.lower()

        # Legal analysis queries
        if any(
            word in question_lower
            for word in [
                "court",
                "constitutional",
                "legal",
                "ruling",
                "case",
                "justice",
                "judge",
            ]
        ):
            return "legal_analysis"
        # Economic policy queries
        elif any(
            word in question_lower
            for word in [
                "tariff",
                "trade",
                "economy",
                "market",
                "fed",
                "inflation",
                "labor",
            ]
        ):
            return "economic_policy"
        # Historical comparison queries
        elif any(
            word in question_lower
            for word in ["history", "past", "compare", "historical", "era", "previous"]
        ):
            return "historical_comparison"
        # Guest expert queries
        elif any(
            word in question_lower
            for word in ["professor", "expert", "guest", "interview"]
        ):
            return "expert_opinion"
        # Factual queries
        elif any(
            word in question_lower for word in ["what", "who", "when", "where", "which"]
        ):
            return "factual"
        # Analytical queries
        elif any(
            word in question_lower for word in ["how", "why", "explain", "analyze"]
        ):
            return "analytical"
        # Comparative queries
        elif any(
            word in question_lower
            for word in ["compare", "versus", "difference", "contrast"]
        ):
            return "comparative"
        # Opinion queries
        elif any(
            word in question_lower
            for word in ["opinion", "think", "believe", "feel", "view"]
        ):
            return "opinion"
        else:
            return "general"

    def _clean_and_normalize(self, question: str) -> str:
        """Clean and normalize the question"""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", question.strip())

        # Remove common question words that don't add semantic value
        stop_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        ]
        words = cleaned.split()
        cleaned_words = [w for w in words if w.lower() not in stop_words]

        return " ".join(cleaned_words)

    def _get_synonyms(self, question: str) -> List[str]:
        """Get synonyms for key terms in the question"""
        # Enhanced synonym mapping based on podcast analysis
        synonym_map = {
            "discuss": [
                "talk about",
                "cover",
                "address",
                "explore",
                "analyze",
                "examine",
            ],
            "opinion": [
                "view",
                "perspective",
                "stance",
                "position",
                "take",
                "thoughts",
            ],
            "topic": ["subject", "theme", "issue", "matter", "question"],
            "hosts": [
                "speakers",
                "panelists",
                "Ricky",
                "Brendan",
                "Ricky Ghoshroy",
                "Brendan Kelly",
            ],
            "court": ["Supreme Court", "SCOTUS", "judicial", "tribunal"],
            "economy": ["economic", "markets", "trade", "financial", "fiscal"],
            "government": ["administration", "federal", "state", "political"],
            "policy": ["regulation", "rule", "law", "governance"],
            "conservative": ["right-wing", "Republican", "traditional"],
            "liberal": ["left-wing", "Democratic", "progressive"],
        }

        synonyms = []
        question_lower = question.lower()

        for term, synonym_list in synonym_map.items():
            if term in question_lower:
                synonyms.extend(synonym_list)

        return synonyms
