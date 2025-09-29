# Question 1
Create an implementation plan to implement dynamic threshold filtering, query enhancement and enhanced context formatting.

# Response 1

I'll create a comprehensive implementation plan for the three optimization strategies. Let me first examine the current codebase structure and existing implementations to understand what needs to be done.


[3 tools called]

Now let me examine the existing optimization documentation to understand the specific requirements for each component:


[2 tools called]

Now let me check the current evaluation framework to understand how we can measure the improvements:


[2 tools called]

Now I have a comprehensive understanding of the current system. Let me create a detailed implementation plan for the three optimization strategies.

# RAG Optimization Implementation Plan

Based on my analysis of the current codebase and the optimization recommendations, here's a comprehensive implementation plan for implementing **Dynamic Threshold Filtering**, **Query Enhancement**, and **Enhanced Context Formatting**.

## **Phase 1: Dynamic Threshold Filtering** 🎯

### **Current State Analysis**
- Currently uses `find_most_similar()` with fixed `limit=5`
- No quality control - retrieves top N results regardless of relevance
- Commented-out `find_similar_above_threshold()` method doesn't exist

### **Implementation Steps**

#### **Step 1.1: Extend VectorSearch Class**
**File**: `server/src/gent_disagreement_chat/core/vector_search.py`

**Benefits of Two-Query Approach:**
- **Efficiency**: Only runs fallback query when necessary
- **Database Optimization**: Leverages database indexing for threshold filtering
- **Guaranteed Results**: Always returns at least `min_docs` results
- **Clean Logic**: Clear separation between threshold filtering and fallback behavior

```python
def find_relevant_above_adaptive_threshold(self, query, min_docs=3, max_docs=10, similarity_threshold=0.6):
    """
    Clean two-query approach: try threshold first, fallback if needed
    
    Args:
        query: Search query
        min_docs: Minimum number of documents to return
        max_docs: Maximum number of documents to return  
        similarity_threshold: Minimum similarity score for inclusion
    
    Returns:
        List of relevant documents above threshold
    """
    try:
        embedding = self.embedding_service.generate_embedding(query)
        
        # Query 1: Try to get results above threshold
        threshold_sql = """
            SELECT
                ts.speaker,
                ts.text,
                e.episode_number,
                e.title,
                e.date_published,
                1 - (ts.embedding <=> %s::vector) as similarity
            FROM transcript_segments ts
            JOIN episodes e ON ts.episode_id = e.id
            WHERE 1 - (ts.embedding <=> %s::vector) >= %s
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        results = self.db_manager.execute_query(
            threshold_sql, (embedding, embedding, similarity_threshold, max_docs)
        )
        
        # If we have enough results above threshold, return them
        if len(results) >= min_docs:
            return results
        
        # Query 2: Fallback to top N results regardless of threshold
        fallback_sql = """
            SELECT
                ts.speaker,
                ts.text,
                e.episode_number,
                e.title,
                e.date_published,
                1 - (ts.embedding <=> %s::vector) as similarity
            FROM transcript_segments ts
            JOIN episodes e ON ts.episode_id = e.id
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        fallback_results = self.db_manager.execute_query(
            fallback_sql, (embedding, min_docs)
        )
        
        return fallback_results
        
    except Exception as e:
        print(f"Error in clean adaptive threshold search: {e}")
        raise e

def find_similar_above_threshold(self, query, threshold=0.6, limit=10):
    """Simple threshold-based search for backward compatibility"""
    try:
        embedding = self.embedding_service.generate_embedding(query)
        
        query_sql = """
            SELECT
                ts.speaker,
                ts.text,
                e.episode_number,
                e.title,
                e.date_published,
                1 - (ts.embedding <=> %s::vector) as similarity
            FROM transcript_segments ts
            JOIN episodes e ON ts.episode_id = e.id
            WHERE 1 - (ts.embedding <=> %s::vector) >= %s
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        results = self.db_manager.execute_query(query_sql, (embedding, embedding, threshold, limit))
        return results
        
    except Exception as e:
        print(f"Error in threshold search: {e}")
        raise e
```

<mark> STOPPED HERE

#### **Step 1.2: Update RAGService Configuration**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
class RAGService:
    # Updated search parameters
    DEFAULT_THRESHOLD = 0.6
    DEFAULT_MIN_DOCS = 3
    DEFAULT_MAX_DOCS = 10
    DEFAULT_LIMIT = 5  # Keep for backward compatibility
    
    def __init__(self, database_name="gent_disagreement", use_adaptive_threshold=True):
        self.vector_search = VectorSearch(database_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_adaptive_threshold = use_adaptive_threshold
```

#### **Step 1.3: Update ask_question Methods**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def ask_question(self, question, model="gpt-4o-mini", 
                 min_docs=None, max_docs=None, threshold=None):
    """Implement RAG with adaptive threshold filtering"""
    try:
        # 1. Find relevant transcript segments with adaptive threshold
        if self.use_adaptive_threshold:
            search_results = self.vector_search.find_relevant_above_adaptive_threshold(
                question, 
                min_docs=min_docs or self.DEFAULT_MIN_DOCS,
                max_docs=max_docs or self.DEFAULT_MAX_DOCS,
                similarity_threshold=threshold or self.DEFAULT_THRESHOLD
            )
        else:
            # Fallback to original method
            search_results = self.vector_search.find_most_similar(
                question, limit=self.DEFAULT_LIMIT
            )
        
        # 2. Format context from search results
        formatted_results = self._format_search_results(search_results)
        
        # 3. Create prompt with context
        prompt = self._create_prompt(formatted_results, question)
        
        # 4. Generate response
        return self._generate_streaming_response(prompt, model)
        
    except Exception as e:
        print(f"Error in RAG service: {e}")
        raise e
```

### **Testing Strategy**
- Create unit tests for threshold filtering logic
- Test edge cases (no results above threshold, very few results)
- Validate fallback behavior when min_docs not met

---

## **Phase 2: Query Enhancement** ��

### **Implementation Steps**

#### **Step 2.1: Create Query Enhancement Service**
**File**: `server/src/gent_disagreement_chat/core/query_enhancer.py`

```python
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
            'hosts': ['Ricky Ghoshroy', 'Ricky', 'Brendan Kelly', 'Brendan', 'hosts', 'speakers', 'panelists'],
            'legal_topics': ['Supreme Court', 'SCOTUS', 'constitutional', 'originalism', 'textualism', 'court cases', 'judicial', 'legal reasoning'],
            'economic_topics': ['tariffs', 'trade', 'Bureau of Labor Statistics', 'BLS', 'Federal Reserve', 'Fed', 'markets', 'inflation', 'labor', 'economy'],
            'political_topics': ['authoritarianism', 'democracy', 'soft power', 'foreign policy', 'Trump administration', 'conservative', 'liberal'],
            'guest_experts': ['Professor', 'expert', 'analyst', 'journalist', 'academic'],
            'general_concepts': ['politics', 'economics', 'culture', 'philosophy', 'governance', 'policy']
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
            'original': question,
            'hyde': self._generate_hypothetical_answer(question),
            'expanded': self._expand_query_keywords(question),
            'intent': self._classify_query_intent(question),
            'cleaned': self._clean_and_normalize(question)
        }

    def _generate_hypothetical_answer(self, question: str) -> str:
        """Generate hypothetical answer for HyDE approach"""
        try:
            # Classify the query intent to tailor the HyDE prompt
            intent = self._classify_query_intent(question)

            base_prompt = f'Based on the question "{question}", generate a hypothetical answer that might appear in the "A Gentleman\'s Disagreement" podcast transcript.'

            if intent == 'legal_analysis':
                specific_prompt = "Focus on constitutional law, Supreme Court cases, legal reasoning, and judicial interpretation. Include analytical discussion typical of legal experts."
            elif intent == 'economic_policy':
                specific_prompt = "Focus on economic analysis, policy implications, market trends, and data-driven discussion typical of economic experts."
            elif intent == 'historical_comparison':
                specific_prompt = "Focus on historical parallels, past precedents, and comparative analysis across different eras."
            elif intent == 'expert_opinion':
                specific_prompt = "Focus on academic expertise, scholarly analysis, and expert commentary from professors or journalists."
            else:
                specific_prompt = "Focus on thoughtful political commentary, nuanced debate, and analytical discussion between the hosts."

            prompt = f"{base_prompt} {specific_prompt} Keep it concise (1-2 sentences) and match the intellectual, conversational tone of the podcast."

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
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
        for category, terms in self.podcast_vocabulary.items():
            if any(term in question_lower for term in terms):
                expanded_terms.extend([t for t in terms if t not in question_lower])

        # Add common synonyms
        expanded_terms.extend(self._get_synonyms(question))

        return ' '.join(list(set(expanded_terms)))  # Remove duplicates

    def _classify_query_intent(self, question: str) -> str:
        """Classify the intent of the query based on podcast content patterns"""
        question_lower = question.lower()

        # Legal analysis queries
        if any(word in question_lower for word in ['court', 'constitutional', 'legal', 'ruling', 'case', 'justice', 'judge']):
            return 'legal_analysis'
        # Economic policy queries
        elif any(word in question_lower for word in ['tariff', 'trade', 'economy', 'market', 'fed', 'inflation', 'labor']):
            return 'economic_policy'
        # Historical comparison queries
        elif any(word in question_lower for word in ['history', 'past', 'compare', 'historical', 'era', 'previous']):
            return 'historical_comparison'
        # Guest expert queries
        elif any(word in question_lower for word in ['professor', 'expert', 'guest', 'interview']):
            return 'expert_opinion'
        # Factual queries
        elif any(word in question_lower for word in ['what', 'who', 'when', 'where', 'which']):
            return 'factual'
        # Analytical queries
        elif any(word in question_lower for word in ['how', 'why', 'explain', 'analyze']):
            return 'analytical'
        # Comparative queries
        elif any(word in question_lower for word in ['compare', 'versus', 'difference', 'contrast']):
            return 'comparative'
        # Opinion queries
        elif any(word in question_lower for word in ['opinion', 'think', 'believe', 'feel', 'view']):
            return 'opinion'
        else:
            return 'general'

    def _clean_and_normalize(self, question: str) -> str:
        """Clean and normalize the question"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', question.strip())

        # Remove common question words that don't add semantic value
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
        words = cleaned.split()
        cleaned_words = [w for w in words if w.lower() not in stop_words]

        return ' '.join(cleaned_words)

    def _get_synonyms(self, question: str) -> List[str]:
        """Get synonyms for key terms in the question"""
        # Enhanced synonym mapping based on podcast analysis
        synonym_map = {
            'discuss': ['talk about', 'cover', 'address', 'explore', 'analyze', 'examine'],
            'opinion': ['view', 'perspective', 'stance', 'position', 'take', 'thoughts'],
            'topic': ['subject', 'theme', 'issue', 'matter', 'question'],
            'hosts': ['speakers', 'panelists', 'Ricky', 'Brendan', 'Ricky Ghoshroy', 'Brendan Kelly'],
            'court': ['Supreme Court', 'SCOTUS', 'judicial', 'tribunal'],
            'economy': ['economic', 'markets', 'trade', 'financial', 'fiscal'],
            'government': ['administration', 'federal', 'state', 'political'],
            'policy': ['regulation', 'rule', 'law', 'governance'],
            'conservative': ['right-wing', 'Republican', 'traditional'],
            'liberal': ['left-wing', 'Democratic', 'progressive']
        }

        synonyms = []
        question_lower = question.lower()

        for term, synonym_list in synonym_map.items():
            if term in question_lower:
                synonyms.extend(synonym_list)

        return synonyms
```

#### **Step 2.2: Integrate Query Enhancement into RAGService**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
from .query_enhancer import QueryEnhancer

class RAGService:
    def __init__(self, database_name="gent_disagreement", use_adaptive_threshold=True,
                 use_query_enhancement=True):
        self.vector_search = VectorSearch(database_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_query_enhancement = use_query_enhancement

        if self.use_query_enhancement:
            self.query_enhancer = QueryEnhancer()

    def ask_question_enhanced(self, question, model="gpt-4o-mini"):
        """Enhanced RAG with query preprocessing"""
        try:
            # 1. Enhance query if enabled
            if self.use_query_enhancement:
                enhanced_query = self.query_enhancer.enhance_query(question)

                # Use multiple query variations for better retrieval
                search_results = self._multi_query_search(enhanced_query)
            else:
                search_results = self.vector_search.find_most_similar(question, limit=10)

            # 2. Format context from search results
            formatted_results = self._format_search_results(search_results)

            # 3. Create prompt with context
            prompt = self._create_prompt(formatted_results, question)

            # 4. Generate response
            return self._generate_streaming_response(prompt, model)

        except Exception as e:
            print(f"Error in enhanced RAG service: {e}")
            raise e

    def _multi_query_search(self, enhanced_query: Dict[str, Any]) -> List[Dict]:
        """Search using multiple query variations and combine results"""
        all_results = []

        # Search with original query
        original_results = self.vector_search.find_most_similar(enhanced_query['original'], limit=5)
        all_results.extend(original_results)

        # Search with HyDE answer
        if enhanced_query['hyde']:
            hyde_results = self.vector_search.find_most_similar(enhanced_query['hyde'], limit=3)
            all_results.extend(hyde_results)

        # Search with expanded query
        if enhanced_query['expanded']:
            expanded_results = self.vector_search.find_most_similar(enhanced_query['expanded'], limit=3)
            all_results.extend(expanded_results)

        # Deduplicate and rerank
        return self._deduplicate_and_rerank(all_results)

    def _deduplicate_and_rerank(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates and rerank results by similarity"""
        # Create unique key for each result
        seen = set()
        unique_results = []

        for result in results:
            key = f"{result['speaker']}_{result['episode_number']}_{hash(result['text'])}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # Sort by similarity score
        return sorted(unique_results, key=lambda x: x['similarity'], reverse=True)[:10]

    def _enhance_speaker_context(self, results: List[Dict], enhanced_query: Dict[str, Any]) -> List[Dict]:
        """Add speaker-aware context for better attribution"""
        for result in results:
            speaker = result['speaker']

            # Add speaker role identification
            if speaker in ['Ricky Ghoshroy', 'Ricky']:
                result['speaker_role'] = 'co-host'
                result['speaker_full_name'] = 'Ricky Ghoshroy'
            elif speaker in ['Brendan Kelly', 'Brendan']:
                result['speaker_role'] = 'co-host'
                result['speaker_full_name'] = 'Brendan Kelly'
            elif 'Professor' in speaker or 'Dr.' in speaker:
                result['speaker_role'] = 'academic_expert'
                result['speaker_full_name'] = speaker
            elif any(title in speaker for title in ['journalist', 'reporter', 'writer']):
                result['speaker_role'] = 'media_expert'
                result['speaker_full_name'] = speaker
            else:
                result['speaker_role'] = 'guest'
                result['speaker_full_name'] = speaker

            # Add intent-based relevance scoring
            query_intent = enhanced_query.get('intent', 'general')
            if query_intent == 'expert_opinion' and result['speaker_role'] in ['academic_expert', 'media_expert']:
                result['relevance_boost'] = 0.1
            elif query_intent == 'legal_analysis' and 'Professor' in speaker and 'Law' in result.get('text', ''):
                result['relevance_boost'] = 0.15
            else:
                result['relevance_boost'] = 0.0

            # Boost similarity score based on speaker relevance
            result['similarity'] += result['relevance_boost']

        return results
```

#### **Step 2.3: Add Speaker-Aware Query Processing**
**File**: `server/src/gent_disagreement_chat/core/query_enhancer.py`

```python
    def enhance_query_with_speaker_context(self, question: str) -> Dict[str, Any]:
        """Enhanced query processing with speaker-aware context"""
        base_enhancement = self.enhance_query(question)

        # Detect speaker-specific queries
        question_lower = question.lower()
        speaker_context = {}

        if any(name in question_lower for name in ['ricky', 'ricky ghoshroy']):
            speaker_context['target_speaker'] = 'Ricky Ghoshroy'
            speaker_context['speaker_filter'] = True
        elif any(name in question_lower for name in ['brendan', 'brendan kelly']):
            speaker_context['target_speaker'] = 'Brendan Kelly'
            speaker_context['speaker_filter'] = True
        elif any(term in question_lower for term in ['professor', 'expert', 'guest']):
            speaker_context['target_speaker'] = 'expert'
            speaker_context['speaker_filter'] = True
        else:
            speaker_context['speaker_filter'] = False

        base_enhancement['speaker_context'] = speaker_context
        return base_enhancement
```

### **Testing Strategy**
- Test query enhancement with various question types (legal, economic, political, historical)
- Validate HyDE generation quality for each intent category
- Test multi-query search effectiveness with podcast-specific terminology
- Test speaker-aware processing and attribution accuracy
- Measure retrieval improvement with enhanced queries vs. baseline
- Validate correct host name recognition (Ricky Ghoshroy, Brendan Kelly)
- Test expert identification and relevance boosting for academic/media guests

### **Key Improvements Based on Transcript Analysis**

#### **Updated Content Understanding**
- **Correct Host Identification**: Ricky Ghoshroy and Brendan Kelly (not Tyler and Ben)
- **Domain-Specific Vocabulary**: Added legal, economic, and political terminology from actual discussions
- **Guest Expert Recognition**: Improved identification of professors, journalists, and academic experts

#### **Enhanced Query Processing**
- **Legal Analysis Queries**: Constitutional law, Supreme Court cases, judicial reasoning
- **Economic Policy Queries**: Tariffs, trade policy, Federal Reserve, labor statistics
- **Historical Comparison Queries**: Past precedents, era comparisons, historical context
- **Expert Opinion Queries**: Academic and journalistic expertise identification

#### **Improved Relevance Scoring**
- **Speaker Role Awareness**: Distinguish between hosts, experts, and guests
- **Intent-Based Boosting**: Higher relevance for expert opinions on specialized topics
- **Attribution Context**: Better tracking of who said what for accurate responses

---

## **Phase 3: Enhanced Context Formatting** 📝

### **Implementation Steps**

#### **Step 3.1: Enhanced Context Formatting**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def _format_search_results_enhanced(self, search_results):
    """Enhanced context formatting with hierarchical organization"""
    
    if not search_results:
        return "No relevant transcript segments found."
    
    # Group by episode
    grouped_results = self._group_by_episode(search_results)
    
    formatted_result = ""
    for episode_group in grouped_results:
        formatted_result += f"## Episode {episode_group['episode']}: {episode_group['title']}\n"
        formatted_result += f"**Relevance**: {episode_group['avg_similarity']:.2f} (Published: {episode_group['date_published']})\n\n"
        
        for result in episode_group['segments']:
            # Add relevance indicator
            relevance_indicator = self._get_relevance_indicator(result['similarity'])
            formatted_result += f"**{relevance_indicator} {result['speaker']}**: {result['text']}\n"
            formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"
    
    return formatted_result

def _group_by_episode(self, search_results):
    """Group search results by episode for better organization"""
    episodes = {}
    
    for result in search_results:
        episode_num = result['episode_number']
        
        if episode_num not in episodes:
            episodes[episode_num] = {
                'episode': episode_num,
                'title': result['title'],
                'date_published': result['date_published'],
                'segments': [],
                'similarities': []
            }
        
        episodes[episode_num]['segments'].append(result)
        episodes[episode_num]['similarities'].append(result['similarity'])
    
    # Calculate average similarity for each episode
    for episode_data in episodes.values():
        episode_data['avg_similarity'] = sum(episode_data['similarities']) / len(episode_data['similarities'])
    
    # Sort episodes by average relevance
    return sorted(episodes.values(), key=lambda x: x['avg_similarity'], reverse=True)

def _get_relevance_indicator(self, similarity):
    """Get visual indicator for relevance level"""
    if similarity >= 0.8:
        return "🎯"  # High relevance
    elif similarity >= 0.6:
        return "⭐"  # Medium relevance
    else:
        return "📝"  # Lower relevance
```

#### **Step 3.2: Speaker-Aware Context Formatting**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def _format_search_results_speaker_aware(self, search_results, enhanced_query=None):
    """Enhanced formatting with speaker role awareness based on transcript analysis"""

    if not search_results:
        return "No relevant transcript segments found."

    # Enhance results with speaker context if available
    if enhanced_query and enhanced_query.get('speaker_context'):
        search_results = self._enhance_speaker_context(search_results, enhanced_query)

    # Group by episode and speaker roles
    grouped_results = self._group_by_episode_and_speakers(search_results)

    formatted_result = ""
    for episode_group in grouped_results:
        formatted_result += f"## Episode {episode_group['episode']}: {episode_group['title']}\n"
        formatted_result += f"**Relevance**: {episode_group['avg_similarity']:.2f}\n\n"

        # Separate expert opinions from host discussions
        if episode_group.get('expert_segments'):
            formatted_result += "### Expert Analysis:\n"
            for result in episode_group['expert_segments']:
                relevance_indicator = self._get_relevance_indicator(result['similarity'])
                formatted_result += f"**{relevance_indicator} 👨‍🎓 {result['speaker_full_name']}**: {result['text']}\n"
                if result.get('speaker_role') == 'academic_expert':
                    formatted_result += f"*{result.get('speaker_credentials', 'Academic Expert')}*\n"
                formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"

        if episode_group.get('host_segments'):
            formatted_result += "### Host Discussion:\n"
            for result in episode_group['host_segments']:
                relevance_indicator = self._get_relevance_indicator(result['similarity'])
                formatted_result += f"**{relevance_indicator} 🎙️ {result['speaker_full_name']}**: {result['text']}\n"
                formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"

        if episode_group.get('other_segments'):
            formatted_result += "### Additional Context:\n"
            for result in episode_group['other_segments']:
                relevance_indicator = self._get_relevance_indicator(result['similarity'])
                formatted_result += f"**{relevance_indicator} {result['speaker_full_name']}**: {result['text']}\n"
                formatted_result += f"*Similarity: {result['similarity']:.2f}*\n\n"

    return formatted_result

def _group_by_episode_and_speakers(self, search_results):
    """Group search results by episode and categorize by speaker roles"""
    episodes = {}

    for result in search_results:
        episode_num = result['episode_number']

        if episode_num not in episodes:
            episodes[episode_num] = {
                'episode': episode_num,
                'title': result['title'],
                'date_published': result.get('date_published', 'Unknown'),
                'expert_segments': [],
                'host_segments': [],
                'other_segments': [],
                'similarities': []
            }

        # Categorize by speaker role
        speaker_role = result.get('speaker_role', 'unknown')
        speaker = result['speaker']

        # Add full name and credentials
        if speaker in ['Ricky Ghoshroy', 'Ricky']:
            result['speaker_full_name'] = 'Ricky Ghoshroy'
            result['speaker_role'] = 'co-host'
        elif speaker in ['Brendan Kelly', 'Brendan']:
            result['speaker_full_name'] = 'Brendan Kelly'
            result['speaker_role'] = 'co-host'
        elif 'Professor' in speaker:
            result['speaker_full_name'] = speaker
            result['speaker_role'] = 'academic_expert'
            result['speaker_credentials'] = f"Professor ({speaker.split()[-1]} expertise)"
        elif any(title in speaker.lower() for title in ['journalist', 'reporter', 'writer']):
            result['speaker_full_name'] = speaker
            result['speaker_role'] = 'media_expert'
        else:
            result['speaker_full_name'] = speaker
            result['speaker_role'] = 'guest'

        # Add to appropriate category
        if result['speaker_role'] in ['academic_expert', 'media_expert']:
            episodes[episode_num]['expert_segments'].append(result)
        elif result['speaker_role'] == 'co-host':
            episodes[episode_num]['host_segments'].append(result)
        else:
            episodes[episode_num]['other_segments'].append(result)

        episodes[episode_num]['similarities'].append(result['similarity'])

    # Calculate average similarity for each episode
    for episode_data in episodes.values():
        if episode_data['similarities']:
            episode_data['avg_similarity'] = sum(episode_data['similarities']) / len(episode_data['similarities'])
        else:
            episode_data['avg_similarity'] = 0.0

    # Sort episodes by average relevance
    return sorted(episodes.values(), key=lambda x: x['avg_similarity'], reverse=True)
```

#### **Step 3.3: Topic-Specific Formatting**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def _format_by_topic_intent(self, search_results, query_intent):
    """Format results based on query intent from transcript analysis"""

    if query_intent == 'legal_analysis':
        return self._format_legal_analysis_results(search_results)
    elif query_intent == 'economic_policy':
        return self._format_economic_policy_results(search_results)
    elif query_intent == 'expert_opinion':
        return self._format_expert_opinion_results(search_results)
    elif query_intent == 'historical_comparison':
        return self._format_historical_comparison_results(search_results)
    else:
        return self._format_search_results_speaker_aware(search_results)

def _format_legal_analysis_results(self, search_results):
    """Format results for legal analysis queries"""
    if not search_results:
        return "No relevant legal analysis found."

    formatted_result = "# Legal Analysis from A Gentleman's Disagreement\n\n"

    # Prioritize legal experts and relevant discussions
    legal_segments = []
    host_legal_segments = []

    for result in search_results:
        text_lower = result['text'].lower()
        is_legal_content = any(term in text_lower for term in
            ['court', 'constitutional', 'legal', 'justice', 'case', 'ruling', 'originalism', 'textualism'])

        if 'Professor' in result['speaker'] and is_legal_content:
            legal_segments.append(result)
        elif result['speaker'] in ['Ricky Ghoshroy', 'Brendan Kelly', 'Ricky', 'Brendan'] and is_legal_content:
            host_legal_segments.append(result)

    if legal_segments:
        formatted_result += "## Expert Legal Analysis:\n"
        for result in legal_segments:
            formatted_result += f"**⚖️ {result['speaker']}**: {result['text']}\n\n"

    if host_legal_segments:
        formatted_result += "## Host Discussion on Legal Issues:\n"
        for result in host_legal_segments:
            formatted_result += f"**🎙️ {result['speaker']}**: {result['text']}\n\n"

    return formatted_result

def _format_economic_policy_results(self, search_results):
    """Format results for economic policy queries"""
    if not search_results:
        return "No relevant economic analysis found."

    formatted_result = "# Economic Policy Analysis from A Gentleman's Disagreement\n\n"

    # Categorize economic content
    expert_analysis = []
    host_discussion = []

    for result in search_results:
        text_lower = result['text'].lower()
        is_economic_content = any(term in text_lower for term in
            ['tariff', 'trade', 'economy', 'market', 'fed', 'inflation', 'labor', 'economic'])

        if is_economic_content:
            if any(title in result['speaker'].lower() for title in ['professor', 'journalist', 'reporter']):
                expert_analysis.append(result)
            else:
                host_discussion.append(result)

    if expert_analysis:
        formatted_result += "## Expert Economic Analysis:\n"
        for result in expert_analysis:
            formatted_result += f"**📊 {result['speaker']}**: {result['text']}\n\n"

    if host_discussion:
        formatted_result += "## Host Economic Discussion:\n"
        for result in host_discussion:
            formatted_result += f"**🎙️ {result['speaker']}**: {result['text']}\n\n"

    return formatted_result

def _format_expert_opinion_results(self, search_results):
    """Format results specifically highlighting expert opinions"""
    if not search_results:
        return "No expert opinions found."

    formatted_result = "# Expert Opinions from A Gentleman's Disagreement\n\n"

    # Group by expert type
    academic_experts = []
    media_experts = []

    for result in search_results:
        if 'Professor' in result['speaker'] or 'Dr.' in result['speaker']:
            academic_experts.append(result)
        elif any(title in result['speaker'].lower() for title in ['journalist', 'reporter', 'writer']):
            media_experts.append(result)

    if academic_experts:
        formatted_result += "## Academic Expert Analysis:\n"
        for result in academic_experts:
            formatted_result += f"**👨‍🎓 {result['speaker']}**: {result['text']}\n\n"

    if media_experts:
        formatted_result += "## Media Expert Analysis:\n"
        for result in media_experts:
            formatted_result += f"**📰 {result['speaker']}**: {result['text']}\n\n"

    return formatted_result

def _format_historical_comparison_results(self, search_results):
    """Format results for historical comparison queries"""
    if not search_results:
        return "No historical comparisons found."

    formatted_result = "# Historical Analysis from A Gentleman's Disagreement\n\n"

    # Look for historical references
    historical_segments = []

    for result in search_results:
        text_lower = result['text'].lower()
        has_historical_content = any(term in text_lower for term in
            ['history', 'historical', 'past', 'era', 'precedent', 'previous', 'decades', 'years ago'])

        if has_historical_content:
            historical_segments.append(result)

    if historical_segments:
        formatted_result += "## Historical Context and Comparisons:\n"
        for result in historical_segments:
            formatted_result += f"**📚 {result['speaker']}**: {result['text']}\n\n"

    return formatted_result
```

#### **Step 3.4: Context Windows and Smart Filtering**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def _format_search_results_with_context(self, search_results, query_filters=None):
    """Add context windows and basic filtering"""
    
    if not search_results:
        return "No relevant transcript segments found."
    
    # Extract basic filters from query
    filters = self._extract_basic_filters(query_filters) if query_filters else {}
    
    # Group and format with context windows
    grouped_results = self._group_by_episode_with_context(search_results)
    
    formatted_result = ""
    for episode_group in grouped_results:
        # Apply episode filter if specified
        if filters.get('episode_number') and episode_group['episode'] != filters['episode_number']:
            continue
        
        formatted_result += f"## Episode {episode_group['episode']}: {episode_group['title']}\n"
        formatted_result += f"**Relevance**: {episode_group['avg_similarity']:.2f}\n\n"
        
        for result in episode_group['segments']:
            # Apply speaker filter if specified
            speaker_match = not filters.get('speaker') or result['speaker'].lower() == filters['speaker'].lower()
            
            if speaker_match:
                formatted_result += f"**🎯 {result['speaker']}**: {result['text']}\n"
            else:
                formatted_result += f"**{result['speaker']}**: {result['text']}\n"
            
            # Add context windows if available
            if result.get('context_before') and result.get('context_after'):
                formatted_result += f"*Context: {result['context_before']} [...] {result['context_after']}*\n\n"
    
    return formatted_result

def _extract_basic_filters(self, query):
    """Extract basic filters from query using regex"""
    filters = {}
    
    if not query:
        return filters
    
    # Extract episode number
    episode_match = re.search(r'episode\s+(\d+)', query, re.IGNORECASE)
    if episode_match:
        filters['episode_number'] = int(episode_match.group(1))
    
    # Extract speaker name - corrected host names
    speaker_match = re.search(r'(Ricky|Brendan|Ricky Ghoshroy|Brendan Kelly)', query, re.IGNORECASE)
    if speaker_match:
        filters['speaker'] = speaker_match.group(1)
    
    return filters
```

#### **Step 3.3: Update Main Methods to Use Enhanced Formatting**
**File**: `server/src/gent_disagreement_chat/core/rag_service.py`

```python
def ask_question(self, question, model="gpt-4o-mini", 
                 min_docs=None, max_docs=None, threshold=None, 
                 use_enhanced_formatting=True):
    """Enhanced RAG with improved context formatting"""
    try:
        # 1. Find relevant transcript segments
        if self.use_adaptive_threshold:
            search_results = self.vector_search.find_relevant_above_adaptive_threshold(
                question, 
                min_docs=min_docs or self.DEFAULT_MIN_DOCS,
                max_docs=max_docs or self.DEFAULT_MAX_DOCS,
                similarity_threshold=threshold or self.DEFAULT_THRESHOLD
            )
        else:
            search_results = self.vector_search.find_most_similar(question, limit=10)
        
        # 2. Format context with enhanced formatting
        if use_enhanced_formatting:
            formatted_results = self._format_search_results_enhanced(search_results)
        else:
            formatted_results = self._format_search_results(search_results)
        
        # 3. Create prompt with context
        prompt = self._create_prompt(formatted_results, question)
        
        # 4. Generate response
        return self._generate_streaming_response(prompt, model)
        
    except Exception as e:
        print(f"Error in RAG service: {e}")
        raise e
```

### **Testing Strategy**
- Test enhanced formatting with various result sets
- Validate episode grouping logic
- Test filtering functionality
- Compare readability of enhanced vs. original formatting

---

## **Phase 4: Integration and Evaluation** ��

### **Step 4.1: Create Enhanced RAGService Factory**
**File**: `server/src/gent_disagreement_chat/core/enhanced_rag_service.py`

```python
from .rag_service import RAGService
from .vector_search import VectorSearch
from .query_enhancer import QueryEnhancer

class EnhancedRAGService(RAGService):
    """
    Enhanced RAG service with all optimizations enabled
    """
    
    def __init__(self, database_name="gent_disagreement", 
                 enable_threshold_filtering=True,
                 enable_query_enhancement=True,
                 enable_enhanced_formatting=True):
        
        super().__init__(database_name, use_adaptive_threshold=True, use_query_enhancement=True)
        
        self.enable_threshold_filtering = enable_threshold_filtering
        self.enable_query_enhancement = enable_query_enhancement
        self.enable_enhanced_formatting = enable_enhanced_formatting
        
        # Initialize query enhancer
        if self.enable_query_enhancement:
            self.query_enhancer = QueryEnhancer()
    
    def ask_question_optimized(self, question, model="gpt-4o-mini", **kwargs):
        """
        Fully optimized RAG with all enhancements
        """
        try:
            # Step 1: Query Enhancement
            if self.enable_query_enhancement:
                enhanced_query = self.query_enhancer.enhance_query(question)
                search_results = self._multi_query_search_optimized(enhanced_query)
            else:
                search_results = self.vector_search.find_most_similar(question, limit=10)
            
            # Step 2: Enhanced Formatting
            if self.enable_enhanced_formatting:
                formatted_results = self._format_search_results_enhanced(search_results)
            else:
                formatted_results = self._format_search_results(search_results)
            
            # Step 3: Generate Response
            prompt = self._create_prompt(formatted_results, question)
            return self._generate_streaming_response(prompt, model)
            
        except Exception as e:
            print(f"Error in optimized RAG service: {e}")
            raise e
```

### **Step 4.2: Update Evaluation Framework**
**File**: `server/src/gent_disagreement_chat/evaluation/optimization_evaluator.py`

```python
from .precision_tracker import PrecisionTracker
from ..core.rag_service import RAGService
from ..core.enhanced_rag_service import EnhancedRAGService

class OptimizationEvaluator:
    """Evaluates the impact of RAG optimizations"""
    
    def __init__(self, database_name="gent_disagreement"):
        self.precision_tracker = PrecisionTracker()
        self.database_name = database_name
    
    def evaluate_optimizations(self, test_questions=None):
        """Comprehensive evaluation of all optimizations"""
        
        if test_questions is None:
            test_questions = self.precision_tracker.get_default_test_questions()
        
        # Create baseline
        baseline_service = RAGService(self.database_name, use_adaptive_threshold=False, use_query_enhancement=False)
        baseline_run_id = self.precision_tracker.create_baseline_evaluation(baseline_service, test_questions)
        
        # Test individual optimizations
        results = {}
        
        # Test threshold filtering only
        threshold_service = RAGService(self.database_name, use_adaptive_threshold=True, use_query_enhancement=False)
        results['threshold_only'] = self.precision_tracker.evaluate_improved_system(threshold_service, baseline_run_id)
        
        # Test query enhancement only
        query_service = RAGService(self.database_name, use_adaptive_threshold=False, use_query_enhancement=True)
        results['query_only'] = self.precision_tracker.evaluate_improved_system(query_service, baseline_run_id)
        
        # Test enhanced formatting only
        format_service = RAGService(self.database_name, use_adaptive_threshold=False, use_query_enhancement=False)
        results['format_only'] = self.precision_tracker.evaluate_improved_system(format_service, baseline_run_id)
        
        # Test all optimizations combined
        enhanced_service = EnhancedRAGService(self.database_name)
        results['all_optimizations'] = self.precision_tracker.evaluate_improved_system(enhanced_service, baseline_run_id)
        
        return {
            'baseline': baseline_run_id,
            'individual_tests': results,
            'test_questions': test_questions
        }
    
    def generate_optimization_report(self, evaluation_results):
        """Generate comprehensive optimization report"""
        
        report = {
            'evaluation_summary': evaluation_results,
            'recommendations': self._generate_recommendations(evaluation_results),
            'performance_metrics': self._calculate_performance_metrics(evaluation_results)
        }
        
        return report
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Analyze individual optimization impacts
        # Generate specific recommendations for deployment
        
        return recommendations
    
    def _calculate_performance_metrics(self, results):
        """Calculate performance improvement metrics"""
        # Calculate precision, recall, and other metrics
        # Compare baseline vs. optimized performance
        
        return {}

    def evaluate_speaker_attribution(self, test_results):
        """Evaluate accuracy of speaker identification and role assignment"""
        attribution_metrics = {
            'host_identification_accuracy': 0.0,
            'expert_identification_accuracy': 0.0,
            'speaker_role_accuracy': 0.0,
            'speaker_filtering_precision': 0.0
        }

        correct_host_ids = 0
        correct_expert_ids = 0
        correct_roles = 0
        total_speakers = 0

        for result in test_results:
            if 'speaker_results' in result:
                for speaker_result in result['speaker_results']:
                    total_speakers += 1

                    # Check host identification
                    if speaker_result['actual_speaker'] in ['Ricky Ghoshroy', 'Brendan Kelly']:
                        if speaker_result['identified_role'] == 'co-host':
                            correct_host_ids += 1

                    # Check expert identification
                    if 'Professor' in speaker_result['actual_speaker']:
                        if speaker_result['identified_role'] == 'academic_expert':
                            correct_expert_ids += 1

                    # Check overall role accuracy
                    if speaker_result['identified_role'] == speaker_result['expected_role']:
                        correct_roles += 1

        if total_speakers > 0:
            attribution_metrics['host_identification_accuracy'] = correct_host_ids / total_speakers
            attribution_metrics['expert_identification_accuracy'] = correct_expert_ids / total_speakers
            attribution_metrics['speaker_role_accuracy'] = correct_roles / total_speakers

        return attribution_metrics

    def get_podcast_specific_test_questions(self):
        """Generate test questions based on actual podcast content analysis"""
        return [
            # Legal analysis questions
            {
                'question': "What did Professor Bierman say about originalism and textualism?",
                'expected_speakers': ['Professor Jack Bierman'],
                'query_intent': 'legal_analysis',
                'expected_role': 'academic_expert'
            },
            {
                'question': "How does Brendan view Supreme Court decisions on constitutional issues?",
                'expected_speakers': ['Brendan Kelly', 'Brendan'],
                'query_intent': 'legal_analysis',
                'expected_role': 'co-host'
            },

            # Economic policy questions
            {
                'question': "What was discussed about tariffs and trade policy with Lydia DePhillis?",
                'expected_speakers': ['Lydia DePhillis'],
                'query_intent': 'economic_policy',
                'expected_role': 'media_expert'
            },
            {
                'question': "How does Ricky analyze economic policy and market trends?",
                'expected_speakers': ['Ricky Ghoshroy', 'Ricky'],
                'query_intent': 'economic_policy',
                'expected_role': 'co-host'
            },

            # Historical comparison questions
            {
                'question': "What historical parallels were drawn about authoritarianism?",
                'expected_speakers': ['Ricky Ghoshroy', 'Brendan Kelly', 'Professor Jack Bierman'],
                'query_intent': 'historical_comparison',
                'expected_role': 'mixed'
            },

            # Speaker-specific questions
            {
                'question': "What are Ricky's views on foreign policy and soft power?",
                'expected_speakers': ['Ricky Ghoshroy', 'Ricky'],
                'query_intent': 'opinion',
                'expected_role': 'co-host'
            },
            {
                'question': "How does Brendan analyze legal cases and judicial reasoning?",
                'expected_speakers': ['Brendan Kelly', 'Brendan'],
                'query_intent': 'legal_analysis',
                'expected_role': 'co-host'
            },

            # Expert opinion questions
            {
                'question': "What expert analysis was provided about economic data and Bureau of Labor Statistics?",
                'expected_speakers': ['Lydia DePhillis'],
                'query_intent': 'expert_opinion',
                'expected_role': 'media_expert'
            },
            {
                'question': "What did constitutional law experts say about Supreme Court term analysis?",
                'expected_speakers': ['Professor Jack Bierman'],
                'query_intent': 'expert_opinion',
                'expected_role': 'academic_expert'
            }
        ]
```

#### **Step 4.2.1: Speaker Attribution Evaluator**
**File**: `server/src/gent_disagreement_chat/evaluation/speaker_attribution_evaluator.py`

```python
from typing import Dict, List, Any
import re

class SpeakerAttributionEvaluator:
    """Evaluate accuracy of speaker identification and attribution in enhanced RAG"""

    def __init__(self):
        self.known_hosts = ['Ricky Ghoshroy', 'Ricky', 'Brendan Kelly', 'Brendan']
        self.expert_indicators = ['Professor', 'Dr.', 'journalist', 'reporter', 'writer']

    def evaluate_speaker_accuracy(self, search_results: List[Dict], expected_speakers: List[str]) -> Dict[str, float]:
        """
        Evaluate speaker identification accuracy

        Args:
            search_results: Results from enhanced RAG with speaker information
            expected_speakers: List of expected speakers for the query

        Returns:
            Dictionary with accuracy metrics
        """
        metrics = {
            'speaker_identification_accuracy': 0.0,
            'speaker_role_accuracy': 0.0,
            'host_recognition_accuracy': 0.0,
            'expert_recognition_accuracy': 0.0,
            'speaker_filtering_precision': 0.0
        }

        if not search_results:
            return metrics

        total_results = len(search_results)
        correct_speaker_ids = 0
        correct_roles = 0
        correct_host_recognition = 0
        correct_expert_recognition = 0
        relevant_speakers = 0

        for result in search_results:
            speaker = result.get('speaker', '')
            speaker_full_name = result.get('speaker_full_name', speaker)
            speaker_role = result.get('speaker_role', 'unknown')

            # Check speaker identification accuracy
            if speaker_full_name in expected_speakers or speaker in expected_speakers:
                correct_speaker_ids += 1
                relevant_speakers += 1

            # Check role assignment accuracy
            expected_role = self._get_expected_role(speaker_full_name)
            if speaker_role == expected_role:
                correct_roles += 1

            # Check host recognition
            if speaker_full_name in self.known_hosts:
                if speaker_role == 'co-host':
                    correct_host_recognition += 1

            # Check expert recognition
            if any(indicator in speaker_full_name for indicator in self.expert_indicators):
                if speaker_role in ['academic_expert', 'media_expert']:
                    correct_expert_recognition += 1

        # Calculate metrics
        metrics['speaker_identification_accuracy'] = correct_speaker_ids / total_results if total_results > 0 else 0.0
        metrics['speaker_role_accuracy'] = correct_roles / total_results if total_results > 0 else 0.0
        metrics['host_recognition_accuracy'] = correct_host_recognition / total_results if total_results > 0 else 0.0
        metrics['expert_recognition_accuracy'] = correct_expert_recognition / total_results if total_results > 0 else 0.0
        metrics['speaker_filtering_precision'] = relevant_speakers / total_results if total_results > 0 else 0.0

        return metrics

    def evaluate_topic_formatting_accuracy(self, formatted_results: str, query_intent: str) -> Dict[str, float]:
        """
        Evaluate accuracy of topic-specific formatting

        Args:
            formatted_results: Formatted search results string
            query_intent: Expected query intent category

        Returns:
            Dictionary with formatting accuracy metrics
        """
        metrics = {
            'topic_header_accuracy': 0.0,
            'expert_section_presence': 0.0,
            'host_section_presence': 0.0,
            'appropriate_formatting': 0.0
        }

        # Check for appropriate headers based on intent
        if query_intent == 'legal_analysis':
            if 'Legal Analysis' in formatted_results or '⚖️' in formatted_results:
                metrics['topic_header_accuracy'] = 1.0
        elif query_intent == 'economic_policy':
            if 'Economic Policy' in formatted_results or '📊' in formatted_results:
                metrics['topic_header_accuracy'] = 1.0
        elif query_intent == 'expert_opinion':
            if 'Expert' in formatted_results and ('👨‍🎓' in formatted_results or '📰' in formatted_results):
                metrics['topic_header_accuracy'] = 1.0

        # Check for expert and host sections
        if 'Expert' in formatted_results:
            metrics['expert_section_presence'] = 1.0
        if 'Host' in formatted_results:
            metrics['host_section_presence'] = 1.0

        # Overall formatting appropriateness
        if metrics['topic_header_accuracy'] > 0 and (metrics['expert_section_presence'] > 0 or metrics['host_section_presence'] > 0):
            metrics['appropriate_formatting'] = 1.0

        return metrics

    def _get_expected_role(self, speaker_name: str) -> str:
        """Get expected role for a given speaker"""
        if speaker_name in self.known_hosts:
            return 'co-host'
        elif 'Professor' in speaker_name or 'Dr.' in speaker_name:
            return 'academic_expert'
        elif any(title in speaker_name.lower() for title in ['journalist', 'reporter', 'writer']):
            return 'media_expert'
        else:
            return 'guest'

    def generate_speaker_attribution_report(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive speaker attribution report"""

        report = {
            'overall_accuracy': evaluation_results.get('speaker_identification_accuracy', 0.0),
            'role_assignment_quality': evaluation_results.get('speaker_role_accuracy', 0.0),
            'host_vs_expert_distinction': {
                'host_accuracy': evaluation_results.get('host_recognition_accuracy', 0.0),
                'expert_accuracy': evaluation_results.get('expert_recognition_accuracy', 0.0)
            },
            'filtering_precision': evaluation_results.get('speaker_filtering_precision', 0.0),
            'recommendations': self._generate_speaker_recommendations(evaluation_results)
        }

        return report

    def _generate_speaker_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on speaker attribution metrics"""
        recommendations = []

        if metrics.get('speaker_identification_accuracy', 0.0) < 0.8:
            recommendations.append("Improve speaker name normalization and matching logic")

        if metrics.get('host_recognition_accuracy', 0.0) < 0.9:
            recommendations.append("Enhance host identification patterns to better recognize Ricky and Brendan")

        if metrics.get('expert_recognition_accuracy', 0.0) < 0.8:
            recommendations.append("Improve expert identification by expanding credential detection patterns")

        if metrics.get('speaker_filtering_precision', 0.0) < 0.7:
            recommendations.append("Refine speaker filtering to reduce noise in results")

        return recommendations
```

### **Step 4.3: Updated Configuration Management**
**File**: `server/src/gent_disagreement_chat/core/config.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration for RAG optimizations"""
    
    # Threshold filtering
    enable_threshold_filtering: bool = True
    default_threshold: float = 0.6
    min_docs: int = 3
    max_docs: int = 10
    
    # Query enhancement
    enable_query_enhancement: bool = True
    use_hyde: bool = True
    use_query_expansion: bool = True
    use_intent_classification: bool = True
    
    # Enhanced formatting
    enable_enhanced_formatting: bool = True
    group_by_episode: bool = True
    show_relevance_indicators: bool = True
    include_context_windows: bool = False

    # Speaker-aware processing (based on transcript analysis)
    enable_speaker_awareness: bool = True
    boost_expert_opinions: bool = True
    separate_host_guest_content: bool = True
    enable_speaker_role_detection: bool = True

    # Topic-specific formatting (based on actual podcast content)
    enable_topic_formatting: bool = True
    legal_analysis_formatting: bool = True
    economic_policy_formatting: bool = True
    expert_opinion_formatting: bool = True
    historical_comparison_formatting: bool = True

    # Model settings
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    
    @classmethod
    def from_environment(cls):
        """Create config from environment variables"""
        import os

        return cls(
            enable_threshold_filtering=os.getenv('ENABLE_THRESHOLD_FILTERING', 'true').lower() == 'true',
            default_threshold=float(os.getenv('DEFAULT_THRESHOLD', '0.6')),
            min_docs=int(os.getenv('MIN_DOCS', '3')),
            max_docs=int(os.getenv('MAX_DOCS', '10')),
            enable_query_enhancement=os.getenv('ENABLE_QUERY_ENHANCEMENT', 'true').lower() == 'true',
            enable_enhanced_formatting=os.getenv('ENABLE_ENHANCED_FORMATTING', 'true').lower() == 'true',

            # Speaker-aware processing
            enable_speaker_awareness=os.getenv('ENABLE_SPEAKER_AWARENESS', 'true').lower() == 'true',
            boost_expert_opinions=os.getenv('BOOST_EXPERT_OPINIONS', 'true').lower() == 'true',
            separate_host_guest_content=os.getenv('SEPARATE_HOST_GUEST_CONTENT', 'true').lower() == 'true',
            enable_speaker_role_detection=os.getenv('ENABLE_SPEAKER_ROLE_DETECTION', 'true').lower() == 'true',

            # Topic-specific formatting
            enable_topic_formatting=os.getenv('ENABLE_TOPIC_FORMATTING', 'true').lower() == 'true',
            legal_analysis_formatting=os.getenv('LEGAL_ANALYSIS_FORMATTING', 'true').lower() == 'true',
            economic_policy_formatting=os.getenv('ECONOMIC_POLICY_FORMATTING', 'true').lower() == 'true',
            expert_opinion_formatting=os.getenv('EXPERT_OPINION_FORMATTING', 'true').lower() == 'true',
            historical_comparison_formatting=os.getenv('HISTORICAL_COMPARISON_FORMATTING', 'true').lower() == 'true'
        )
```

---

## **Implementation Timeline** ⏰

### **Week 1: Dynamic Threshold Filtering**
- [ ] Implement `find_relevant_above_adaptive_threshold()` method
- [ ] Add threshold-based search methods
- [ ] Update RAGService with threshold filtering
- [ ] Create unit tests
- [ ] Test with existing evaluation framework

### **Week 2: Query Enhancement**
- [ ] Create QueryEnhancer class
- [ ] Implement HyDE generation
- [ ] Add query expansion and intent classification
- [ ] Integrate multi-query search
- [ ] Test query enhancement effectiveness

### **Week 3: Enhanced Context Formatting (Updated for Transcript Analysis)**
- [ ] Implement speaker-aware result grouping (hosts vs. experts)
- [ ] Add topic-specific formatting for legal, economic, political content
- [ ] Create speaker role detection and attribution
- [ ] Update formatting methods with correct host names (Ricky & Brendan)
- [ ] Test speaker attribution accuracy and topic formatting

### **Week 4: Integration and Evaluation (Enhanced with Speaker Attribution)**
- [ ] Create EnhancedRAGService with speaker-aware capabilities
- [ ] Implement SpeakerAttributionEvaluator for transcript-based testing
- [ ] Test with podcast-specific questions (legal, economic, expert analysis)
- [ ] Validate host vs. expert distinction accuracy
- [ ] Generate comprehensive performance and attribution reports

---

## **Success Metrics** 📈

### **Quantitative Metrics**
- **Precision@5**: Target 15-25% improvement
- **Average Similarity**: Target >0.7 for retrieved results
- **Response Quality**: Measured via automated metrics
- **Retrieval Diversity**: Improved episode coverage
- **Speaker Attribution Accuracy**: Target >90% for host identification
- **Expert Recognition Rate**: Target >85% for academic/media experts
- **Topic Classification Accuracy**: Target >80% for legal/economic queries

### **Qualitative Metrics**
- **Context Readability**: Better organized, speaker-aware results
- **Query Understanding**: Better handling of legal, economic, and expert queries
- **Relevance Filtering**: Fewer irrelevant results with speaker-based filtering
- **Host vs. Expert Distinction**: Clear separation of perspectives in formatting
- **Topic-Specific Formatting**: Appropriate formatting based on query intent

### **Performance Metrics**
- **Response Time**: Maintain <2x current response time
- **API Costs**: Monitor OpenAI API usage increases
- **Memory Usage**: Track any significant increases

---

## **Risk Mitigation** 🛡️

### **Technical Risks**
- **API Rate Limits**: Implement caching for query enhancements
- **Response Time**: Monitor and optimize if needed
- **Complexity**: Maintain backward compatibility

### **Quality Risks**
- **Over-filtering**: Ensure min_docs prevents empty results
- **Query Drift**: Validate HyDE doesn't change question meaning
- **Formatting Complexity**: Keep enhanced formatting readable
- **Speaker Misattribution**: Ensure accurate host vs. expert identification
- **Topic Misclassification**: Validate intent classification accuracy

### **Transcript-Specific Risks**
- **Host Name Errors**: Ensure correct recognition of Ricky Ghoshroy and Brendan Kelly
- **Expert Credential Loss**: Maintain academic and media expert distinctions
- **Topic Drift**: Prevent legal/economic content misclassification

---

## **Key Updates Based on Transcript Analysis** 🔄

### **Critical Corrections Made**
- ✅ **Host Names**: Corrected from "Tyler & Ben" to "Ricky Ghoshroy & Brendan Kelly"
- ✅ **Domain Vocabulary**: Added legal, economic, and political terminology from actual discussions
- ✅ **Speaker Roles**: Enhanced identification of co-hosts, academic experts, and media experts
- ✅ **Query Intents**: Added legal_analysis, economic_policy, historical_comparison categories
- ✅ **Test Questions**: Created based on actual podcast content and expert guests

### **New Capabilities Added**
- 🆕 **Speaker-Aware Formatting**: Separate host discussions from expert analysis
- 🆕 **Topic-Specific Formatting**: Tailored presentation for legal, economic, and political content
- 🆕 **Expert Attribution**: Enhanced identification and credentialing of academic/media guests
- 🆕 **Intent-Based Relevance**: Boost expert opinions for specialized queries
- 🆕 **Comprehensive Evaluation**: Speaker attribution accuracy and topic classification metrics

This updated implementation plan provides a comprehensive roadmap for implementing all three optimization strategies while accurately reflecting the actual content and structure of "A Gentleman's Disagreement" podcast. Each phase builds upon transcript-based insights, ensuring the system works optimally with the real podcast data.