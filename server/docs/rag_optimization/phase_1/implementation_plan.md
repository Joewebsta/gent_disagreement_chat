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
**File**: `server/src/gent_disagreement_rag/core/vector_search.py`

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
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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
**File**: `server/src/gent_disagreement_rag/core/query_enhancer.py`

```python
import re
from typing import Dict, List, Any
from openai import OpenAI
import os

class QueryEnhancer:
    """Handles query preprocessing and enhancement for better retrieval"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Podcast-specific vocabulary expansion
        self.podcast_vocabulary = {
            'hosts': ['Tyler', 'Ben', 'hosts', 'speakers', 'panelists'],
            'topics': ['politics', 'economics', 'technology', 'culture', 'philosophy'],
            'concepts': ['libertarianism', 'conservatism', 'progressivism', 'markets', 'regulation']
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
            prompt = f"""
            Based on the question "{question}", generate a hypothetical answer that might appear in a podcast transcript.
            Focus on the type of information that would be discussed, not specific facts.
            Keep it concise (1-2 sentences).
            """
            
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
        """Classify the intent of the query"""
        question_lower = question.lower()
        
        # Simple keyword-based classification
        if any(word in question_lower for word in ['what', 'who', 'when', 'where', 'which']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why', 'explain', 'analyze']):
            return 'analytical'
        elif any(word in question_lower for word in ['compare', 'versus', 'difference', 'contrast']):
            return 'comparative'
        elif any(word in question_lower for word in ['opinion', 'think', 'believe', 'feel']):
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
        # Simple synonym mapping for podcast context
        synonym_map = {
            'discuss': ['talk about', 'cover', 'address', 'explore'],
            'opinion': ['view', 'perspective', 'stance', 'position'],
            'topic': ['subject', 'theme', 'issue', 'matter'],
            'hosts': ['speakers', 'panelists', 'Tyler', 'Ben']
        }
        
        synonyms = []
        question_lower = question.lower()
        
        for term, synonym_list in synonym_map.items():
            if term in question_lower:
                synonyms.extend(synonym_list)
        
        return synonyms
```

#### **Step 2.2: Integrate Query Enhancement into RAGService**
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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
```

### **Testing Strategy**
- Test query enhancement with various question types
- Validate HyDE generation quality
- Test multi-query search effectiveness
- Measure retrieval improvement with enhanced queries

---

## **Phase 3: Enhanced Context Formatting** 📝

### **Implementation Steps**

#### **Step 3.1: Enhanced Context Formatting**
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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

#### **Step 3.2: Context Windows and Smart Filtering**
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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
    
    # Extract speaker name
    speaker_match = re.search(r'(Tyler|Ben)', query, re.IGNORECASE)
    if speaker_match:
        filters['speaker'] = speaker_match.group(1)
    
    return filters
```

#### **Step 3.3: Update Main Methods to Use Enhanced Formatting**
**File**: `server/src/gent_disagreement_rag/core/rag_service.py`

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
**File**: `server/src/gent_disagreement_rag/core/enhanced_rag_service.py`

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
**File**: `server/src/gent_disagreement_rag/evaluation/optimization_evaluator.py`

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
```

### **Step 4.3: Configuration Management**
**File**: `server/src/gent_disagreement_rag/core/config.py`

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
            enable_enhanced_formatting=os.getenv('ENABLE_ENHANCED_FORMATTING', 'true').lower() == 'true'
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

### **Week 3: Enhanced Context Formatting**
- [ ] Implement hierarchical result grouping
- [ ] Add relevance indicators and episode organization
- [ ] Create context windows functionality
- [ ] Update formatting methods
- [ ] Test formatting improvements

### **Week 4: Integration and Evaluation**
- [ ] Create EnhancedRAGService
- [ ] Update evaluation framework
- [ ] Run comprehensive optimization tests
- [ ] Generate performance reports
- [ ] Document findings and recommendations

---

## **Success Metrics** 📈

### **Quantitative Metrics**
- **Precision@5**: Target 15-25% improvement
- **Average Similarity**: Target >0.7 for retrieved results
- **Response Quality**: Measured via automated metrics
- **Retrieval Diversity**: Improved episode coverage

### **Qualitative Metrics**
- **Context Readability**: Better organized, hierarchical results
- **Query Understanding**: Better handling of various question types
- **Relevance Filtering**: Fewer irrelevant results included

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

This implementation plan provides a comprehensive roadmap for implementing all three optimization strategies while maintaining system reliability and measurability. Each phase builds upon the previous one, allowing for iterative testing and validation of improvements.