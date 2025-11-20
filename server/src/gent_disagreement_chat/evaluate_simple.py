"""Simple evaluation script for quick feedback on retrieval improvements

This script provides a low-tech, manual evaluation approach that:
- Tests representative questions across different categories
- Shows what's being retrieved (episodes, speakers, similarities)
- Prompts for manual quality assessment
- Saves results to JSON for comparison over time

Usage:
    cd server/src/gent_disagreement_chat
    python evaluate_simple.py
"""

import json
from collections import Counter
from core.rag_service import RAGService

EPISODE_SCOPED = "Episode-Scoped Questions"
MULTI_EPISODE_TOPICAL = "Multi-Episode Topical Questions"
SPEAKER_SPECIFIC = "Speaker-Specific Questions"
COMPARATIVE = "Comparative/Contrasting Questions"
TEMPORAL = "Temporal/Chronological Questions"
FACTUAL_LOOKUP = "Factual Lookup Questions"
ANALYTICAL = "Analytical/Opinion Questions"
SUMMARIZATION = "Summarization Questions"

# Test questions covering different types
TEST_QUESTIONS = [
    # 1. Episode-Scoped Questions
    (
        "Summarize episode 180's discussion with Professor Jack Beerman about the Supreme Court",
        EPISODE_SCOPED
    ),
    # (
    #     "What topics did Brendan and Ricky cover in episode 181?",
    #     EPISODE_SCOPED
    # ),
    # (
    #     "What did Lydia DePhillis discuss about Nepal in episode 182?",
    #     EPISODE_SCOPED
    # ),
    # (
    #     "Tell me about the Kennedy Center discussion in the August 25th episode",
    #     EPISODE_SCOPED
    # ),
    # (
    #     "What happened in the latest episode about economics and tariffs?",
    #     EPISODE_SCOPED
    # ),

    # 2. Multi-Episode Topical Questions
    (
        "What are the hosts' views on Trump's use of presidential power across all episodes?",
        MULTI_EPISODE_TOPICAL
    ),
    # (
    #     "How do Brendan and Ricky analyze Supreme Court decisions throughout the podcast?",
    #     MULTI_EPISODE_TOPICAL
    # ),
    # (
    #     "What do they think about tariffs based on all their discussions?",
    #     MULTI_EPISODE_TOPICAL
    # ),
    # (
    #     "Tell me about all the times they discuss authoritarianism and democratic norms",
    #     MULTI_EPISODE_TOPICAL
    # ),
    # (
    #     "What are their overall views on economic policy and free markets?",
    #     MULTI_EPISODE_TOPICAL
    # ),

    # 3. Speaker-Specific Questions
    # (
    #     "What does Ricky think about gerrymandering and redistricting?",
    #     SPEAKER_SPECIFIC
    # ),
    # (
    #     "What are Brendan's views on originalism and textualism in constitutional law?",
    #     SPEAKER_SPECIFIC
    # ),
    # (
    #     "What did Professor Jack Beerman say about the Supreme Court's politicization?",
    #     SPEAKER_SPECIFIC
    # ),
    # (
    #     "Has Ricky ever discussed his concerns about creeping authoritarianism?",
    #     SPEAKER_SPECIFIC
    # ),
    # (
    #     "What did Lydia DePhillis explain about Chinese EV manufacturers?",
    #     SPEAKER_SPECIFIC
    # ),

    # 4. Comparative/Contrasting Questions
    # (
    #     "Do Brendan and Ricky disagree on whether Democrats should fight gerrymandering with gerrymandering?",
    #     COMPARATIVE
    # ),
    # (
    #     "Compare Professor Beerman's view of the Warren Court to his view of the current Supreme Court",
    #     COMPARATIVE
    # ),
    # (
    #     "How do the hosts' views on free market capitalism differ when discussing Intel vs. discussing tariffs?",
    #     COMPARATIVE
    # ),
    # (
    #     "What are the differences between Ricky's and Brendan's reactions to Trump's DC National Guard deployment?",
    #     COMPARATIVE
    # ),
    # (
    #     "Compare what they discussed about the economy in episode 181 versus episode 182",
    #     COMPARATIVE
    # ),

    # 5. Temporal/Chronological Questions
    (
        "What have Brendan and Ricky discussed in the last month based on these episodes?",
        TEMPORAL
    ),
    # (
    #     "How has their discussion of Trump's second term evolved from episode 180 to 182?",
    #     TEMPORAL
    # ),
    # (
    #     "Track their analysis of Supreme Court power from the July discussion to August",
    #     TEMPORAL
    # ),
    # (
    #     "What were they predicting about the economy in late August 2024?",
    #     TEMPORAL
    # ),
    # (
    #     "How did the conversation shift from constitutional law in episode 180 to economics in episode 182?",
    #     TEMPORAL
    # ),

    # 6. Factual Lookup Questions
    # (
    #     "Which episode featured Professor Jack Beerman from Boston University?",
    #     FACTUAL_LOOKUP
    # ),
    # (
    #     "What case did they mention about religious exemptions to school curriculum?",
    #     FACTUAL_LOOKUP
    # ),
    # (
    #     "When did they discuss the Bureau of Labor Statistics firing?",
    #     FACTUAL_LOOKUP
    # ),
    # (
    #     "Which episode talked about electric vehicles in Nepal?",
    #     FACTUAL_LOOKUP
    # ),
    # (
    #     "Have they ever discussed the Posse Comitatus Act?",
    #     FACTUAL_LOOKUP
    # ),

    # 7. Analytical/Opinion Questions
    # (
    #     "Why do the hosts seem particularly concerned about the normalization of authoritarian tactics?",
    #     ANALYTICAL
    # ),
    # (
    #     "What underlying philosophy drives their skepticism of government intervention in private markets?",
    #     ANALYTICAL
    # ),
    # (
    #     "How do they approach disagreement between progressive and conservative viewpoints?",
    #     ANALYTICAL
    # ),
    # (
    #     "What patterns emerge in their analysis of Trump administration policies?",
    #     ANALYTICAL
    # ),
    # (
    #     "Why do they frequently reference historical precedents when discussing current events?",
    #     ANALYTICAL
    # ),

    # 8. Summarization Questions
    # (
    #     "Summarize the main themes across these three episodes from August 2024",
    #     SUMMARIZATION
    # ),
    # (
    #     "What are the key Supreme Court cases and issues they discussed in episode 180?",
    #     SUMMARIZATION
    # ),
    # (
    #     "Give me an overview of their concerns about democratic institutions based on these episodes",
    #     SUMMARIZATION
    # ),
    # (
    #     "What topics related to economics and trade do they cover most often?",
    #     SUMMARIZATION
    # ),
    # (
    #     "Summarize the different types of government overreach they identify across these episodes",
    #     SUMMARIZATION
    # ),
]


def evaluate():
    """Run simple evaluation and print results"""
    print("Initializing RAG Service...")
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

        if not segments:
            print("\n⚠️  No segments retrieved!")
            print(f"\n{'─'*80}")
            print("Is this retrieval good? (y/n/notes): ", end="")
            feedback = input()

            results.append({
                "question": question,
                "category": category,
                "num_segments": 0,
                "num_episodes": 0,
                "speaker_balance": {},
                "avg_similarity": 0,
                "feedback": feedback
            })
            continue

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
        print("\nTop 3 segments:")
        for i, segment in enumerate(segments[:3], 1):
            print(f"\n{i}. Episode {segment['episode_number']} - {segment['name']} (sim: {segment['similarity']:.3f})")
            text = segment['text']
            print(f"   {text}")

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

        # Only calculate averages if we have segments
        items_with_segments = [i for i in items if i['num_segments'] > 0]
        if items_with_segments:
            print(f"  Avg segments: {sum(i['num_segments'] for i in items_with_segments) / len(items_with_segments):.1f}")
            print(f"  Avg episodes: {sum(i['num_episodes'] for i in items_with_segments) / len(items_with_segments):.1f}")
            print(f"  Avg similarity: {sum(i['avg_similarity'] for i in items_with_segments) / len(items_with_segments):.3f}")
        else:
            print("  No segments retrieved for any questions in this category")

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    evaluate()
