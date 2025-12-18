## Low hanging fruit

## Future improvements

- LLM should correct incorrect published dates. It should use the published date in the formatted results. It should correct itself if the date in the user provided prompt is incorrect.
- Asking about the latest episode should filter by the episode with the largest number...
- Balance broad queries about an episode so that transcript segements from both Brendan and Ricky are included

## Done

- Ricky should be referred to as "Ricky" not "Ghoshroy" by the LLM
- If the LLM receives 0 segments, it should say it doesn't know anything or doesn't have an answer to provide rather than hallucinate.
- Episode-scoped LLM response should spell out the entire episode name. Update the prompt to ensure this happens
  - Example: "In Episode 181: [NAME GOES HERE]"

## Episode-Scoped Questions

- “Tell me about the Kennedy Center discussion in the August 25th episode”
  - Verdict 1: 😤 Ugly
  - The content is correct due to semantic search for the "Kennedy Center", BUT an episode was not published on that day.
- “What happened in the latest episode about economics and tariffs?”
  - Verdict 1: 😤 Ugly
  - The response is correct, but the filter episode number was None. How can I fix this?
    - `filters {'episode_number': None, 'speaker': None, 'date_range': None}`
- “Summarize episode 180's discussion with Professor Jack Beermann about the Supreme Court”
  - Verdict 1: 😤 Ugly
  - The inclusion of the professor's name filters the segements by the prof, which isnt what I really want here. Is it?
  - `filters {'episode_number': 180, 'speaker': 'Professor Jack Beermann', 'date_range': None}`
- “Summarize episode 180's discussion about the Supreme Court”
  - Verdict 1: ⛔ Bad
  - The retrieved segmenents are not balanced. They include BK and the prof, but no Ricky...
  - I should consider generating a summary for each episode and saving it to the DB
- “What topics did Brendan and Ricky cover in episode 181?”
  - Verdict 1: ✅ Good
- “What did Lydia DePhillis discuss about Nepal in episode 182?”
  - Verdict 1: ✅ Good
- “What was discussed about Nepal in episode 182?”
  - Verdict 1: ✅ Good

## Multi-Episode Topical Questions

- “What are the hosts' views on Trump's use of presidential power across all episodes?”
- “How do Brendan and Ricky analyze Supreme Court decisions throughout the podcast?”
- “What do they think about tariffs based on all their discussions?”
- “Tell me about all the times they discuss authoritarianism and democratic norms”
- “What are their overall views on economic policy and free markets?”

## Speaker-Specific Questions

- Who is Professor Jack Beermann
  - Verdict 1: 😤 Ugly
  - Returns 3 low similarity results... Do I need to save descriptions of guests somewhere? Send this info to the LLM?
- “What does Ricky think about gerrymandering and redistricting?”
- “What are Brendan's views on originalism and textualism in constitutional law?”
- “What did Professor Jack Beerman say about the Supreme Court's politicization?”
- “Has Ricky ever discussed his concerns about creeping authoritarianism?”
- “What did Lydia DePhillis explain about Chinese EV manufacturers?”

## Comparative/Contrasting Questions

- “Do Brendan and Ricky disagree on whether Democrats should fight gerrymandering with gerrymandering?”
- “Compare Professor Beerman's view of the Warren Court to his view of the current Supreme Court”
- “How do the hosts' views on free market capitalism differ when discussing Intel vs. discussing tariffs?”
- “What are the differences between Ricky's and Brendan's reactions to Trump's DC National Guard deployment?”
- “Compare what they discussed about the economy in episode 181 versus episode 182”

## Temporal/Chronological Questions

- “What have Brendan and Ricky discussed in the last month based on these episodes?”
- “How has their discussion of Trump's second term evolved from episode 180 to 182?”
- “Track their analysis of Supreme Court power from the July discussion to August”
- “What were they predicting about the economy in late August 2024?”
- “How did the conversation shift from constitutional law in episode 180 to economics in episode 182?”

## Factual Lookup Questions

- “Which episode featured Professor Jack Beerman from Boston University?”
- “What case did they mention about religious exemptions to school curriculum?”
- “When did they discuss the Bureau of Labor Statistics firing?”
- “Which episode talked about electric vehicles in Nepal?”
- “Have they ever discussed the Posse Comitatus Act?”

## Analytical/Opinion Questions

- “Why do the hosts seem particularly concerned about the normalization of authoritarian tactics?”
- “What underlying philosophy drives their skepticism of government intervention in private markets?”
- “How do they approach disagreement between progressive and conservative viewpoints?”
- “What patterns emerge in their analysis of Trump administration policies?”
- “Why do they frequently reference historical precedents when discussing current events?”

## Summarization Questions

- “Summarize the main themes across these three episodes from August 2024”
- “What are the key Supreme Court cases and issues they discussed in episode 180?”
- “Give me an overview of their concerns about democratic institutions based on these episodes”
- “What topics related to economics and trade do they cover most often?”
- “Summarize the different types of government overreach they identify across these episodes”
