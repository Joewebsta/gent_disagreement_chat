### TODO

- SQLAlchemy + Alembic for database management and migrations
- queryParser SPEAKER_MAPPINGS object.

  - What if there are multiple speakers with the same name?
  - This is not very scalable. I should use a more robust solution.

- Update memory management. I don't believe all chat messages are sent to the LLM to generate a response.
  - As a result I can ask:
    - What does Ricky think about Gerry mandering?
    - Then followup with, what about brendan - and this won't work
- Add more episodes
- Process full episode transcripts
- Run the evaluation baseline
- Understand how evaluation actually works...

- Revist "professor" in SPECIAL_SPEAKER_KEYWORDS (QueryParser class) - I think this should be removed
- Should the "speakers" table have a "title" column so that "professor" is separate from "Jack Beerman"?
- Do additinal entries need to be added to SPEAKER_MAPPINGS (QueryParser class) for guests. Currently only includes Ricky and Brendan.
