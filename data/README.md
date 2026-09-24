# Learning Data

The foundation lessons use a small fictional product called Northstar. The data
is repository-owned, contains no personal information, and is deliberately
small enough to inspect by hand.

## Files

| File | Purpose |
|---|---|
| `raw/rag_101_corpus.jsonl` | Eight source documents with stable IDs, titles, content, and metadata |
| `evaluation/golden_questions.jsonl` | Seven answerable questions and one unsupported question for abstention testing |

## Corpus record

```json
{
  "id": "stable-document-id",
  "title": "Human-readable title",
  "content": "Source text",
  "metadata": {
    "product": "northstar",
    "topic": "topic-name",
    "version": "YYYY-MM"
  }
}
```

## Golden-question record

```json
{
  "id": "stable-question-id",
  "question": "Question text",
  "relevant_doc_ids": ["stable-document-id"],
  "reference_answer": "Short expected answer",
  "should_abstain": false
}
```

Callout - Warning:
This dataset proves that the learning pipeline behaves as designed on a small,
controlled fixture. Its perfect retrieval score must not be interpreted as a
production quality estimate.
