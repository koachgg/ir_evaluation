# Data Directory

This directory should contain your dataset files:

## Required Files:
- `SciFact.jsonl` - Corpus documents in JSONL format
- `queries.jsonl` - Query texts in JSONL format  
- `test.tsv` - Test query-document relevance judgments
- `train.tsv` - Training query-document relevance judgments (if available)

## File Formats:

### SciFact.jsonl
Each line should be a JSON object with:
```json
{
  "_id": "document_id",
  "title": "Document title",
  "text": "Document content"
}
```

### queries.jsonl
Each line should be a JSON object with:
```json
{
  "_id": "query_id", 
  "text": "Query text"
}
```

### test.tsv / train.tsv
Tab-separated file with columns:
```
query-id	corpus-id	score
```

## Note:
Due to file size and licensing, actual data files are not included in this repository.
Please download the SciFact dataset from the official source and place files here.
