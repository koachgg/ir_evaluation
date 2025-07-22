# Information Retrieval Systems with Knowledge Distillation

This repository contains implementations of information retrieval systems using knowledge distillation techniques and vector databases.

## Project Overview

This project explores various embedding models and retrieval techniques for scientific document retrieval, with a focus on:
- Knowledge distillation for efficient retrieval models
- Vector database integration using Qdrant
- Embedding model comparison and evaluation
- Retrieval performance analysis

## Features

- **Knowledge Distillation Pipeline**: Teacher-student model training for efficient retrieval
- **Vector Database Integration**: Qdrant integration for scalable vector search
- **Embedding Comparison**: Side-by-side comparison of different embedding models
- **Evaluation Metrics**: Comprehensive evaluation using NDCG, Recall, MAP metrics
- **SciFact Dataset Support**: Specialized handling for scientific fact verification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ir-knowledge-distillation.git
cd ir-knowledge-distillation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API tokens
```

## Usage

### Basic Pipeline
```python
from knowledge_distillation_pipeline import KDRetriever

# Initialize pipeline with config
pipeline = KDRetriever("config/scifact_config.yaml")

# Run complete pipeline
results = pipeline.run_pipeline()
print(f"Results: {results}")
```

### Embedding Comparison
```python
# Compare different embedding models
python scripts/compare_embeddings.py
```

### Vector Database Setup
```python
# Index documents to Qdrant
python scripts/index_to_qdrant.py
```

## Configuration

Update the configuration file `config/scifact_config.yaml` with your settings:

```yaml
teacher_model: "sentence-transformers/all-MiniLM-L6-v2"
student_model: "sentence-transformers/all-MiniLM-L12-v2"
qdrant:
  url: "http://localhost:6333"
  collection: "scifact"
dataset:
  corpus: "data/SciFact.jsonl"
  queries: "data/queries.jsonl"
  qrels: "data/test.tsv"
```

## File Structure

```
├── README.md
├── requirements.txt
├── config/
│   └── scifact_config.yaml
├── src/
│   ├── knowledge_distillation_pipeline.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── model_trainer.py
│   │   └── evaluator.py
├── scripts/
│   ├── compare_embeddings.py
│   ├── index_to_qdrant.py
│   └── evaluation_pipeline.py
├── notebooks/
│   ├── embedding_comparison.ipynb
│   ├── retrieval_evaluation.ipynb
│   └── knowledge_distillation_process.ipynb
└── data/
    ├── SciFact.jsonl
    ├── queries.jsonl
    └── test.tsv
```

## Results

The system has been evaluated on the SciFact dataset with the following results:
- NDCG@10: [Your results]
- Recall@10: [Your results]  
- MAP@10: [Your results]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SciFact dataset for scientific fact verification
- Sentence Transformers library
- Qdrant vector database
- BEIR benchmark framework
