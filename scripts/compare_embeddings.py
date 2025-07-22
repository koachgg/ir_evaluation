#!/usr/bin/env python3
"""
Embedding Comparison Script

This script compares different embedding models on the SciFact dataset
and generates performance metrics for each model.
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from tqdm import tqdm
import argparse

def load_test_data(file_path):
    """Load test dataset"""
    return pd.read_csv(file_path, sep="\t", names=["query-id", "corpus-id", "score"])

def load_queries(file_path):
    """Load queries from JSONL file"""
    with open(file_path, 'r') as f:
        return {q["_id"]: q["text"] for q in map(json.loads, f)}

def evaluate_model(model_name, qdrant_url, collection_name, test_data, queries):
    """Evaluate a specific embedding model"""
    print(f"Evaluating model: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name)
    client = QdrantClient(url=qdrant_url)
    
    results = []
    
    for query_id in tqdm(test_data["query-id"].unique()):
        relevant_docs = set(test_data[test_data["query-id"] == query_id]["corpus-id"])
        query_text = queries.get(str(query_id), f"query-{query_id}")
        
        try:
            # Get query embedding
            query_vector = model.encode(query_text).tolist()
            
            # Search in Qdrant
            search_results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=10
            )
            
            retrieved_docs = [r.payload["id"] for r in search_results]
            
            # Calculate rank of first relevant document
            rank = next((i for i, doc_id in enumerate(retrieved_docs) 
                        if doc_id in relevant_docs), 9)
            
            results.append({
                "query-id": query_id,
                "model": model_name,
                "rank": rank,
                "relevant_found": rank < 9
            })
            
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            results.append({
                "query-id": query_id,
                "model": model_name,
                "rank": 9,
                "relevant_found": False
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", 
                       help="Qdrant server URL")
    parser.add_argument("--collection", default="scifact", 
                       help="Qdrant collection name")
    parser.add_argument("--test-data", default="data/test.tsv", 
                       help="Path to test data")
    parser.add_argument("--queries", default="data/queries.jsonl", 
                       help="Path to queries file")
    parser.add_argument("--output", default="results/embedding_comparison.xlsx", 
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Load data
    test_data = load_test_data(args.test_data)
    queries = load_queries(args.queries)
    
    # Models to compare
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    all_results = []
    
    for model_name in models:
        model_results = evaluate_model(
            model_name, args.qdrant_url, args.collection, test_data, queries
        )
        all_results.append(model_results)
    
    # Combine results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate summary statistics
    summary = final_df.groupby("model").agg({
        "rank": ["mean", "std"],
        "relevant_found": "mean"
    }).round(3)
    
    print("\nSummary Results:")
    print(summary)
    
    # Save results
    final_df.to_excel(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
