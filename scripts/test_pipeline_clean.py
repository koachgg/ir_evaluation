#!/usr/bin/env python3
"""
Clean Test Pipeline

A simplified version of the original test pipeline without company-specific information.
This demonstrates basic RAG pipeline implementation.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def rag_pipeline_demo():
    """
    Demonstrate a basic RAG pipeline with chunking, embedding, and retrieval
    """
    
    # Step 1: Text Chunking
    text = "Paris is the capital of France. France is known for its wine and cheese. The Eiffel Tower is located in Paris."
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Step 2: Embedding Generation
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    chunk_embeddings = embedding_model.encode(chunks)
    print(f"Generated embeddings with shape: {chunk_embeddings.shape}")
    
    # Step 3: Vector Database Setup
    embeddings_np = np.array(chunk_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    print(f"Added {index.ntotal} vectors to FAISS index")
    
    # Step 4: Query and Retrieval
    query = "What is the capital of France?"
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=3)
    
    top_k_chunks = [chunks[i] for i in indices[0]]
    print(f"\nQuery: {query}")
    print("Top retrieved chunks:")
    for i, chunk in enumerate(top_k_chunks):
        print(f"{i+1}. {chunk}")
    
    return top_k_chunks

def simple_re_ranker(query, chunks):
    """
    Simple re-ranking based on keyword overlap
    """
    query_words = set(query.lower().split())
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words.intersection(chunk_words))
        scored_chunks.append((chunk, overlap))
    
    # Sort by overlap score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks]

def main():
    """Main execution function"""
    print("=== RAG Pipeline Demo ===")
    
    # Run basic pipeline
    retrieved_chunks = rag_pipeline_demo()
    
    # Apply simple re-ranking
    query = "What is the capital of France?"
    re_ranked = simple_re_ranker(query, retrieved_chunks)
    
    print("\nAfter re-ranking:")
    for i, chunk in enumerate(re_ranked):
        print(f"{i+1}. {chunk}")
    
    # Generate simple response
    context = " ".join(re_ranked)
    print(f"\nFinal context: {context}")
    print("Note: In a full system, this context would be sent to an LLM for final answer generation.")

if __name__ == "__main__":
    main()
