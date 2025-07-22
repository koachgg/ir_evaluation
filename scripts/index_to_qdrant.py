#!/usr/bin/env python3
"""
Index Documents to Qdrant

This script indexes a document corpus to Qdrant vector database
using a specified embedding model.
"""

import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import gc
import time

class DocumentIndexer:
    """Document indexer for Qdrant vector database"""
    
    def __init__(self, qdrant_url, model_name, batch_size=32):
        """
        Initialize the document indexer
        
        Args:
            qdrant_url (str): Qdrant server URL
            model_name (str): Name of the embedding model
            batch_size (int): Batch size for processing
        """
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
    def create_collection(self, collection_name):
        """Create or recreate Qdrant collection"""
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")
    
    def index_documents(self, corpus_path, collection_name):
        """
        Index documents from corpus file to Qdrant
        
        Args:
            corpus_path (str): Path to corpus JSONL file
            collection_name (str): Qdrant collection name
        """
        print(f"Indexing documents to collection: {collection_name}")
        
        # Count total documents
        with open(corpus_path, 'r', encoding='utf-8') as f:
            total_docs = sum(1 for _ in f)
        
        print(f"Total documents to process: {total_docs}")
        
        # Process documents in batches
        current_batch = []
        processed = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=total_docs, desc="Indexing documents")
            
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    
                    # Combine title and text
                    text = ' '.join(filter(None, [
                        doc.get('title', ''), 
                        doc.get('text', '')
                    ]))
                    
                    if text.strip():
                        current_batch.append({
                            'id': doc.get('_id', str(processed)),
                            'text': text,
                            'title': doc.get('title', ''),
                            'original_text': doc.get('text', '')
                        })
                    
                    if len(current_batch) >= self.batch_size:
                        self._process_batch(current_batch, collection_name)
                        processed += len(current_batch)
                        pbar.update(len(current_batch))
                        current_batch = []
                        
                        # Memory cleanup
                        gc.collect()
                
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                    continue
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
            
            # Process remaining batch
            if current_batch:
                self._process_batch(current_batch, collection_name)
                processed += len(current_batch)
                pbar.update(len(current_batch))
            
            pbar.close()
        
        print(f"Successfully indexed {processed} documents")
    
    def _process_batch(self, batch, collection_name):
        """Process a batch of documents"""
        try:
            # Extract texts for embedding
            texts = [doc['text'] for doc in batch]
            
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Create points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(batch, embeddings)):
                points.append(
                    models.PointStruct(
                        id=hash(doc['id']) % (2**31),  # Convert to positive int
                        vector=embedding.tolist(),
                        payload={
                            "id": doc['id'],
                            "title": doc['title'],
                            "text": doc['original_text']
                        }
                    )
                )
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            
        except Exception as e:
            print(f"Error processing batch: {e}")

def main():
    parser = argparse.ArgumentParser(description="Index documents to Qdrant")
    parser.add_argument("--corpus", required=True, 
                       help="Path to corpus JSONL file")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", 
                       help="Qdrant server URL")
    parser.add_argument("--collection", default="scifact", 
                       help="Qdrant collection name")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize indexer
    indexer = DocumentIndexer(args.qdrant_url, args.model, args.batch_size)
    
    # Create collection
    indexer.create_collection(args.collection)
    
    # Index documents
    indexer.index_documents(args.corpus, args.collection)
    
    print("Indexing completed successfully!")

if __name__ == "__main__":
    main()
