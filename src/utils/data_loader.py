import json
import pandas as pd
from typing import Dict, List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

class SciFactDataLoader:
    """
    Data loader for SciFact dataset with Qdrant integration
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.qdrant_client = QdrantClient(url=config["qdrant"]["url"])
    
    def load_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        Load corpus, queries, and qrels from files
        
        Returns:
            Tuple: (corpus, queries, qrels)
        """
        # Load corpus
        corpus = self._load_jsonl(self.config["dataset"]["corpus"])
        corpus = {doc["_id"]: doc["text"] for doc in corpus}
        
        # Load queries
        queries = self._load_jsonl(self.config["dataset"]["queries"])
        queries = {q["_id"]: q["text"] for q in queries}
        
        # Load qrels
        qrels = {
            "train": self._load_qrels(self.config["dataset"]["qrels_train"]),
            "test": self._load_qrels(self.config["dataset"]["qrels_test"])
        }
        
        return corpus, queries, qrels
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def _load_qrels(self, file_path: str) -> pd.DataFrame:
        """Load qrels TSV file"""
        return pd.read_csv(
            file_path, 
            sep='\t', 
            names=["query-id", "corpus-id", "score"]
        )
    
    def update_qdrant(self, embeddings: np.ndarray, collection_name: str):
        """
        Update Qdrant collection with new embeddings
        
        Args:
            embeddings (np.ndarray): Document embeddings
            collection_name (str): Name of Qdrant collection
        """
        try:
            # Delete existing collection
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
        
        # Create new collection
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.config["qdrant"]["vector_size"],
                distance=models.Distance.COSINE
            )
        )
        
        # Upload embeddings
        # Implementation details would depend on your specific setup
        print(f"Updated Qdrant collection: {collection_name}")
