import pandas as pd
import numpy as np
from typing import Dict, List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score

class RetrievalEvaluator:
    """
    Evaluator for Information Retrieval Systems
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.qdrant_client = QdrantClient(url=config["qdrant"]["url"])
        
    def evaluate(self, model: SentenceTransformer, queries_test: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test queries
        
        Args:
            model: Sentence transformer model to evaluate
            queries_test: Test query-document relevance data
            
        Returns:
            Dict: Evaluation metrics
        """
        results = []
        unique_queries = queries_test["query-id"].unique()
        
        for query_id in unique_queries:
            # Get relevant documents for this query
            relevant_docs = set(
                queries_test[queries_test["query-id"] == query_id]["corpus-id"]
            )
            
            # Retrieve documents using the model
            retrieved_docs = self._retrieve_documents(
                query_id, model, k=max(self.config["evaluation"]["k_values"])
            )
            
            # Calculate metrics for different k values
            for k in self.config["evaluation"]["k_values"]:
                metrics = self._calculate_metrics(
                    retrieved_docs[:k], relevant_docs, k
                )
                metrics["query_id"] = query_id
                metrics["k"] = k
                results.append(metrics)
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _retrieve_documents(self, query_id: str, model: SentenceTransformer, 
                          k: int = 10) -> List[str]:
        """
        Retrieve top-k documents for a given query
        
        Args:
            query_id: Query identifier
            model: Model to use for retrieval
            k: Number of documents to retrieve
            
        Returns:
            List[str]: Retrieved document IDs
        """
        # Create query embedding
        query_text = f"query-{query_id}"  # Placeholder
        query_vector = model.encode(query_text).tolist()
        
        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=self.config["qdrant"]["collection"],
            query_vector=query_vector,
            limit=k
        )
        
        return [r.payload["id"] for r in results]
    
    def _calculate_metrics(self, retrieved_docs: List[str], 
                          relevant_docs: set, k: int) -> Dict:
        """
        Calculate evaluation metrics for retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Number of top documents considered
            
        Returns:
            Dict: Calculated metrics
        """
        # Create relevance scores
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        
        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Calculate IDCG
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
        
        # Calculate NDCG
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        # Calculate Recall@k
        recall_at_k = sum(relevance_scores) / len(relevant_docs) if len(relevant_docs) > 0 else 0
        
        # Calculate MAP@k
        precisions = []
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                precision_at_i = sum(relevance_scores[:i+1]) / (i + 1)
                precisions.append(precision_at_i)
        
        map_at_k = np.mean(precisions) if precisions else 0
        
        return {
            "ndcg": ndcg,
            "recall": recall_at_k,
            "map": map_at_k,
            "dcg": dcg
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate evaluation results across all queries
        
        Args:
            results: List of per-query results
            
        Returns:
            Dict: Aggregated metrics
        """
        df = pd.DataFrame(results)
        
        aggregated = {}
        for k in self.config["evaluation"]["k_values"]:
            k_results = df[df["k"] == k]
            aggregated[f"ndcg@{k}"] = k_results["ndcg"].mean()
            aggregated[f"recall@{k}"] = k_results["recall"].mean()
            aggregated[f"map@{k}"] = k_results["map"].mean()
        
        return aggregated
