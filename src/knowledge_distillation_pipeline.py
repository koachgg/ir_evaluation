import yaml
from utils.data_loader import SciFactDataLoader
from utils.model_trainer import KDTrainer
from utils.evaluator import RetrievalEvaluator

class KDRetriever:
    """
    Knowledge Distillation Retriever Pipeline
    
    This class implements a complete pipeline for knowledge distillation
    in information retrieval tasks, specifically designed for scientific
    document retrieval.
    """
    
    def __init__(self, config_path):
        """
        Initialize the KD Retriever with configuration
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = SciFactDataLoader(self.config)
        self.trainer = KDTrainer(self.config)
        self.evaluator = RetrievalEvaluator(self.config)

    def run_pipeline(self):
        """
        Execute the complete knowledge distillation pipeline
        
        Returns:
            dict: Evaluation metrics including NDCG, Recall, MAP
        """
        # 1. Load data
        print("Loading dataset...")
        corpus, queries, qrels = self.data_loader.load_data()
        
        # 2. Knowledge Distillation
        print("Starting knowledge distillation...")
        student_model = self.trainer.distill(
            teacher_model=self.config["teacher_model"],
            student_model=self.config["student_model"],
            queries=queries,
            qrels_train=qrels["train"]
        )
        
        # 3. Re-embed Corpus with distilled model
        print("Re-embedding corpus with student model...")
        corpus_embeddings = student_model.encode(list(corpus.values()))
        self.data_loader.update_qdrant(
            embeddings=corpus_embeddings,
            collection_name=self.config["qdrant"]["collection"]
        )
        
        # 4. Evaluate performance
        print("Evaluating model performance...")
        metrics = self.evaluator.evaluate(
            model=student_model,
            queries_test=qrels["test"]
        )
        
        return metrics

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = KDRetriever("config/scifact_config.yaml")
    results = pipeline.run_pipeline()
    print(f"Final Metrics: {results}")
