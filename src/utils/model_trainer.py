import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import numpy as np
from torch.utils.data import DataLoader, Dataset

class KnowledgeDistillationDataset(Dataset):
    """Dataset for knowledge distillation training"""
    
    def __init__(self, queries: List[str], passages: List[str]):
        self.queries = queries
        self.passages = passages
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx], self.passages[idx]

class KDTrainer:
    """
    Knowledge Distillation Trainer for Information Retrieval
    """
    
    def __init__(self, config: Dict):
        """
        Initialize KD trainer with configuration
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def distill(self, teacher_model: str, student_model: str, 
                queries: Dict, qrels_train: Dict) -> SentenceTransformer:
        """
        Perform knowledge distillation from teacher to student model
        
        Args:
            teacher_model (str): Name/path of teacher model
            student_model (str): Name/path of student model
            queries (Dict): Query dictionary
            qrels_train (Dict): Training qrels
            
        Returns:
            SentenceTransformer: Distilled student model
        """
        print(f"Loading teacher model: {teacher_model}")
        teacher = SentenceTransformer(teacher_model).to(self.device)
        
        print(f"Loading student model: {student_model}")
        student = SentenceTransformer(student_model).to(self.device)
        
        # Prepare training data
        train_queries, train_passages = self._prepare_training_data(queries, qrels_train)
        
        # Create dataset and dataloader
        dataset = KnowledgeDistillationDataset(train_queries, train_passages)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config["training"]["batch_size"], 
            shuffle=True
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            student.parameters(), 
            lr=self.config["training"]["learning_rate"]
        )
        
        # Training loop
        student.train()
        teacher.eval()
        
        for epoch in range(self.config["training"]["num_epochs"]):
            total_loss = 0
            
            for batch_queries, batch_passages in dataloader:
                optimizer.zero_grad()
                
                # Get teacher and student embeddings
                with torch.no_grad():
                    teacher_embeddings = teacher.encode(
                        batch_queries, convert_to_tensor=True
                    )
                
                student_embeddings = student.encode(
                    batch_queries, convert_to_tensor=True
                )
                
                # Calculate distillation loss
                loss = self._distillation_loss(
                    student_embeddings, 
                    teacher_embeddings
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}, "
                  f"Average Loss: {avg_loss:.4f}")
        
        return student
    
    def _prepare_training_data(self, queries: Dict, qrels_train: Dict):
        """Prepare training data from queries and qrels"""
        train_queries = []
        train_passages = []
        
        # Implementation would extract relevant query-passage pairs
        # from the training data
        
        return train_queries, train_passages
    
    def _distillation_loss(self, student_embeddings: torch.Tensor, 
                          teacher_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate knowledge distillation loss
        
        Args:
            student_embeddings: Student model embeddings
            teacher_embeddings: Teacher model embeddings
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # MSE loss between teacher and student embeddings
        mse_loss = nn.MSELoss()
        return mse_loss(student_embeddings, teacher_embeddings)
