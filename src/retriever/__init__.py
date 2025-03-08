import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple

class Retriever:
    def __init__(self, embeddings=None, texts=None, metadata=None, embedding_file=None):
        """
        Initialize a retriever for searching embeddings.
        
        Args:
            embeddings (numpy.ndarray, optional): Matrix of embeddings
            texts (list, optional): List of texts corresponding to embeddings
            metadata (list, optional): List of metadata dictionaries
            embedding_file (str, optional): Path to file containing embeddings
        """
        if embedding_file:
            with open(embedding_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.model_name = data.get('model_name', 'unknown')
        else:
            self.embeddings = embeddings
            self.texts = texts
            self.metadata = metadata
            self.model_name = 'unknown'
        
        # Create FAISS index for fast similarity search
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product is cosine sim for normalized vectors
        
        # Normalize vectors for cosine similarity
        self.normalized_embeddings = self._normalize_vectors(self.embeddings)
        self.index.add(self.normalized_embeddings)
        
    def _normalize_vectors(self, vectors):
        """Normalize vectors to unit length for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def search(self, query_vector, top_k=5) -> List[Dict[str, Any]]:
        """
        Search for most similar texts to the query vector.
        
        Args:
            query_vector (numpy.ndarray): Query embedding vector
            top_k (int): Number of results to return
            
        Returns:
            list: List of result dictionaries
        """
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1)
        query_vector = self._normalize_vectors(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.texts) or idx < 0:
                continue
                
            results.append({
                'score': float(scores[0][i]),
                'text': self.texts[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def save(self, output_file='data/embeddings/retriever.pkl'):
        """Save the retriever to a file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # We can't directly pickle FAISS index, so we'll save the components
        save_data = {
            'embeddings': self.embeddings,
            'texts': self.texts,
            'metadata': self.metadata,
            'model_name': self.model_name
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load(cls, input_file='data/embeddings/retriever.pkl'):
        """Load a retriever from a file"""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            embeddings=data['embeddings'],
            texts=data['texts'],
            metadata=data['metadata']
        )
