import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

class Vectorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        """
        Initialize a vectorizer for creating text embeddings.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            device (str): Device to use for computation ('cpu', 'cuda', etc.)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
    def embed_text(self, text):
        """
        Create an embedding for a single text.
        
        Args:
            text (str): The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        return self.model.encode(text)
    
    def embed_batch(self, texts, batch_size=32):
        """
        Create embeddings for a batch of texts.
        
        Args:
            texts (list): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2', output_dir='data/embeddings', batch_size=32):
    """
    Create embeddings for text chunks.
    
    Args:
        chunks (list): List of chunk dictionaries
        model_name (str): Name of the sentence transformer model to use
        output_dir (str): Directory to save embeddings
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (embeddings, texts, metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    vectorizer = Vectorizer(model_name)
    
    texts = [chunk['text'] for chunk in chunks]
    metadata = [chunk['metadata'] for chunk in chunks if 'metadata' in chunk]
    
    embeddings = vectorizer.embed_batch(texts, batch_size=batch_size)
    
    with open(os.path.join(output_dir, 'embeddings.pkl'), 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'texts': texts, 'metadata': metadata}, f)
    
    return embeddings, texts, metadata
