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
                'metadata': self.metadata[idx] if idx in self.metadata else ''
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

# Alternative implementation using chromadb (uncomment if you prefer)
"""
import chromadb
from chromadb.utils import embedding_functions

class ChromaRetriever:
    def __init__(self, collection_name="starwars_knowledge", embedding_model="all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, chunks):
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{
            'title': chunk['title'],
            'section': chunk['section'],
            'url': chunk['url']
        } for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Added {len(ids)} documents to ChromaDB collection")
    
    def search(self, query, top_k=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
"""

if __name__ == "__main__":
    from src.embeddings.vectorizer import load_embeddings, Vectorizer
    
    # Example usage
    embeddings, texts, metadata, model_name = load_embeddings('../../data/embeddings/starwars_embeddings.pkl')
    
    retriever = Retriever(embeddings, texts, metadata)
    
    # Create a query embedding
    vectorizer = Vectorizer(model_name)
    query = "Who is Darth Vader?"
    query_embedding = vectorizer.embed_text(query)
    
    # Search
    results = retriever.search(query_embedding, top_k=5)
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (score: {result['score']:.4f}):")
        print(f"Title: {result['metadata']['title']}")
        print(f"Section: {result['metadata']['section']}")
        print(f"URL: {result['metadata']['url']}")
        print(f"Text snippet: {result['text'][:200]}...")
        print("-" * 50)
