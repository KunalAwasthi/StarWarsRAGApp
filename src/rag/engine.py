import os
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import ollama  # For running Ollama models

# Import from other modules
from src.embeddings.vectorizer import Vectorizer
from src.retriever.search import Retriever

# Optional local LLM imports

class StarWarsRAG:
    def __init__(self, 
                 embeddings_path='data/embeddings/starwars_embeddings.pkl',
                 model_name='all-MiniLM-L6-v2',
                 llm_provider='openai',
                 llm_model='gpt-3.5-turbo',
                 api_key=None,
                 use_local_model=False,
                 local_model_path="models/llama-3-8b.Q4_K_M.gguf"):
        """
        Initialize the Star Wars RAG engine.
        
        Args:
            embeddings_path (str): Path to embeddings file
            model_name (str): Name of the embedding model
            llm_provider (str): LLM provider ('openai', 'anthropic', or 'local')
            llm_model (str): LLM model name
            api_key (str, optional): API key for the LLM provider
            use_local_model (bool): Whether to use a local LLM instead of an API
            local_model_path (str): Path to local GGUF model if using llama-cpp
        """
        self.retriever = Retriever(embedding_file=embeddings_path)
        self.vectorizer = Vectorizer(model_name)
        
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        
        if api_key and not use_local_model:
            if llm_provider == 'openai':
                import openai
                openai.api_key = api_key
            elif llm_provider == 'anthropic':
                os.environ['ANTHROPIC_API_KEY'] = api_key
        else:
            if llm_provider == 'openai':
                import openai
                openai.api_key = os.environ.get('OPENAI_API_KEY')

        # Initialize local LLM if required
        self.local_llm = None
        if use_local_model:
            if ollama:
                self.local_llm = "ollama"  # Use Ollama for inference
                
        self.conversation_history = []

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant content."""
        query_embedding = self.vectorizer.embed_text(query)
        results = self.retriever.search(query_embedding, top_k=top_k)
        return results

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM."""
        context_parts = []
        for i, result in enumerate(results):
            metadata = result['metadata']
            text = result['text']
            context_part = f"[Document {i+1}] {metadata['title'] if 'title' in metadata else ''}"
            if 'section' in metadata and metadata['section']:
                context_part += f" - {metadata['section']}"
            context_part += f"\nURL: {text}\n"
            context_parts.append(context_part)
        print(f'Context parts: {context_parts}')
        return "\n---\n".join(context_parts)

    def answer_question(self, 
                        question: str, 
                        top_k: int = 5, 
                        temperature: float = 0.3,
                        max_tokens: int = 500,
                        include_sources: bool = True,
                        use_history: bool = False) -> Dict[str, Any]:
        """Answer a question using RAG."""
        search_results = self.search(question, top_k=top_k)
        print(search_results)
        context = self.format_context(search_results)
        print(context)

        current_date = datetime.now().strftime("%Y-%m-%d")
        system_prompt = f"""You are an expert on Star Wars lore, helping to answer questions based ONLY on the provided context.
Today's date is {current_date}.
If the context doesn't contain enough information to answer the question, admit that you don't know rather than making up information.
Always stick to the facts presented in the context and avoid making up details not present in the provided information.
"""

        if include_sources:
            system_prompt += "Include citations to the sources you used in your answer by referring to the document numbers."

        user_prompt = f"""Question: {question}

Context:
{context}

Answer the question based only on the provided context. Be comprehensive but stay focused on what was asked."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        # Local LLM Inference
        if self.use_local_model:
            return self.generate_with_local_llm(user_prompt, max_tokens)

        # API-based Inference
        return self.generate_with_api_llm(messages, temperature, max_tokens)

    def generate_with_local_llm(self, prompt: str, max_tokens: int):
        """Generate a response using a local LLM (llama.cpp or Ollama)."""
        if self.local_llm == "ollama":
            response = ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": prompt}])
            answer = response["message"]["content"]
        else:
            raise RuntimeError("Local LLM is not initialized correctly.")
        
        return {"answer": answer}

    def generate_with_api_llm(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
        """Generate a response using OpenAI or Anthropic API."""
        if self.llm_provider == 'openai':
            import openai
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response["choices"][0]["message"]["content"].strip()

        elif self.llm_provider == 'anthropic':
            import anthropic
            client = anthropic.Client()
            response = client.messages.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response["content"].strip()
        else:
            raise ValueError("Invalid LLM provider.")

        return {"answer": answer}
