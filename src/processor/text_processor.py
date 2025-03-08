import os
import pickle
import json
import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spacy model (optional)
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Spacy model not found. Using NLTK for sentence tokenization.")
    nlp = None

def clean_text(text):
    """
    Clean the text by removing excessive whitespace, citations, etc.
    """
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    return text

def split_into_sentences(text):
    """
    Split text into sentences using NLTK or spaCy
    """
    if nlp:
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    else:
        return sent_tokenize(text)

def chunk_documents(documents, chunk_size=500, overlap=50, output_dir='data/processed'):
    """
    Process documents and split them into overlapping chunks of text.
    
    Args:
        documents (list): List of document dictionaries
        chunk_size (int): Target size of chunks in words
        overlap (int): Number of words to overlap between chunks
        output_dir (str): Directory to save processed chunks
        
    Returns:
        list: List of chunk dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    
    for doc_idx, doc in enumerate(tqdm(documents, desc="Processing documents")):
        title = doc['title']
        url = doc['url']
        
        # Process each content section
        for section_idx, section in enumerate(doc['content']):
            section_title = section['section']
            text = clean_text(section['text'])
            
            if not text:
                continue
                
            # Split into sentences
            sentences = split_into_sentences(text)
            
            current_chunk = []
            current_length = 0
            chunk_id = 0
            
            for sentence in sentences:
                words = sentence.split()
                sentence_length = len(words)
                
                # If adding this sentence would exceed chunk size and we already have content
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Store the current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_id_str = f"{doc_idx:04d}_{section_idx:02d}_{chunk_id:02d}"
                    
                    chunk = {
                        'id': chunk_id_str,
                        'title': title,
                        'section': section_title,
                        'text': chunk_text,
                        'url': url,
                        'doc_id': doc_idx,
                        'section_id': section_idx,
                        'chunk_id': chunk_id
                    }
                    
                    chunks.append(chunk)
                    
                    # Save individual chunk
                    with open(f"{output_dir}/chunk_{chunk_id_str}.json", 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    # Start a new chunk with overlap
                    overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                    current_chunk = overlap_tokens
                    current_length = len(' '.join(current_chunk).split())
                    chunk_id += 1
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id_str = f"{doc_idx:04d}_{section_idx:02d}_{chunk_id:02d}"
                
                chunk = {
                    'id': chunk_id_str,
                    'title': title,
                    'section': section_title,
                    'text': chunk_text,
                    'url': url,
                    'doc_id': doc_idx,
                    'section_id': section_idx,
                    'chunk_id': chunk_id
                }
                
                chunks.append(chunk)
                
                # Save individual chunk
                with open(f"{output_dir}/chunk_{chunk_id_str}.json", 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    # Save all chunks
    with open(f"{output_dir}/all_chunks.pkl", 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def load_chunks(input_dir='data/processed'):
    """
    Load chunks from pickle or individual JSON files
    """
    pkl_path = f"{input_dir}/all_chunks.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    # If pickle doesn't exist, try to load from JSON files
    chunks = []
    for filename in os.listdir(input_dir):
        if filename.startswith('chunk_') and filename.endswith('.json'):
            try:
                with open(f"{input_dir}/{filename}", 'r', encoding='utf-8') as f:
                    chunks.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return chunks

if __name__ == "__main__":
    from src.scraper.scraper import load_documents
    
    # Example usage
    documents = load_documents('../../data/raw')
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=50,
        output_dir='../../data/processed'
    )
    print(f"Created {len(chunks)} chunks")
