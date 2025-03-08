import os
import pickle
import json
import logging
from typing import List, Dict, Any
import time
import hashlib
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data, filepath):
    """Save data as JSON to a file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath):
    """Load JSON data from a file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(data, filepath):
    """Save data as pickle to a file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Load pickled data from a file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_cache_key(text):
    """Generate a cache key from text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def rate_limit_function(func, rate_limit=1):
    """
    Rate limit a function to be called at most once per rate_limit seconds.
    
    Args:
        func: The function to rate limit
        rate_limit: Minimum seconds between calls
        
    Returns:
        Wrapped function
    """
    last_called = [0]  # Use list to make it mutable in closure
    
    def wrapper(*args, **kwargs):
        now = time.time()
        elapsed = now - last_called[0]
        
        if elapsed < rate_limit:
            time.sleep(rate_limit - elapsed)
            
        result = func(*args, **kwargs)
        last_called[0] = time.time()
        return result
        
    return wrapper

def truncate_text(text, max_length=100, ellipsis="..."):
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(ellipsis)] + ellipsis

def clean_filename(filename):
    """Clean a string to be used as a filename."""
    # Replace spaces and keep only alphanumeric characters, dashes, and underscores
    return re.sub(r'[^\w\-_\.]', '_', filename)

def format_search_results(results: List[Dict[str, Any]], max_length=200) -> str:
    """Format search results for display."""
    output = []
    
    for i, result in enumerate(results):
        score = result.get('score', 0) * 100  # Convert to percentage
        metadata = result.get('metadata', {})
        title = metadata.get('title', 'Untitled')
        url = metadata.get('url', '')
        
        text = result.get('text', '')
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        output.append(f"{i+1}. {title} ({score:.1f}% match)")
        output.append(f"   URL: {url}")
        output.append(f"   {text}")
        output.append("")
        
    return "\n".join(output)

def timed_operation(func):
    """Decorator to time an operation."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

if __name__ == "__main__":
    # Example usage
    example_data = {"key": "value", "nested": {"key": "value"}}
    
    # Test save and load functions
    temp_dir = "../../temp"
    ensure_dir(temp_dir)
    
    json_path = f"{temp_dir}/test.json"
    save_json(example_data, json_path)
    loaded_json = load_json(json_path)
    print("JSON data loaded:", loaded_json)
    
    pickle_path = f"{temp_dir}/test.pkl"
    save_pickle(example_data, pickle_path)
    loaded_pickle = load_pickle(pickle_path)
    print("Pickle data loaded:", loaded_pickle)
