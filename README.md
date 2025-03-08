# Star Wars RAG Application

A Retrieval-Augmented Generation (RAG) application that scrapes, chunks, embeds, searches, and performs inference using a local LLM or an API.

## Features
- **Web Scraping**: Scrapes data from the Star Wars Wiki.
- **Chunking**: Processes scraped data into manageable text chunks.
- **Embeddings**: Generates vector embeddings for chunked documents.
- **Search**: Searches relevant documents using vector similarity.
- **Inference**: Answers questions using a local LLM or an API (OpenAI/Anthropic).
- **Interactive CLI**: Allows users to perform all operations via `app/cli.py`.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)
- API keys for OpenAI/Anthropic (if using external LLMs)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/starwars-rag.git
   cd starwars-rag
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependecies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables: Create a .env file and add your API keys:
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    ```
### Usage

Run the CLI tool to perform different operations:

1. Scrape Data
    ```bash
    python app/cli.py scrape --start-url "https://starwars.fandom.com/wiki/Main_Page" --max-pages 100 --output-dir data/raw
    ```

2. Process Data into Chunks
    ```bash
    python app/cli.py process --input-dir data/raw --output-dir data/chunks --chunk-size 512 --overlap 128
    ```

3. Generate Embeddings
    ```bash
    python app/cli.py embed --input-dir data/chunks --output-dir data/embeddings --model llama3.2 --batch-size 32
    ```

4. Interactive Mode
    ```bash
    python app/cli.py interactive --embeddings data/embeddings/embeddings.pkl --provider local --model llama3.2 --top-k 5
    ```

5. Ask a Single Question
    ```bash
    python app/cli.py question "Who is Darth Vader?" --embeddings data/embeddings --provider openai --model gpt-4
    ```

6. Perform a Search
    ```bash
    python app/cli.py search "Jedi Order" --embeddings data/embeddings --top-k 5
    ```

### Directory Structure

```
starwars-rag/
│-- app/
│   ├── cli.py  # Command-line interface
│   ├── src/
│       ├── scraper/  # Web scraper module
│       ├── processor/  # Chunking module
│       ├── embeddings/  # Embedding generator
│       ├── rag/  # RAG engine
│       ├── utils/  # Helper functions
│-- data/
│   ├── raw/  # Scraped data
│   ├── chunks/  # Processed chunks
│   ├── embeddings/  # Embedding vectors
│-- requirements.txt  # Dependencies
│-- README.md  # Documentation
```