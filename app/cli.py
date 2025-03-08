import argparse
import os
import sys
import time
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scraper.scraper import scrape_starwars_wiki, load_documents
from src.processor.text_processor import chunk_documents, load_chunks
from src.embeddings.vectorizer import create_embeddings
from src.rag.engine import StarWarsRAG
from src.utils.helpers import format_search_results, timed_operation

# Load environment variables
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from environment variables or prompt user"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def get_anthropic_api_key():
    """Get Anthropic API key from environment variables or prompt user"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic API key not found in environment variables.")
        api_key = input("Please enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = api_key
    return api_key

@timed_operation
def run_scraper(args):
    """Run the scraper"""
    scrape_starwars_wiki(
        args.start_url,
        max_pages=args.max_pages,
        output_dir=args.output_dir,
        delay=args.delay
    )

@timed_operation
def run_processor(args):
    """Process documents into chunks"""
    documents = load_documents(args.input_dir)
    chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output_dir
    )

@timed_operation
def run_embeddings(args):
    """Create embeddings for chunks"""
    chunks = load_chunks(args.input_dir)
    create_embeddings(
        chunks,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

@timed_operation
def run_interactive(args):
    """Run interactive mode"""
    # Set up API key for the chosen provider
    if args.provider == 'openai':
        api_key = get_openai_api_key()
    elif args.provider == 'anthropic':
        api_key = get_anthropic_api_key()
    else:
        api_key = None
    
    # Initialize RAG engine
    rag = StarWarsRAG(
        embeddings_path=args.embeddings,
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=api_key,
        use_local_model=args.use_local_llm,
    )
    
    # Interactive loop
    print("\n" + "=" * 50)
    print("Star Wars Knowledge Base - Interactive Mode")
    print("Type 'exit', 'quit', or 'q' to exit")
    print("Type 'sources' to toggle showing sources")
    print("Type 'new' to start a new conversation")
    print("=" * 50 + "\n")
    
    show_sources = True
    use_history = args.use_history
    
    while True:
        try:
            question = input("\nAsk a question about Star Wars: ")
            question = question.strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            if question.lower() == 'sources':
                show_sources = not show_sources
                print(f"Source display is now {'ON' if show_sources else 'OFF'}")
                continue
                
            if question.lower() == 'new':
                print("Starting a new conversation")
                rag.clear_history()
                continue
                
            if not question:
                continue
            
            # Show a spinner while processing
            spinner = tqdm(total=0, desc="Thinking", bar_format="{desc}")
            
            try:
                # Answer the question
                result = rag.answer_question(
                    question,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    include_sources=show_sources,
                    use_history=use_history
                )
                
                # Print the answer
                spinner.close()
                print("\nAnswer:")
                print(result['answer'])
                
                # Print sources if enabled
                if result and show_sources and 'sources' in result:
                    print("\nSources:")
                    for i, source in enumerate(result['sources']):
                        print(f"{i+1}. {source['title']} ({source['score']:.4f})")
                        print(f"   URL: {source['url']}")
            
            except Exception as e:
                spinner.close()
                logging.error(f"Error: {e}", exc_info=True)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

@timed_operation
def run_question(args):
    """Answer a single question"""
    # Set up API key for the chosen provider
    if args.provider == 'openai':
        api_key = get_openai_api_key()
    elif args.provider == 'anthropic':
        api_key = get_anthropic_api_key()
    else:
        api_key = None
    
    # Initialize RAG engine
    rag = StarWarsRAG(
        embeddings_path=args.embeddings,
        llm_provider=args.provider,
        llm_model=args.model,
        api_key=api_key
    )
    
    # Answer the question
    result = rag.answer_question(
        args.question,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_sources=True
    )
    
    # Print the answer
    print("\nQuestion:", args.question)
    print("\nAnswer:")
    print(result['answer'])
    
    # Print sources
    if result['sources']:
        print("\nSources:")
        for i, source in enumerate(result['sources']):
            print(f"{i+1}. {source['title']} ({source['score']:.4f})")
            print(f"   URL: {source['url']}")

@timed_operation
def run_search(args):
    """Run a search without using LLM"""
    # Initialize RAG engine
    rag = StarWarsRAG(
        embeddings_path=args.embeddings
    )
    
    # Search
    results = rag.search(args.query, top_k=args.top_k)
    
    # Print results
    print("\nSearch results for:", args.query)
    print(format_search_results(results))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Star Wars RAG Application')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scraper command
    scraper_parser = subparsers.add_parser('scrape', help='Scrape Star Wars Wiki')
    scraper_parser.add_argument('--start-url', type=str, default='https://starwars.fandom.com/wiki/Main_Page',
                           help='URL to start scraping from')
    scraper_parser.add_argument('--max-pages', type=int, default=100,
                           help='Maximum number of pages to scrape')
    scraper_parser.add_argument('--output-dir', type=str, default='data/raw',
                           help='Directory to save scraped data')
    scraper_parser.add_argument('--delay', type=float, default=1,
                           help='Delay between requests in seconds')
    scraper_parser.set_defaults(func=run_scraper)
    
    # Processor command
    processor_parser = subparsers.add_parser('process', help='Process scraped documents into chunks')
    processor_parser.add_argument('--input-dir', type=str, default='data/raw',
                                  help='Directory containing raw scraped data')
    processor_parser.add_argument('--output-dir', type=str, default='data/chunks',
                                  help='Directory to save processed chunks')
    processor_parser.add_argument('--chunk-size', type=int, default=512,
                                  help='Chunk size for processing text')
    processor_parser.add_argument('--overlap', type=int, default=128,
                                  help='Overlap between chunks')
    processor_parser.set_defaults(func=run_processor)
    
    # Embeddings command
    embeddings_parser = subparsers.add_parser('embed', help='Generate embeddings for document chunks')
    embeddings_parser.add_argument('--input-dir', type=str, default='data/chunks',
                                   help='Directory containing processed chunks')
    embeddings_parser.add_argument('--output-dir', type=str, default='data/embeddings',
                                   help='Directory to save embeddings')
    embeddings_parser.add_argument('--model', type=str, default='llama3.2',
                                   help='Embedding model to use')
    embeddings_parser.add_argument('--batch-size', type=int, default=32,
                                   help='Batch size for processing embeddings')
    embeddings_parser.set_defaults(func=run_embeddings)
    
    # Interactive mode command
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive question-answering mode')
    interactive_parser.add_argument('--embeddings', type=str, default='data/embeddings/embeddings.pkl',
                                    help='Path to embeddings file')
    interactive_parser.add_argument('--provider', type=str, choices=['openai', 'anthropic', 'local'], default='local',
                                    help='LLM provider')
    interactive_parser.add_argument('--model', type=str, default='llama3.2',
                                    help='LLM model to use')
    interactive_parser.add_argument('--top-k', type=int, default=5,
                                    help='Number of top results to consider')
    interactive_parser.add_argument('--temperature', type=float, default=0.7,
                                    help='LLM temperature setting')
    interactive_parser.add_argument('--max-tokens', type=int, default=512,
                                    help='Maximum tokens in response')
    interactive_parser.add_argument('--use-history', action='store_true',
                                    help='Use chat history for context')
    interactive_parser.add_argument('--use-local-llm', action='store_true', default=True,
                                    help='Use local LLM')
    interactive_parser.set_defaults(func=run_interactive)
    
    # Single question command
    question_parser = subparsers.add_parser('question', help='Ask a single question')
    question_parser.add_argument('question', type=str, help='Question to ask')
    question_parser.add_argument('--embeddings', type=str, default='data/embeddings',
                                 help='Path to embeddings file')
    question_parser.add_argument('--provider', type=str, choices=['openai', 'anthropic'], default='openai',
                                 help='LLM provider')
    question_parser.add_argument('--model', type=str, default='gpt-4',
                                 help='LLM model to use')
    question_parser.add_argument('--top-k', type=int, default=5,
                                 help='Number of top results to consider')
    question_parser.add_argument('--temperature', type=float, default=0.7,
                                 help='LLM temperature setting')
    question_parser.add_argument('--max-tokens', type=int, default=512,
                                 help='Maximum tokens in response')
    question_parser.set_defaults(func=run_question)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Perform a search without LLM')
    search_parser.add_argument('query', type=str, help='Query to search for')
    search_parser.add_argument('--embeddings', type=str, default='data/embeddings',
                               help='Path to embeddings file')
    search_parser.add_argument('--top-k', type=int, default=5,
                               help='Number of top results to return')
    search_parser.set_defaults(func=run_search)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
