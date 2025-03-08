import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import re
import os
import json
import pickle
from tqdm import tqdm

def scrape_starwars_wiki(start_url, max_pages=100, output_dir='data/raw', delay=1):
    """
    Scrapes the Star Wars wiki starting from the given URL.
    
    Args:
        start_url (str): URL to start scraping from
        max_pages (int): Maximum number of pages to scrape
        output_dir (str): Directory to save raw HTML files
        delay (float): Delay between requests in seconds
        
    Returns:
        list: List of dictionaries containing document information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load robots.txt to respect the site's rules
    robots_url = "https://starwars.fandom.com/robots.txt"
    try:
        robots_response = requests.get(robots_url)
        if robots_response.status_code == 200:
            # Simple check if we're allowed to scrape
            if "Disallow: /wiki/" in robots_response.text:
                print("Warning: The robots.txt file disallows scraping /wiki/ pages. Consider using the site's API instead.")
    except Exception as e:
        print(f"Could not check robots.txt: {e}")
    
    visited_urls = set()
    to_visit = [start_url]
    documents = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; StarWarsResearchBot/1.0; +https://example.org/bot)',
    }
    
    pbar = tqdm(total=max_pages, desc="Scraping pages")
    
    while to_visit and len(visited_urls) < max_pages:
        url = to_visit.pop(0)
        if url in visited_urls:
            continue
            
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status code {response.status_code}")
                continue
                
            visited_urls.add(url)
            pbar.update(1)
            
            # Save raw HTML
            page_id = url.split("/wiki/")[-1]
            safe_filename = re.sub(r'[^\w\-_\.]', '_', page_id)
            with open(f"{output_dir}/{safe_filename}.html", 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content from main article area
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            if not content_div:
                continue
                
            # Skip navigation and info boxes
            paragraphs = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol'])
            content = []
            current_section = "Introduction"
            
            for element in paragraphs:
                if element.name.startswith('h'):
                    current_section = element.get_text().strip()
                    continue
                    
                text = element.get_text()
                # Clean the text
                text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
                text = re.sub(r'\s+', ' ', text).strip()
                
                if text:
                    content.append({
                        "section": current_section,
                        "text": text
                    })
            
            if content:
                title = soup.find('h1', {'id': 'firstHeading'}).get_text()
                document = {
                    'title': title,
                    'url': url,
                    'content': content,
                    'html_file': f"{safe_filename}.html",
                    'categories': [tag.get_text() for tag in soup.find_all('li', {'class': 'category normal'})]
                }
                documents.append(document)
                
                # Save each document separately for resilience
                with open(f"{output_dir}/{safe_filename}.json", 'w', encoding='utf-8') as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)
            
            # Find more links to visit
            if len(to_visit) + len(visited_urls) < max_pages * 2:  # Buffer of links to choose from
                links = content_div.find_all('a')
                for link in links:
                    href = link.get('href')
                    if href and href.startswith('/wiki/') and ':' not in href:
                        # Skip special pages, categories, etc.
                        if any(x in href for x in ['Special:', 'File:', 'Category:', 'Help:', 'Talk:']):
                            continue
                        full_url = urllib.parse.urljoin('https://starwars.fandom.com', href)
                        if full_url not in visited_urls and full_url not in to_visit:
                            to_visit.append(full_url)
            
            # Be respectful of the server
            time.sleep(delay)
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    
    pbar.close()
    
    # Save the full document collection
    with open(f"{output_dir}/all_documents.pkl", 'wb') as f:
        pickle.dump(documents, f)
        
    print(f"Scraped {len(documents)} documents from {len(visited_urls)} pages")
    return documents

def load_documents(input_dir='data/raw'):
    """
    Load documents from pickle or from individual JSON files
    """
    pkl_path = f"{input_dir}/all_documents.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    # If pickle doesn't exist, try to load from JSON files
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            try:
                with open(f"{input_dir}/{filename}", 'r', encoding='utf-8') as f:
                    documents.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return documents

if __name__ == "__main__":
    # Example usage
    docs = scrape_starwars_wiki(
        'https://starwars.fandom.com/wiki/Main_Page',
        max_pages=10,
        output_dir='../../data/raw',
        delay=1
    )
    print(f"Scraped {len(docs)} documents")
