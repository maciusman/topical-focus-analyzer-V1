import requests
from bs4 import BeautifulSoup
import re
import time
import random
from urllib.parse import urlparse
import concurrent.futures

def extract_main_content(url, timeout=30, user_agent=None):
    """
    Extract the main content from a webpage using BeautifulSoup.
    
    Args:
        url (str): URL to extract content from
        timeout (int): Request timeout in seconds
        user_agent (str, optional): Custom user agent string
        
    Returns:
        str: Extracted main content text or empty string if extraction failed
    """
    # Define a list of user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    
    # Use provided user agent or randomly select one
    headers = {
        'User-Agent': user_agent if user_agent else random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # First try to use trafilatura if available
        try:
            import trafilatura
            extracted_text = trafilatura.extract(response.text, include_comments=False, 
                                              include_tables=False, no_fallback=False)
            if extracted_text:
                return clean_text(extracted_text)
        except ImportError:
            # If trafilatura is not available, continue with BeautifulSoup
            pass
        
        # Fallback to BeautifulSoup
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove script, style, nav, header, footer elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content by common patterns
        main_content = None
        
        # Look for article or main tags first
        main_content = soup.find('article') or soup.find('main')
        
        # If not found, look for common content div patterns
        if not main_content:
            for div in soup.find_all('div', {'id': re.compile(r'content|main|post', re.I)}):
                main_content = div
                break
                
        # If still not found, look for divs with 'content' or 'article' in class names
        if not main_content:
            for div in soup.find_all('div', {'class': re.compile(r'content|article|post', re.I)}):
                main_content = div
                break
        
        # If nothing was found, use the body tag
        if not main_content:
            main_content = soup.body
        
        # Extract text from the main content
        extracted_text = main_content.get_text(separator=' ', strip=True) if main_content else ''
        
        # Clean the extracted text
        return clean_text(extracted_text)
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def clean_text(text):
    """
    Clean the extracted text by removing extra whitespace, URLs, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_text_for_analysis(text):
    """
    Preprocess text for vectorization - simplified version without Spacy.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common English stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 'when', 'where', 'how', 'to', 'for', 'with', 'by', 'about', 'against', 
                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
                 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 
                 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
                 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return " ".join(filtered_words)

def batch_extract_content(urls, max_workers=5, delay=1.0):
    """
    Extract content from multiple URLs in parallel with rate limiting.
    
    Args:
        urls (list): List of URLs to extract content from
        max_workers (int): Maximum number of parallel workers
        delay (float): Delay between requests in seconds to avoid rate limiting
        
    Returns:
        dict: Dictionary mapping URLs to extracted content
    """
    results = {}
    domains_last_request = {}  # Track last request time per domain
    
    def extract_with_rate_limit(url):
        # Extract domain to apply rate limiting per domain
        domain = urlparse(url).netloc
        
        # Check if we need to wait before making another request to this domain
        if domain in domains_last_request:
            elapsed = time.time() - domains_last_request[domain]
            if elapsed < delay:
                time.sleep(delay - elapsed)
        
        # Extract content
        content = extract_main_content(url)
        
        # Update last request time for this domain
        domains_last_request[domain] = time.time()
        
        return url, content
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_with_rate_limit, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                url, content = future.result()
                results[url] = content
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results[url] = ""
    
    return results

# Test function
def test_content_extractor():
    test_urls = [
        "https://www.python.org/about/",
        "https://en.wikipedia.org/wiki/Web_scraping",
        "https://www.bbc.com/news"
    ]
    
    for url in test_urls:
        print(f"\nTesting content extraction for: {url}")
        content = extract_main_content(url)
        
        # Print a preview
        preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Extracted content ({len(content)} characters):")
        print(preview)
        
        # Test preprocessing
        processed = preprocess_text_for_analysis(content[:1000])
        print("\nPreprocessed content:")
        print(processed[:200] + "..." if len(processed) > 200 else processed)

if __name__ == "__main__":
    test_content_extractor()