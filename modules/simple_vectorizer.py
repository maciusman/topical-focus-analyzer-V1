from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def process_url_path(url):
    """
    Process a URL to extract and clean the path.
    
    Args:
        url (str): URL to process
        
    Returns:
        str: Cleaned path
    """
    try:
        # Parse URL and get path
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Clean the path
        path = path.strip('/')  # Remove leading/trailing slashes
        
        # Replace hyphens, underscores, and other separators with spaces
        path = re.sub(r'[-_/+]', ' ', path)
        
        # Remove common file extensions
        path = re.sub(r'\.(html|htm|php|aspx|jsp|asp)$', '', path)
        
        # Convert to lowercase
        path = path.lower()
        
        return path
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return ""

def vectorize_urls_and_content(urls, content_dict=None, use_url_paths=True, use_content=True, url_weight=0.3):
    """
    Vectorize both URL paths and content with weighted combination.
    
    Args:
        urls (list): List of URLs to vectorize
        content_dict (dict, optional): Pre-extracted content dictionary (URL -> content)
        use_url_paths (bool): Whether to use URL paths in the analysis
        use_content (bool): Whether to use page content in the analysis
        url_weight (float): Weight to give URL vectors vs content (0-1)
    
    Returns:
        tuple: (URLs, paths, contents, vectorizer, matrix)
    """
    # Process URL paths
    processed_paths = []
    for url in urls:
        path = process_url_path(url)
        processed_paths.append(path)
    
    # Prepare content if needed
    if use_content and content_dict:
        # Import the preprocessor function
        from modules.content_extractor import preprocess_text_for_analysis
        
        # Ensure all URLs have a content entry (even if empty)
        processed_contents = []
        for url in urls:
            if url in content_dict:
                # Preprocess the content
                processed_content = preprocess_text_for_analysis(content_dict[url])
                processed_contents.append(processed_content)
            else:
                processed_contents.append("")
    else:
        # Create empty content list if not using content
        processed_contents = [""] * len(urls)
        
    # Create vectors based on what we're using
    if use_url_paths and not use_content:
        # Only use URL paths
        vectorizer = TfidfVectorizer(
            min_df=1,           # Include terms that appear in at least 1 document
            max_df=0.95,        # Exclude terms that appear in >95% of documents
            stop_words='english',  # Remove English stop words
            ngram_range=(1, 3), # Use unigrams, bigrams, and trigrams
            token_pattern=r'(?u)\b\w+\b',  # Consider single characters as tokens
            use_idf=True,       # Use inverse document frequency weighting
            smooth_idf=True,    # Add 1 to document frequencies to avoid division by zero
            sublinear_tf=True   # Apply sublinear tf scaling (1+log(tf))
        )
        
        # Handle empty corpus case
        if not any(processed_paths):
            print("Warning: All processed paths are empty. Creating dummy matrix.")
            matrix = np.zeros((len(urls), 1))
            return urls, processed_paths, processed_contents, vectorizer, matrix
        
        # Fit and transform the processed paths
        matrix = vectorizer.fit_transform(processed_paths)
        return urls, processed_paths, processed_contents, vectorizer, matrix
        
    elif not use_url_paths and use_content:
        # Only use content
        content_vectorizer = TfidfVectorizer(
            min_df=1,           # Include terms that appear in at least 1 document
            max_df=0.95,        # Exclude terms that appear in >95% of documents
            stop_words='english',  # Remove English stop words
            ngram_range=(1, 2), # Use unigrams and bigrams
            token_pattern=r'(?u)\b\w+\b',  # Consider single characters as tokens
            use_idf=True,       # Use inverse document frequency weighting
            smooth_idf=True,    # Add 1 to document frequencies to avoid division by zero
            sublinear_tf=True   # Apply sublinear tf scaling
        )
        
        # Handle empty corpus case
        if not any(processed_contents):
            print("Warning: All processed contents are empty. Creating dummy matrix.")
            matrix = np.zeros((len(urls), 1))
            return urls, processed_paths, processed_contents, content_vectorizer, matrix
        
        # Fit and transform the processed contents
        matrix = content_vectorizer.fit_transform(processed_contents)
        return urls, processed_paths, processed_contents, content_vectorizer, matrix
    
    elif use_url_paths and use_content:
        # Combine URL paths and content with weighting
        # Process URL paths
        url_vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.95, stop_words='english', ngram_range=(1, 2)
        )
        
        # Handle empty URL paths
        if any(processed_paths):
            url_matrix = url_vectorizer.fit_transform(processed_paths)
            # Convert to dense array for easier manipulation
            url_matrix_array = url_matrix.toarray()
        else:
            print("Warning: All processed paths are empty.")
            url_matrix_array = np.zeros((len(urls), 1))
        
        # Process content
        content_vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.95, stop_words='english', ngram_range=(1, 2)
        )
        
        # Handle empty content
        if any(processed_contents):
            content_matrix = content_vectorizer.fit_transform(processed_contents)
            # Convert to dense array for easier manipulation
            content_matrix_array = content_matrix.toarray()
        else:
            print("Warning: All processed contents are empty.")
            content_matrix_array = np.zeros((len(urls), 1))
        
        # Calculate similarity matrices
        # If matrix dimensionality is too small, just use identity matrices
        if url_matrix_array.shape[1] > 1:
            url_similarity = cosine_similarity(url_matrix_array)
        else:
            url_similarity = np.eye(len(urls))
            
        if content_matrix_array.shape[1] > 1:
            content_similarity = cosine_similarity(content_matrix_array)
        else:
            content_similarity = np.eye(len(urls))
        
        # Combine similarity matrices
        combined_similarity = (url_weight * url_similarity) + ((1 - url_weight) * content_similarity)
        
        return urls, processed_paths, processed_contents, None, combined_similarity
    
    else:
        # Neither URL paths nor content - just return dummy matrix
        matrix = np.eye(len(urls))  # Identity matrix as fallback
        return urls, processed_paths, processed_contents, None, matrix

# Test function
def test_vectorizer():
    test_urls = [
        'https://example.com/products/category/item1.html',
        'https://example.com/products/category/item2.html',
        'https://example.com/blog/2023/01/post-title',
        'https://example.com/about-us',
        'https://example.com/contact',
        'https://example.com/products/another-category/item3.php'
    ]
    
    # Mock content
    content_dict = {
        'https://example.com/products/category/item1.html': 
            "This is a product page about item 1. It has features and specifications.",
        'https://example.com/products/category/item2.html': 
            "Product page for item 2. Similar to item 1 but with different specs.",
        'https://example.com/blog/2023/01/post-title': 
            "This is a blog post about something interesting in our industry.",
        'https://example.com/about-us': 
            "About our company. We've been in business since 2010 and specialize in products.",
        'https://example.com/contact': 
            "Contact our team. You can reach us by phone or email.",
        'https://example.com/products/another-category/item3.php': 
            "Item 3 is in a different category but still a product we sell."
    }
    
    # Test URL-only vectorization
    print("\nTesting URL-only vectorization:")
    url_list, processed_paths, _, vectorizer, url_matrix = vectorize_urls_and_content(
        test_urls, content_dict=content_dict, use_url_paths=True, use_content=False
    )
    
    print(f"URL Matrix Shape: {url_matrix.shape}")
    
    # Test content-only vectorization
    print("\nTesting content-only vectorization:")
    url_list, _, processed_content, _, content_matrix = vectorize_urls_and_content(
        test_urls, content_dict=content_dict, use_url_paths=False, use_content=True
    )
    
    print(f"Content Matrix Shape: {content_matrix.shape}")
    
    # Test combined vectorization
    print("\nTesting combined vectorization:")
    url_list, processed_paths, processed_content, _, combined_matrix = vectorize_urls_and_content(
        test_urls, content_dict=content_dict, use_url_paths=True, use_content=True, url_weight=0.3
    )
    
    print(f"Combined Matrix Shape: {combined_matrix.shape}")
    
    # Print sample
    for i in range(min(3, len(url_list))):
        print(f"\nURL: {url_list[i]}")
        print(f"Path: {processed_paths[i]}")
        print(f"Content: {processed_content[i][:50]}...")

if __name__ == "__main__":
    test_vectorizer()