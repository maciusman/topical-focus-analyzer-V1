import requests
from bs4 import BeautifulSoup
import gzip
from io import BytesIO

def parse_sitemap(sitemap_url, url_filter=None, url_exclude=None):
    """
    Extract all page URLs from a given sitemap URL.
    
    Args:
        sitemap_url (str): URL of the sitemap to parse
        url_filter (str, optional): Only include URLs containing this string
        url_exclude (str, optional): Exclude URLs containing this string
    
    Returns:
        list: List of page URLs found in the sitemap
    """
    urls = []
    
    try:
        response = requests.get(sitemap_url, timeout=15)
        
        # Check if the response is gzipped
        content = None
        if sitemap_url.endswith('.gz'):
            try:
                content = gzip.decompress(response.content)
            except Exception as e:
                print(f"Error decompressing gzipped sitemap: {e}")
                return []
        else:
            content = response.content
        
        # Parse the XML content
        soup = BeautifulSoup(content, 'lxml')
        
        # Look for <url> tags (standard sitemap)
        for url_tag in soup.find_all('url'):
            loc = url_tag.find('loc')
            if loc and loc.text:
                page_url = loc.text.strip()
                urls.append(page_url)
        
        # If no <url> tags found, check for <loc> directly (simplified sitemap)
        if not urls:
            for loc in soup.find_all('loc'):
                if loc.parent.name != 'sitemap':  # Skip sitemap index entries
                    page_url = loc.text.strip()
                    urls.append(page_url)
        
        # Apply filtering if specified
        if url_filter or url_exclude:
            filtered_urls = []
            for url in urls:
                include = True
                
                # Apply inclusion filter if specified
                if url_filter and url_filter.strip() and url_filter.strip().lower() not in url.lower():
                    include = False
                
                # Apply exclusion filter if specified
                if url_exclude and url_exclude.strip() and url_exclude.strip().lower() in url.lower():
                    include = False
                
                if include:
                    filtered_urls.append(url)
            
            urls = filtered_urls
        
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
    
    return urls

# Optional function to test the sitemap parser
def test_sitemap_parser():
    test_sitemaps = [
        'https://www.python.org/sitemap.xml',
        'https://www.sitemaps.org/sitemap.xml'
    ]
    
    for sitemap in test_sitemaps:
        print(f"\nTesting sitemap: {sitemap}")
        urls = parse_sitemap(sitemap)
        print(f"Found {len(urls)} URLs, first 5:")
        for url in urls[:5]:
            print(f"  - {url}")
        
        # Test filtering
        if urls:
            filter_term = "about"
            filtered_urls = parse_sitemap(sitemap, url_filter=filter_term)
            print(f"\nFiltered with term '{filter_term}': found {len(filtered_urls)} URLs")

if __name__ == "__main__":
    test_sitemap_parser()