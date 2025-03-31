import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

def find_sitemaps(domain):
    """
    Find all sitemap URLs for a given domain.
    
    Args:
        domain (str): Domain name (e.g., 'example.com')
        
    Returns:
        list: List of sitemap URLs
    """
    # Normalize domain (remove protocol if present)
    domain = domain.strip().lower()
    if domain.startswith(('http://', 'https://')):
        domain = re.sub(r'^https?://', '', domain)
    
    # Remove trailing slash if present
    domain = domain.rstrip('/')
    
    # Try both http and https protocols
    protocols = ['https', 'http']
    sitemaps = []
    
    for protocol in protocols:
        # First try robots.txt to find sitemaps
        robots_url = f"{protocol}://{domain}/robots.txt"
        
        try:
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                # Parse robots.txt line by line
                for line in response.text.splitlines():
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemaps.append(sitemap_url)
                
                # If we found sitemaps, no need to try the other protocol
                if sitemaps:
                    break
        except requests.RequestException as e:
            print(f"Error fetching robots.txt from {robots_url}: {e}")
            # Continue with the other protocol or common paths
    
    # If no sitemaps found in robots.txt, try common sitemap paths
    if not sitemaps:
        common_paths = [
            '/sitemap.xml',
            '/sitemap_index.xml',
            '/sitemap-index.xml',
            '/sitemapindex.xml',
            '/wp-sitemap.xml',  # WordPress
            '/sitemap1.xml'
        ]
        
        for protocol in protocols:
            for path in common_paths:
                sitemap_url = f"{protocol}://{domain}{path}"
                try:
                    response = requests.get(sitemap_url, timeout=10)
                    if response.status_code == 200 and 'xml' in response.headers.get('Content-Type', ''):
                        sitemaps.append(sitemap_url)
                except requests.RequestException:
                    continue
    
    # Process sitemap indexes to get actual sitemaps
    processed_sitemaps = []
    for sitemap_url in sitemaps.copy():
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Check if it's a sitemap index (has <sitemapindex> tag)
                sitemap_index = soup.find('sitemapindex')
                if sitemap_index:
                    # Extract sitemap URLs from the index
                    for sitemap_tag in sitemap_index.find_all('sitemap'):
                        loc = sitemap_tag.find('loc')
                        if loc and loc.text:
                            processed_sitemaps.append(loc.text.strip())
                else:
                    # It's a regular sitemap, keep it
                    processed_sitemaps.append(sitemap_url)
        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {e}")
            # Keep the original sitemap URL in case it's valid despite the error
            processed_sitemaps.append(sitemap_url)
    
    # Return unique sitemap URLs
    return list(set(processed_sitemaps))

# Optional function to test the sitemap finder
def test_sitemap_finder():
    test_domains = [
        'python.org',
        'wikipedia.org',
        'nytimes.com'
    ]
    
    for domain in test_domains:
        print(f"\nTesting domain: {domain}")
        sitemaps = find_sitemaps(domain)
        print(f"Found {len(sitemaps)} sitemaps:")
        for sitemap in sitemaps:
            print(f"  - {sitemap}")

if __name__ == "__main__":
    test_sitemap_finder()