import pandas as pd
import numpy as np
# Use sklearn's pairwise_distances, it's fine
from sklearn.metrics import pairwise_distances
# Use scipy for pdist/squareform if you prefer for pairwise, but sklearn works
# from scipy.spatial.distance import pdist, squareform
import re
from urllib.parse import urlparse

def calculate_metrics(url_list, processed_paths, coordinates_df, centroid):
    """
    Calculate distances, adaptive focus/radius scores, and pairwise distances.

    Focus Score: Measures how tightly points cluster around the centroid.
                 100 = all points at centroid, 0 = average point is as far as the furthest point.
    Radius Score: Measures how far the furthest point is relative to the overall diameter
                  of the point cloud. 100 = furthest point defines the max extent,
                  0 = all points at centroid.

    Args:
        url_list (list): Original URL list
        processed_paths (list): Processed paths from the URLs
        coordinates_df (pd.DataFrame): DataFrame with 'x' and 'y' columns (2D coordinates)
        centroid (tuple): (centroid_x, centroid_y) coordinates of the 2D points

    Returns:
        tuple: (final DataFrame, focus score, radius score, pairwise distance matrix)
    """
    num_points = len(coordinates_df)
    if num_points == 0:
        # Handle empty input
        return pd.DataFrame(columns=['url', 'processed_path', 'x', 'y', 'distance_from_centroid', 'page_type', 'page_depth']), 0, 0, np.array([])

    # Create a copy of the coordinates DataFrame
    result_df = coordinates_df.copy()

    # Add URL and processed path columns
    result_df['url'] = url_list
    result_df['processed_path'] = processed_paths

    # Calculate Euclidean distance from centroid for each point
    result_df['distance_from_centroid'] = np.sqrt(
        (result_df['x'] - centroid[0])**2 + (result_df['y'] - centroid[1])**2
    )

    # Calculate key distance metrics
    avg_distance = result_df['distance_from_centroid'].mean()
    max_distance_from_centroid = result_df['distance_from_centroid'].max()

    # --- Pairwise Distances and Max Spread ---
    pairwise_dist_matrix = np.array([[0.0]]) # Default for single point
    max_pairwise_dist = 0.0 # Default for single point

    if num_points > 1:
        points = coordinates_df[['x', 'y']].values
        pairwise_dist_matrix = pairwise_distances(points)
        # Find the maximum distance between any two points (diameter of the cloud)
        # Exclude the diagonal (distance from a point to itself)
        if pairwise_dist_matrix.size > 1:
             max_pairwise_dist = np.max(pairwise_dist_matrix)

    # --- Adaptive Score Calculation ---
    epsilon = 1e-9 # Small value to prevent division by zero

    # Focus Score Calculation:
    # Compare average distance to the maximum distance from the centroid.
    if max_distance_from_centroid < epsilon:
        # If max distance is ~0, all points are at the centroid, perfect focus.
        focus_score = 100.0
    else:
        # Normalized average distance (0 to 1). Closer to 0 means more focused.
        normalized_avg_dist = avg_distance / (max_distance_from_centroid + epsilon)
        # Invert and scale to 0-100.
        focus_score = 100.0 * (1.0 - normalized_avg_dist)

    # Radius Score Calculation:
    # Compare the max distance from the centroid to the overall max pairwise distance (diameter).
    if max_pairwise_dist < epsilon:
         # If the max pairwise distance is ~0, all points are identical, no radius.
        radius_score = 0.0
    else:
        # How much of the total diameter is covered by the distance from the centroid?
        radius_score = 100.0 * (max_distance_from_centroid / (max_pairwise_dist + epsilon))

    # Clamp scores to the valid range [0, 100] just in case of floating point nuances
    focus_score = max(0.0, min(100.0, focus_score))
    radius_score = max(0.0, min(100.0, radius_score))

    # --- Add Page Type and Depth ---
    result_df['page_type'] = result_df['url'].apply(identify_page_type)
    result_df['page_depth'] = result_df['url'].apply(get_page_depth)

    # Final DataFrame contains all URL data and metrics
    return result_df, focus_score, radius_score, pairwise_dist_matrix

# --- Helper Functions (identify_page_type, get_page_depth) remain the same ---
def identify_page_type(url):
    """
    Identify the likely page type based on URL patterns.
    (Code is identical to previous version - kept for completeness)
    """
    url_lower = url.lower()
    path = urlparse(url).path.lower()
    if not path: path = '/' # Handle cases where path might be None or empty string

    # Home page (stricter check)
    if path == '/' or path == '/index.html' or path == '/index.php' or path == '/index.asp':
        return 'Home'

    # Blog patterns
    if re.search(r'/blog(?:/|$)|/article(?:/|$)|/post(?:/|$)|/news(?:/|$)', path) or \
       re.search(r'/\d{4}/\d{2}(?:/\d{2})?(?:/|$)', path): # Date patterns like /2023/01/15/
        return 'Blog'

    # Product patterns
    if re.search(r'/product(?:/|$)|/item(?:/|$)|/sku(?:/|$)|/shop(?:/|$)', path) or \
       re.search(r'/p/\w+', path): # Changed \d+ to \w+ for more general product IDs
        return 'Product'

    # Category patterns
    if re.search(r'/category(?:/|$)|/cat(?:/|$)|/collection(?:/|$)|/department(?:/|$)|/section(?:/|$)', path):
        return 'Category'

    # About/Info pages
    if re.search(r'/about(?:/|$)|/company(?:/|$)|/team(?:/|$)|/history(?:/|$)|/mission(?:/|$)|/faq(?:/|$)|/help(?:/|$)|/support(?:/|$)', path):
        return 'Info'

    # Contact pages
    if re.search(r'/contact(?:/|$)|/reach-us(?:/|$)|/get-in-touch(?:/|$)', path):
        return 'Contact'

    # Default for unknown patterns
    return 'Other'

def get_page_depth(url):
    """
    Calculate the depth of a page (number of directory levels).
    (Code is identical to previous version - kept for completeness)
    """
    path = urlparse(url).path
    # Remove potential filename at the end before splitting
    if '.' in path.split('/')[-1]:
        path = '/'.join(path.split('/')[:-1])
    # Count segments, but ignore empty segments from start/end slashes
    segments = [s for s in path.strip('/').split('/') if s]
    return len(segments)


# --- find_potential_duplicates function remains the same ---
def find_potential_duplicates(result_df, pairwise_dist_matrix, threshold=1.0):
    """
    Find potential duplicate content based on URL proximity in vector space.
    (Code is mostly identical - small print fix)
    """
    duplicates = []
    n = len(result_df)

    if n <= 1 or pairwise_dist_matrix is None or pairwise_dist_matrix.size == 0:
         print("Not enough data points or invalid distance matrix to find duplicates.")
         return duplicates

    # Debug info
    print(f"\n--- Duplicate Analysis ---")
    print(f"Matrix shape: {pairwise_dist_matrix.shape}, Num points: {n}")
    print(f"Distance threshold: {threshold}")

    # Calculate some stats about the distance matrix
    non_zero_distances = pairwise_dist_matrix[np.triu_indices(n, k=1)] # Upper triangle excluding diagonal
    if len(non_zero_distances) > 0:
        print(f"Min non-zero pairwise distance: {non_zero_distances.min():.4f}")
        print(f"Max pairwise distance: {non_zero_distances.max():.4f}")
        print(f"Mean pairwise distance: {non_zero_distances.mean():.4f}")
    else:
        print("Warning: All pairwise distances are zero!")

    # Find duplicate candidates
    indices = np.where((pairwise_dist_matrix > epsilon) & (pairwise_dist_matrix < threshold)) # Use epsilon
    # indices will contain pairs, need to map them back correctly
    idx_pairs = list(zip(indices[0], indices[1]))
    # Filter to avoid duplicates (i, j) and (j, i), only keep i < j
    unique_pairs = set()
    for i, j in idx_pairs:
        if i < j:
             unique_pairs.add((i,j))

    for i, j in unique_pairs:
         path1 = result_df.iloc[i]['processed_path']
         path2 = result_df.iloc[j]['processed_path']
         # Optional: Check if paths are different to reduce noise if desired
         # if path1 != path2:
         duplicates.append({
                'url1': result_df.iloc[i]['url'],
                'url2': result_df.iloc[j]['url'],
                'distance': pairwise_dist_matrix[i, j],
                'path1': path1,
                'path2': path2
            })


    # Sort by distance (closest pairs first)
    duplicates.sort(key=lambda x: x['distance'])

    print(f"Found {len(duplicates)} potential duplicate pairs below threshold {threshold}")
    print(f"--- End Duplicate Analysis ---")
    return duplicates


# --- test_analyzer function (Updated to use new logic) ---
def test_analyzer():
    # Sample data
    urls = [
        'https://example.com/', # Home
        'https://example.com/products/category/item1.html', # Product
        'https://example.com/products/category/item2.html', # Product
        'https://example.com/blog/2023/01/post-title', # Blog
        'https://example.com/about-us', # Info
        'https://example.com/contact', # Contact
        'https://example.com/products/another-category/item3.php', # Product
        'https://example.com/products/category/' # Category
    ]

    processed_paths = [
        '', # Home
        'products category item1',
        'products category item2',
        'blog 2023 01 post title',
        'about us',
        'contact',
        'products another category item3',
        'products category'
    ]

    # Mock 2D coordinates (increased spread for better testing)
    coordinates = np.array([
        [0.1, 0.1], # Home near center
        [5.0, 5.0], # item1
        [5.5, 5.2], # item2 (close to item1)
        [-4.0, 3.0], # blog
        [0.5, -3.0], # about
        [0.8, -3.2], # contact (close to about)
        [6.0, -4.0], # item3 (different area)
        [5.2, 4.8]  # category (near items 1&2)
    ])

    coordinates_df = pd.DataFrame(coordinates, columns=['x', 'y'])
    # Calculate centroid
    centroid = coordinates.mean(axis=0)
    print(f"Data Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")


    # --- Calculate metrics using the UPDATED function ---
    result_df, focus_score, radius_score, pairwise_dist_matrix = calculate_metrics(
        urls, processed_paths, coordinates_df, centroid
    )

    print("\nResult DataFrame:")
    # Display relevant columns for clarity
    print(result_df[['url', 'page_type', 'page_depth', 'x', 'y', 'distance_from_centroid']].round(2))

    print(f"\nFocus Score: {focus_score:.2f}/100")
    print(f"Radius Score: {radius_score:.2f}/100")

    print("\nPotential Duplicates (Threshold=1.0):")
    # Use a threshold relevant to the scale of mock coordinates
    duplicates = find_potential_duplicates(result_df, pairwise_dist_matrix, threshold=1.0)
    if duplicates:
        for dup in duplicates[:5]: # Show top 5 closest
            print(f"  - {dup['url1']} <-> {dup['url2']} (distance: {dup['distance']:.3f})")
    else:
        print("  No potential duplicates found below the threshold.")

# Add a global epsilon if needed by find_potential_duplicates outside the main func
epsilon = 1e-9

if __name__ == "__main__":
    test_analyzer()