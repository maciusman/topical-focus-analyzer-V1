import pandas as pd
import numpy as np
# Use sklearn's pairwise_distances, it's efficient
from sklearn.metrics import pairwise_distances
import re
from urllib.parse import urlparse

# --- Define epsilon globally as it's used in multiple places ---
epsilon = 1e-9 # Small value to prevent division by zero or issues with log/powers

# --- CORRECTED calculate_metrics FUNCTION ---
# Added k1 and k2 arguments with default values matching Streamlit slider defaults
def calculate_metrics(url_list, processed_paths, coordinates_df, centroid, k1=5.0, k2=5.0):
    """
    Calculate distances, scaled focus/radius scores, and pairwise distances.

    Focus Score: Measures how tightly points cluster around the centroid.
                 Scaled by k1. Higher k1 makes the score more sensitive (drops faster with distance).
                 100 = perfect focus, lower values = less focus.
    Radius Score: Measures how far the furthest point is relative to the overall diameter.
                  Scaled by k2. Higher k2 makes the score more sensitive to spread.
                  Reflects the extent of topic coverage.

    Args:
        url_list (list): Original URL list
        processed_paths (list): Processed paths from the URLs
        coordinates_df (pd.DataFrame): DataFrame with 'x' and 'y' columns (2D coordinates)
        centroid (tuple): (centroid_x, centroid_y) coordinates of the 2D points
        k1 (float): Scaling factor for Focus Score calculation. Default is 5.0.
        k2 (float): Scaling factor for Radius Score calculation. Default is 5.0.

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
    max_pairwise_dist = 0.0 # Default for single point (diameter)

    if num_points > 1:
        points = coordinates_df[['x', 'y']].values
        pairwise_dist_matrix = pairwise_distances(points)
        # Find the maximum distance between any two points (diameter of the cloud)
        if pairwise_dist_matrix.size > 1:
             max_pairwise_dist = np.max(pairwise_dist_matrix)

    # --- Scaled Score Calculation using k1 and k2 ---

    # Focus Score Calculation: Uses average distance relative to max distance, scaled by k1.
    # Higher k1 means score drops faster as average distance increases relative to max.
    if max_distance_from_centroid < epsilon:
        # If max distance is ~0, all points are at the centroid, perfect focus.
        focus_score = 100.0
    else:
        # Normalized average distance (0 to 1). Closer to 0 means more focused.
        normalized_avg_dist = avg_distance / (max_distance_from_centroid + epsilon)
        # Use k1 in an exponential way: makes the score more sensitive for higher k1.
        # (1 - normalized_avg_dist) is high (near 1) for focused sites. Raising to power k1 keeps it high.
        # If less focused (normalized_avg_dist > 0), (1 - norm...) is < 1, power k1 reduces it faster.
        # Divide k1 by its default 5.0 to moderate the scaling effect across the slider range [1, 20]
        focus_score = 100.0 * (1.0 - normalized_avg_dist)**(k1 / 5.0)

    # Radius Score Calculation: Uses max distance from centroid relative to diameter, scaled by k2.
    # Higher k2 makes score rise faster as max distance approaches diameter.
    if max_pairwise_dist < epsilon:
        # If the max pairwise distance is ~0, all points are identical, effectively zero radius.
        radius_score = 0.0
    else:
        # Ratio of max distance from centroid to the total diameter (0 to 1)
        radius_ratio = max_distance_from_centroid / (max_pairwise_dist + epsilon)
        # Use k2 in an exponential growth formula: makes score approach 100 faster for higher k2.
        # (1 - exp(-ratio * k2)) increases from 0 towards 1 as ratio*k2 increases.
        # Divide k2 by its default 5.0 to moderate the scaling effect across the slider range [1, 20]
        radius_score = 100.0 * (1.0 - np.exp(-radius_ratio * (k2 / 5.0)))

    # Clamp scores to the valid range [0, 100]
    focus_score = max(0.0, min(100.0, focus_score))
    radius_score = max(0.0, min(100.0, radius_score))

    # --- Add Page Type and Depth ---
    result_df['page_type'] = result_df['url'].apply(identify_page_type)
    result_df['page_depth'] = result_df['url'].apply(get_page_depth)

    # Final DataFrame contains all URL data and metrics
    # Reorder columns for slightly better readability if desired
    final_cols = ['url', 'processed_path', 'page_type', 'page_depth', 'x', 'y', 'distance_from_centroid']
    # Ensure all expected columns exist before reordering
    final_cols = [col for col in final_cols if col in result_df.columns]
    result_df = result_df[final_cols]

    return result_df, focus_score, radius_score, pairwise_dist_matrix

# --- Helper Functions (identify_page_type, get_page_depth) remain the same ---
def identify_page_type(url):
    """
    Identify the likely page type based on URL patterns.
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
    """
    path = urlparse(url).path
    # Remove potential filename at the end before splitting
    if '.' in path.split('/')[-1]:
        path = '/'.join(path.split('/')[:-1])
    # Count segments, but ignore empty segments from start/end slashes
    segments = [s for s in path.strip('/').split('/') if s]
    return len(segments)


# --- find_potential_duplicates function remains the same, uses global epsilon ---
def find_potential_duplicates(result_df, pairwise_dist_matrix, threshold=1.0):
    """
    Find potential duplicate content based on URL proximity in vector space.
    """
    duplicates = []
    n = len(result_df)

    # Check if enough data and valid matrix
    if n <= 1 or pairwise_dist_matrix is None or pairwise_dist_matrix.shape != (n, n):
        # Optionally print a warning or log this
        # print("Warning: Not enough data points or invalid distance matrix to find duplicates.")
        return duplicates

    # Use numpy indexing for efficiency
    # Find indices where distance is between epsilon (non-zero) and the threshold
    # Use np.triu_indices to avoid duplicate pairs (i,j) and (j,i) and self-pairs (i,i)
    indices_upper_triangle = np.triu_indices(n, k=1)
    distances_upper_triangle = pairwise_dist_matrix[indices_upper_triangle]

    # Filter these distances by the threshold
    close_pairs_indices = np.where((distances_upper_triangle > epsilon) & (distances_upper_triangle < threshold))[0]

    # Get the original row/column indices for these close pairs
    row_indices = indices_upper_triangle[0][close_pairs_indices]
    col_indices = indices_upper_triangle[1][close_pairs_indices]

    # Build the list of duplicates
    for i, j in zip(row_indices, col_indices):
        duplicates.append({
            'url1': result_df.iloc[i]['url'],
            'url2': result_df.iloc[j]['url'],
            'distance': pairwise_dist_matrix[i, j],
            'path1': result_df.iloc[i]['processed_path'],
            'path2': result_df.iloc[j]['processed_path']
        })

    # Sort by distance (closest pairs first)
    duplicates.sort(key=lambda x: x['distance'])

    return duplicates


# --- test_analyzer function (Updated to use new logic and pass k1, k2) ---
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
    # Pass example k1, k2 values (e.g., the defaults)
    test_k1 = 5.0
    test_k2 = 5.0
    print(f"\n--- Running Test with k1={test_k1}, k2={test_k2} ---") # Add info line

    result_df, focus_score, radius_score, pairwise_dist_matrix = calculate_metrics(
        urls, processed_paths, coordinates_df, centroid, k1=test_k1, k2=test_k2
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

    # --- Test with different k values ---
    test_k1_high = 15.0
    test_k2_low = 2.0
    print(f"\n--- Running Test with k1={test_k1_high}, k2={test_k2_low} ---")

    result_df_2, focus_score_2, radius_score_2, _ = calculate_metrics(
        urls, processed_paths, coordinates_df, centroid, k1=test_k1_high, k2=test_k2_low
    )
    print(f"\nFocus Score (High k1): {focus_score_2:.2f}/100")
    print(f"Radius Score (Low k2): {radius_score_2:.2f}/100")


if __name__ == "__main__":
    test_analyzer()
