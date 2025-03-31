import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

def reduce_dimensions_and_find_centroid(tfidf_matrix, random_state=42, perplexity=30, sample_limit=5000):
    """
    Reduce TF-IDF vectors to 2D using t-SNE and find the centroid.
    
    Args:
        tfidf_matrix: TF-IDF matrix from scikit-learn
        random_state (int): Random seed for reproducibility
        perplexity (int): t-SNE perplexity parameter
        sample_limit (int): Maximum number of samples to process at once
        
    Returns:
        tuple: (DataFrame with 'x' and 'y' columns, centroid coordinates tuple)
    """
    # Check if matrix is empty or invalid
    if tfidf_matrix is None or tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        return pd.DataFrame(columns=['x', 'y']), (0, 0)
    
    n_samples = tfidf_matrix.shape[0]
    
    # For very large datasets, we might need to sample or use PCA first
    coordinates = None
    if n_samples > sample_limit:
        # Convert to dense array for easier sampling
        if hasattr(tfidf_matrix, 'toarray'):
            tfidf_array = tfidf_matrix.toarray()
        else:
            tfidf_array = np.array(tfidf_matrix)
        
        # Use PCA first to reduce dimensions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = PCA(n_components=min(50, tfidf_array.shape[1]), random_state=random_state)
            reduced_data = pca.fit_transform(tfidf_array)
            
            # Then use t-SNE on the PCA-reduced data
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, n_samples-1),  # perplexity must be less than n_samples
                n_iter=1000,
                random_state=random_state
            )
            coordinates = tsne.fit_transform(reduced_data)
    else:
        # For smaller datasets, use t-SNE directly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne = TSNE(
                n_components=2,
                perplexity=min(perplexity, n_samples-1),  # perplexity must be less than n_samples
                n_iter=1000,
                random_state=random_state
            )
            
            # Check if we need to convert from sparse to dense
            if hasattr(tfidf_matrix, 'toarray'):
                coordinates = tsne.fit_transform(tfidf_matrix.toarray())
            else:
                coordinates = tsne.fit_transform(tfidf_matrix)
    
    # Create DataFrame with the 2D coordinates
    coordinates_df = pd.DataFrame(coordinates, columns=['x', 'y'])
    
    # Calculate centroid
    centroid_x = coordinates_df['x'].mean()
    centroid_y = coordinates_df['y'].mean()
    centroid = (centroid_x, centroid_y)
    
    return coordinates_df, centroid

# Optional function to test the dimensionality reducer
def test_dimensionality_reducer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample data
    texts = [
        "products category item",
        "products category another item",
        "blog post title",
        "about us page",
        "contact information",
        "products different category item"
    ]
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"Original TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # Reduce dimensions and find centroid
    coordinates_df, centroid = reduce_dimensions_and_find_centroid(tfidf_matrix)
    
    print("\n2D Coordinates:")
    print(coordinates_df)
    
    print(f"\nCentroid: {centroid}")

if __name__ == "__main__":
    test_dimensionality_reducer()