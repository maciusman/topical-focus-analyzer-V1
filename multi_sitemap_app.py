import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from dotenv import load_dotenv
import numpy as np

# Import our custom modules
from modules.sitemap_finder import find_sitemaps
from modules.sitemap_parser import parse_sitemap
from modules.content_extractor import batch_extract_content, preprocess_text_for_analysis
from modules.simple_vectorizer import vectorize_urls_and_content
from modules.dimensionality_reducer import reduce_dimensions_and_find_centroid
from modules.analyzer import calculate_metrics, find_potential_duplicates
from modules.llm_summarizer import get_gemini_summary

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Topical Focus Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🔍 Topical Focus Analyzer")
st.markdown("""
This tool analyzes the topical focus of a website by examining both URL structure and page content.
It visualizes how tightly focused or widely spread the content topics are.
""")

# Sidebar for input parameters
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Domain input
    domain = st.text_input("Enter a domain (e.g., example.com):", value="")
    
    # URL filtering options
    st.subheader("URL Filtering")
    
    # Enhanced filtering with multiple include/exclude options
    with st.expander("URL Filters", expanded=True):
        # Include filters (up to 3)
        st.markdown("**Include URLs containing:**")
        include_filters = []
        for i in range(3):
            filter_value = st.text_input(f"Include filter #{i+1}:", key=f"include_{i}")
            if filter_value:
                include_filters.append(filter_value)
        
        # Exclude filters (up to 3)
        st.markdown("**Exclude URLs containing:**")
        exclude_filters = []
        for i in range(3):
            filter_value = st.text_input(f"Exclude filter #{i+1}:", key=f"exclude_{i}")
            if filter_value:
                exclude_filters.append(filter_value)
        
        # Filter logic
        filter_logic = st.radio(
            "Include filter logic:",
            ["Match ANY filter (OR)", "Match ALL filters (AND)"],
            index=0,
            help="For multiple include filters, choose whether URLs should match any or all of the filters"
        )
        include_logic_any = filter_logic == "Match ANY filter (OR)"
    
    # Content analysis options
    st.subheader("Content Analysis Options")
    analyze_content = st.checkbox("Analyze Page Content (slower but more accurate)", value=True)
    
    if analyze_content:
        use_urls_too = st.checkbox("Also use URL paths (combined analysis)", value=True)
        url_weight = st.slider("URL Path Weight vs. Content", 0.0, 1.0, 0.3,
                               help="Higher values give more importance to URL paths vs page content")
        
        # Advanced content options
        with st.expander("Advanced Content Options"):
            max_workers = st.slider("Maximum Parallel Workers", 1, 10, 3,
                                   help="Higher values scrape pages faster but may trigger rate limits")
            request_delay = st.slider("Delay Between Requests (seconds)", 0.1, 5.0, 1.0,
                                     help="Longer delays reduce risk of rate limiting")
    else:
        use_urls_too = True
        url_weight = 1.0  # Only use URLs
        max_workers = 3
        request_delay = 1.0
    
    # Advanced options
    with st.expander("Advanced Analysis Options"):
        max_urls = st.slider("Maximum URLs to analyze:", 10, 1000, 100, 
                           help="Lower values are faster but less comprehensive")
        perplexity = st.slider("t-SNE Perplexity:", 5, 50, 15, 
                               help="Lower values preserve local structure, higher values preserve global structure")
        focus_k = st.slider("Focus Score Scaling (k1):", 1.0, 20.0, 5.0, 
                            help="Higher values make the focus score more sensitive to distance variations")
        radius_k = st.slider("Radius Score Scaling (k2):", 1.0, 20.0, 5.0,
                             help="Higher values make the radius score more sensitive to maximum distances")
    
    # Google API key for Gemini
    use_gemini = st.checkbox("Generate AI summary with Gemini", value=True)
    if use_gemini:
        st.info("Ensure you've set your GOOGLE_API_KEY in the .env file or provide it below.")
        google_api_key = st.text_input("Google API Key (optional):", 
                                       value="", 
                                       type="password",
                                       help="Leave empty to use the key from your .env file")
    
    # Analysis button
    analyze_button = st.button("Find Sitemaps", use_container_width=True)

# Initialize session state
if 'sitemaps' not in st.session_state:
    st.session_state.sitemaps = None
if 'selected_sitemaps' not in st.session_state:
    st.session_state.selected_sitemaps = []
if 'urls' not in st.session_state:
    st.session_state.urls = None
if 'url_sources' not in st.session_state:
    st.session_state.url_sources = {}  # Tracks which sitemap each URL came from
if 'content_dict' not in st.session_state:
    st.session_state.content_dict = None
if 'processed_content_dict' not in st.session_state:
    st.session_state.processed_content_dict = {}
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'focus_score' not in st.session_state:
    st.session_state.focus_score = None
if 'radius_score' not in st.session_state:
    st.session_state.radius_score = None
if 'pairwise_distances' not in st.session_state:
    st.session_state.pairwise_distances = None
if 'llm_summary' not in st.session_state:
    st.session_state.llm_summary = None
if 'centroid' not in st.session_state:
    st.session_state.centroid = None
if 'processed_content' not in st.session_state:
    st.session_state.processed_content = None

# Step 1: Find Sitemaps when the button is clicked
if analyze_button and domain:
    with st.spinner("Finding sitemaps..."):
        st.session_state.sitemaps = find_sitemaps(domain)
    
    if not st.session_state.sitemaps:
        st.error(f"No sitemaps found for {domain}. Try checking if the domain is correct.")
    else:
        st.success(f"Found {len(st.session_state.sitemaps)} sitemap(s)!")
        # Reset selected sitemaps when finding new ones
        st.session_state.selected_sitemaps = []

# Step 2: Select multiple sitemaps if sitemaps were found
if st.session_state.sitemaps:
    st.subheader("Available Sitemaps")
    
    # Display sitemaps with checkboxes for multi-selection
    sitemap_cols = st.columns(2)
    
    with sitemap_cols[0]:
        # Create checkboxes for each sitemap
        all_sitemaps = st.session_state.sitemaps
        
        # Add a select all checkbox
        select_all = st.checkbox("Select All Sitemaps", key="select_all_sitemaps")
        
        # Individual sitemap checkboxes
        selected_sitemaps = []
        for i, sitemap in enumerate(all_sitemaps):
            # If select_all is checked, pre-select all checkboxes
            is_checked = select_all or sitemap in st.session_state.selected_sitemaps
            if st.checkbox(f"{i+1}. {sitemap}", value=is_checked, key=f"sitemap_{i}"):
                selected_sitemaps.append(sitemap)
        
        # Save selected sitemaps to session state
        st.session_state.selected_sitemaps = selected_sitemaps
    
    with sitemap_cols[1]:
        # Show selected sitemaps count and list
        st.markdown(f"**{len(selected_sitemaps)} sitemaps selected**")
        if selected_sitemaps:
            with st.expander("View selected sitemaps"):
                for i, sitemap in enumerate(selected_sitemaps):
                    st.write(f"{i+1}. {sitemap}")
    
    # Process sitemap button
    process_button = st.button("Process Selected Sitemaps", use_container_width=True, disabled=len(selected_sitemaps) == 0)
    
    if len(selected_sitemaps) == 0 and process_button:
        st.warning("Please select at least one sitemap to process.")
    
    # Step 3: Process the selected sitemaps
    if process_button and st.session_state.selected_sitemaps:
        # Parse Sitemaps
        with st.spinner("Parsing sitemaps..."):
            all_urls = []
            url_sources = {}  # Track which sitemap each URL came from
            
            # Process each selected sitemap
            for sitemap_url in st.session_state.selected_sitemaps:
                st.info(f"Parsing sitemap: {sitemap_url}")
                
                # Apply the filters based on logic
                def url_passes_filters(url, include_filters, exclude_filters, include_logic_any):
                    # Check exclude filters first (any match excludes the URL)
                    for exclude in exclude_filters:
                        if exclude and exclude.lower() in url.lower():
                            return False
                    
                    # If no include filters, include the URL
                    if not include_filters:
                        return True
                    
                    # Check include filters based on logic
                    if include_logic_any:
                        # ANY logic: URL passes if it matches any include filter
                        return any(include.lower() in url.lower() for include in include_filters if include)
                    else:
                        # ALL logic: URL passes if it matches all include filters
                        return all(include.lower() in url.lower() for include in include_filters if include)
                
                # Parse sitemap (without initial filtering)
                sitemap_urls = parse_sitemap(sitemap_url)
                
                # Apply custom filters
                filtered_urls = [
                    url for url in sitemap_urls 
                    if url_passes_filters(url, include_filters, exclude_filters, include_logic_any)
                ]
                
                # Track source sitemap for each URL
                for url in filtered_urls:
                    url_sources[url] = sitemap_url
                
                # Add to overall URL list
                all_urls.extend(filtered_urls)
                
                st.success(f"Found {len(filtered_urls)} URLs in sitemap (after filtering)")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = [url for url in all_urls if not (url in seen or seen.add(url))]
            
            # Limit the number of URLs if needed
            if len(unique_urls) > max_urls:
                st.warning(f"Limiting analysis to {max_urls} URLs out of {len(unique_urls)} found across all sitemaps.")
                unique_urls = unique_urls[:max_urls]
                # Update url_sources to include only the URLs we're using
                url_sources = {url: source for url, source in url_sources.items() if url in unique_urls}
            
            # Store in session state
            st.session_state.urls = unique_urls
            st.session_state.url_sources = url_sources
            
            # Display sample of URLs for verification
            if unique_urls:
                with st.expander("Sample of URLs found (click to expand)"):
                    for i, url in enumerate(unique_urls[:10]):
                        source = url_sources.get(url, "Unknown")
                        st.write(f"{i+1}. {url} (from: {source})")
                    if len(unique_urls) > 10:
                        st.write(f"... and {len(unique_urls) - 10} more")
            
            # Add a small delay to ensure UI updates are visible
            time.sleep(0.5)
        
        if not st.session_state.urls:
            st.error("No URLs found in the selected sitemaps after applying filters.")
        else:
            st.success(f"Found {len(st.session_state.urls)} unique URLs across all selected sitemaps!")
            
            # Content Extraction Step (if enabled)
            if analyze_content:
                with st.spinner("Extracting page content... This may take a while..."):
                    st.info(f"Extracting content from {len(st.session_state.urls)} pages with {max_workers} parallel workers...")
                    st.warning("This step may take several minutes depending on the website and number of pages.")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Function to extract content with progress updates
                    def extract_with_progress(urls):
                        results = {}
                        total_urls = len(urls)
                        
                        for i, url in enumerate(urls):
                            try:
                                # Extract content for a single URL
                                from modules.content_extractor import extract_main_content
                                content = extract_main_content(url)
                                results[url] = content
                                
                                # Update progress
                                progress = (i + 1) / total_urls
                                progress_bar.progress(progress)
                                
                                # Show current status
                                if (i + 1) % 5 == 0 or (i + 1) == total_urls:
                                    st.text(f"Processed {i + 1} of {total_urls} URLs...")
                                
                                # Add a delay to avoid rate limiting
                                if i < total_urls - 1:
                                    time.sleep(request_delay)
                                    
                            except Exception as e:
                                st.error(f"Error extracting content from {url}: {str(e)}")
                                results[url] = ""
                        
                        return results
                    
                    # For smaller sites, use sequential extraction with progress
                    if len(st.session_state.urls) <= 20:
                        content_dict = extract_with_progress(st.session_state.urls)
                    else:
                        # For larger sites, use parallel extraction
                        content_dict = batch_extract_content(
                            st.session_state.urls, 
                            max_workers=max_workers, 
                            delay=request_delay
                        )
                        progress_bar.progress(1.0)
                    
                    # Store in session state
                    st.session_state.content_dict = content_dict
                    
                    # Process content for vectorization and store both raw and processed versions
                    processed_content_dict = {}
                    for url, content in content_dict.items():
                        processed_content = preprocess_text_for_analysis(content)
                        processed_content_dict[url] = processed_content
                    
                    # Store processed content dictionary for the Content Inspector tab
                    st.session_state.processed_content_dict = processed_content_dict
                    
                    # Show content statistics
                    content_lengths = [len(content) for content in content_dict.values()]
                    avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                    empty_content_count = sum(1 for length in content_lengths if length == 0)
                    
                    st.info(f"Content extraction complete! Average content length: {avg_content_length:.1f} characters")
                    if empty_content_count > 0:
                        st.warning(f"Could not extract content from {empty_content_count} URLs")
                    
                    # Show sample of extracted content
                    with st.expander("Sample of extracted content (click to expand)"):
                        for i, (url, content) in enumerate(list(content_dict.items())[:3]):
                            source = st.session_state.url_sources.get(url, "Unknown")
                            st.write(f"**URL:** {url} (from: {source})")
                            preview = content[:500] + "..." if len(content) > 500 else content
                            st.text_area(f"Content preview {i+1}", preview, height=150)
                    
                    time.sleep(0.5)
            else:
                # If not analyzing content, set content_dict to None
                st.session_state.content_dict = None
                st.session_state.processed_content_dict = {}
            
            # Vectorization step
            with st.spinner("Vectorizing content..."):
                st.info(f"Processing data through vectorization...")
                
                # Determine vectorization approach based on settings
                if analyze_content and use_urls_too:
                    st.info("Using combined URL path and content analysis")
                elif analyze_content:
                    st.info("Using content-only analysis")
                else:
                    st.info("Using URL path-only analysis")
                
                # Perform vectorization
                url_list, processed_paths, processed_content, vectorizer, matrix = vectorize_urls_and_content(
                    st.session_state.urls,
                    content_dict=st.session_state.content_dict,
                    use_url_paths=use_urls_too or not analyze_content,
                    use_content=analyze_content,
                    url_weight=url_weight
                )
                
                # Store processed content
                st.session_state.processed_content = processed_content
                
                # Show vectorization details
                matrix_type = "TF-IDF" if hasattr(matrix, "toarray") else "Similarity/Embedding"
                st.info(f"Vectorization complete. Matrix type: {matrix_type}, Shape: {matrix.shape}")
                
                with st.expander("Vectorization Details (click to expand)"):
                    st.subheader("How Vectorization Works")
                    
                    if analyze_content and use_urls_too:
                        st.markdown("""
                        **Combined URL + Content Analysis:**
                        1. URL paths are processed and vectorized using TF-IDF
                        2. Page content is processed and vectorized separately
                        3. The two vector spaces are combined with weighting
                        4. This captures both URL structure and actual content
                        """)
                    elif analyze_content:
                        st.markdown("""
                        **Content-Only Analysis:**
                        1. Page content is extracted, cleaned, and processed
                        2. Key terms are weighted using TF-IDF
                        3. Pages with similar content will cluster together
                        """)
                    else:
                        st.markdown("""
                        **URL-Only Analysis:**
                        1. URL paths are extracted and processed
                        2. Path components are weighted using TF-IDF
                        3. URLs with similar paths will cluster together
                        """)
                    
                    # Show samples
                    if len(url_list) > 0:
                        st.subheader("Sample Data")
                        for i in range(min(3, len(url_list))):
                            source = st.session_state.url_sources.get(url_list[i], "Unknown")
                            st.write(f"**URL:** {url_list[i]} (from: {source})")
                            st.write(f"**Processed Path:** {processed_paths[i]}")
                            if analyze_content:
                                content_preview = processed_content[i][:200] + "..." if len(processed_content[i]) > 200 else processed_content[i]
                                st.write(f"**Processed Content:** {content_preview}")
                            st.write("---")
                
                time.sleep(0.5)
            
            # Dimensionality Reduction
            with st.spinner("Reducing dimensions (t-SNE)... This may take time..."):
                st.info("Starting t-SNE dimensionality reduction process...")
                st.warning("This step may take several minutes for larger datasets!")
                
                # Create a progress indicator
                progress_placeholder = st.empty()
                progress_placeholder.text("Running t-SNE...")
                
                # Perform dimensionality reduction
                coordinates_df, centroid = reduce_dimensions_and_find_centroid(
                    matrix, 
                    perplexity=min(perplexity, len(url_list)-1)
                )
                st.session_state.centroid = centroid
                
                progress_placeholder.empty()
                st.info(f"t-SNE complete. Centroid located at: ({centroid[0]:.2f}, {centroid[1]:.2f})")
                time.sleep(0.5)
            
            # Calculate Metrics
            with st.spinner("Calculating metrics..."):
                # --- CORRECTED SECTION START ---
                # Now correctly passing the slider values (focus_k, radius_k)
                # These correspond to k1 and k2 expected by calculate_metrics
                results_df, focus_score, radius_score, pairwise_dist_matrix = calculate_metrics(
                    url_list=url_list,
                    processed_paths=processed_paths,
                    coordinates_df=coordinates_df,
                    centroid=centroid,
                    k1=focus_k,  # Pass the value from the 'Focus Score Scaling' slider
                    k2=radius_k   # Pass the value from the 'Radius Score Scaling' slider
                )
                # --- CORRECTED SECTION END ---
                
                # Add content preview column if available
                if analyze_content and st.session_state.processed_content:
                    results_df['content_preview'] = st.session_state.processed_content
                    # Truncate long previews
                    results_df['content_preview'] = results_df['content_preview'].apply(
                        lambda x: x[:200] + "..." if len(x) > 200 else x
                    )
                
                # Add source sitemap column
                results_df['source_sitemap'] = results_df['url'].apply(
                    lambda url: st.session_state.url_sources.get(url, "Unknown")
                )
                
                # Store results in session state
                st.session_state.results_df = results_df
                st.session_state.focus_score = focus_score
                st.session_state.radius_score = radius_score
                st.session_state.pairwise_distances = pairwise_dist_matrix
                
                st.info(f"Metrics calculated. Focus Score: {focus_score:.1f}, Radius Score: {radius_score:.1f}")
            
            # Generate LLM Summary
            if use_gemini:
                with st.spinner("Generating AI summary..."):
                    # Sort by distance from centroid
                    sorted_df = st.session_state.results_df.sort_values('distance_from_centroid')
                    
                    # Get top 5 most focused and most divergent URLs
                    top_focused = sorted_df['url'].head(5).tolist()
                    top_divergent = sorted_df['url'].tail(5).tolist()
                    
                    # Get page type distribution
                    page_types = sorted_df['page_type'].value_counts().to_dict()
                    
                    # Get API key from input or environment
                    api_key = google_api_key if google_api_key else os.getenv("GOOGLE_API_KEY")
                    
                    # Generate summary
                    st.session_state.llm_summary = get_gemini_summary(
                        api_key,
                        focus_score,
                        radius_score,
                        len(url_list),
                        top_focused,
                        top_divergent,
                        page_types
                    )
                    
                    st.success("AI summary generated!")

# Display Results if available
if st.session_state.results_df is not None:
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "URL Details", 
        "Visual Map (t-SNE)", 
        "Cannibalization/Clusters", 
        "Content Inspector"
    ])
    
    # Tab 1: Overview
    with tab1:
        # Display scores in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Site Focus Score", f"{st.session_state.focus_score:.1f}/100",
                     help="Higher score means more focused/coherent topics. 100 is maximum focus.")
        
        with col2:
            st.metric("Site Radius Score", f"{st.session_state.radius_score:.1f}/100",
                     help="Higher score means wider topic coverage. 100 is maximum spread.")
        
        # Display sitemap sources if multiple were used
        if len(set(st.session_state.url_sources.values())) > 1:
            st.subheader("Sitemap Distribution")
            sitemap_counts = st.session_state.results_df['source_sitemap'].value_counts()
            
            # Create pie chart for sitemap distribution
            fig_sitemap = px.pie(
                values=sitemap_counts.values,
                names=sitemap_counts.index,
                title="URLs by Source Sitemap"
            )
            st.plotly_chart(fig_sitemap, use_container_width=True)
        
        # Display LLM Summary if available
        if st.session_state.llm_summary:
            st.subheader("AI Analysis")
            st.markdown(st.session_state.llm_summary)
        
        # Page Type Distribution
        st.subheader("Page Type Distribution")
        page_type_counts = st.session_state.results_df['page_type'].value_counts()
        
        # Create pie chart
        fig_pie = px.pie(
            values=page_type_counts.values,
            names=page_type_counts.index,
            title="Content Distribution by Page Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Display histogram of distances
        st.subheader("Distance Distribution")
        fig_hist = px.histogram(
            st.session_state.results_df,
            x="distance_from_centroid",
            nbins=30,
            title="Distribution of URL Distances from Topic Centroid",
            labels={"distance_from_centroid": "Distance from Centroid"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Show most focused and divergent URLs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Focused URLs")
            focused_df = st.session_state.results_df.nsmallest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']]
            focused_df = focused_df.rename(columns={'distance_from_centroid': 'distance'})
            st.dataframe(focused_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Most Divergent URLs")
            divergent_df = st.session_state.results_df.nlargest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']]
            divergent_df = divergent_df.rename(columns={'distance_from_centroid': 'distance'})
            st.dataframe(divergent_df, hide_index=True, use_container_width=True)
    
    # Tab 2: URL Details
    with tab2:
        st.subheader("URL Analysis Details")
        
        # Add search and filtering options
        filter_cols = st.columns([2, 1, 1])
        
        with filter_cols[0]:
            search_term = st.text_input("Search URLs:")
        
        with filter_cols[1]:
            # Filter by page type
            page_types = ["All"] + sorted(st.session_state.results_df['page_type'].unique().tolist())
            selected_page_type = st.selectbox("Filter by Page Type:", page_types)
        
        with filter_cols[2]:
            # Filter by source sitemap if multiple were used
            if len(set(st.session_state.url_sources.values())) > 1:
                sitemaps = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist())
                selected_sitemap = st.selectbox("Filter by Sitemap:", sitemaps)
            else:
                selected_sitemap = "All"
        
        # Display the DataFrame with pagination
        filtered_df = st.session_state.results_df.copy()
        
        # Apply search filter if provided
        if search_term:
            mask = filtered_df['url'].str.contains(search_term, case=False)
            filtered_df = filtered_df[mask]
        
        # Apply page type filter if not "All"
        if selected_page_type != "All":
            filtered_df = filtered_df[filtered_df['page_type'] == selected_page_type]
        
        # Apply sitemap filter if not "All"
        if selected_sitemap != "All":
            filtered_df = filtered_df[filtered_df['source_sitemap'] == selected_sitemap]
        
        # Select columns to display
        display_columns = ['url', 'page_type', 'source_sitemap', 'page_depth', 'distance_from_centroid']
        
        # Add content preview if available
        if 'content_preview' in filtered_df.columns:
            display_columns.append('content_preview')
        
        # Display dataframe with the selected columns
        display_df = filtered_df[display_columns]
        display_df = display_df.rename(columns={
            'distance_from_centroid': 'distance',
            'page_depth': 'depth',
            'source_sitemap': 'sitemap'
        })
        
        st.dataframe(
            display_df.sort_values('distance'),
            use_container_width=True,
            column_config={
                "url": st.column_config.TextColumn("URL"),
                "page_type": st.column_config.TextColumn("Page Type"),
                "sitemap": st.column_config.TextColumn("Sitemap Source"),
                "depth": st.column_config.NumberColumn("Depth"),
                "distance": st.column_config.NumberColumn("Distance", format="%.3f"),
                "content_preview": st.column_config.TextColumn("Content Preview"),
            }
        )
        
        # Show filter stats
        st.info(f"Showing {len(filtered_df)} URLs out of {len(st.session_state.results_df)} total URLs")
    
    # Tab 3: Visual Map (t-SNE)
    with tab3:
        st.subheader("Topical Map Visualization")
        
        # Color and filter options
        visual_cols = st.columns([2, 1, 1])
        
        with visual_cols[0]:
            # Color options
            color_options = {
                "Distance from Centroid": "distance_from_centroid",
                "Page Type": "page_type",
                "Page Depth": "page_depth",
                "Source Sitemap": "source_sitemap"
            }
            color_by = st.selectbox("Color points by:", list(color_options.keys()))
        
        with visual_cols[1]:
            # Point size option
            point_size = st.slider("Point Size:", 3, 15, 8)
        
        with visual_cols[2]:
            # Filter visualization by sitemap source
            if len(set(st.session_state.url_sources.values())) > 1:
                viz_sitemaps = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist())
                viz_sitemap = st.selectbox("Show sitemap:", viz_sitemaps, key="viz_sitemap_filter")
            else:
                viz_sitemap = "All"
        
        # Filter dataframe for visualization if needed
        viz_df = st.session_state.results_df.copy()
        if viz_sitemap != "All":
            viz_df = viz_df[viz_df['source_sitemap'] == viz_sitemap]
        
        # Create the scatter plot
        if color_by == "Page Type" or color_by == "Source Sitemap":
            fig = px.scatter(
                viz_df,
                x="x",
                y="y",
                color=color_options[color_by],
                hover_name="url",
                hover_data=["distance_from_centroid", "page_depth", "source_sitemap"],
                title="t-SNE Visualization of Content Clustering",
                labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
                size_max=point_size
            )
        else:
            fig = px.scatter(
                viz_df,
                x="x",
                y="y",
                color=color_options[color_by],
                color_continuous_scale="Viridis",
                hover_name="url",
                hover_data=["page_type", "distance_from_centroid", "page_depth", "source_sitemap"],
                title="t-SNE Visualization of Content Clustering",
                labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
                size_max=point_size
            )
        
        # Add centroid marker
        if st.session_state.centroid:
            fig.add_trace(
                go.Scatter(
                    x=[st.session_state.centroid[0]],
                    y=[st.session_state.centroid[1]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=15,
                        color="red",
                        line=dict(color="black", width=1)
                    ),
                    name="Topic Centroid",
                    hoverinfo="name"
                )
            )
        
        # Update layout for better appearance
        fig.update_layout(
            height=700,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **How to interpret this visualization:**
        * Each point represents a URL from the sitemap
        * Points that cluster together have similar topics
        * The star marker represents the "topic centroid" - the center of all topics
        * Distances from the centroid reflect how focused or divergent each URL is
        * Colors help identify patterns in content structure
        """)
    
    # Tab 4: Cannibalization/Clusters
    with tab4:
        st.subheader("Content Cannibalization Analysis")
        
        st.markdown("""
        This tab helps identify potentially duplicate or cannibalized content by finding URLs that are 
        very close to each other in the vector space, suggesting similar topics.
        """)
        
        # Add filtering options for cannibalization analysis
        cann_cols = st.columns([2, 1, 1])
        
        with cann_cols[0]:
            # Slider for distance threshold
            threshold = st.slider(
                "Distance Threshold:",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Lower values find more similar content. URLs closer than this distance will be considered potential duplicates."
            )
        
        with cann_cols[1]:
            # Max pairs to show
            max_pairs = st.number_input("Max pairs to display:", 5, 100, 20)
        
        with cann_cols[2]:
            # Filter by sitemap source
            if len(set(st.session_state.url_sources.values())) > 1:
                cann_filter_options = [
                    "All pairs", 
                    "Only pairs from same sitemap", 
                    "Only pairs from different sitemaps"
                ]
                cann_filter = st.selectbox("Filter pairs:", cann_filter_options)
            else:
                cann_filter = "All pairs"
        
        # Find potential duplicates based on threshold
        if st.session_state.results_df is not None and st.session_state.pairwise_distances is not None:
            # Show distance matrix statistics
            with st.expander("Distance Matrix Statistics"):
                dist_matrix = st.session_state.pairwise_distances
                st.write(f"Matrix shape: {dist_matrix.shape}")
                st.write(f"Min distance: {dist_matrix.min()}")
                st.write(f"Max distance: {dist_matrix.max()}")
                st.write(f"Mean distance: {dist_matrix.mean()}")
                st.write(f"Number of zero distances: {(dist_matrix == 0).sum()}")
                
                # Create a histogram of distances
                fig = px.histogram(
                    x=dist_matrix.flatten(),
                    nbins=50,
                    title="Distribution of All Pairwise Distances",
                    labels={"x": "Distance"}
                )
                st.plotly_chart(fig)
            
            duplicates = find_potential_duplicates(
                st.session_state.results_df,
                st.session_state.pairwise_distances,
                threshold
            )
            
            # Filter duplicates based on sitemap source if needed
            if cann_filter != "All pairs" and duplicates:
                filtered_duplicates = []
                for dup in duplicates:
                    url1 = dup['url1']
                    url2 = dup['url2']
                    source1 = st.session_state.url_sources.get(url1, "Unknown")
                    source2 = st.session_state.url_sources.get(url2, "Unknown")
                    
                    if cann_filter == "Only pairs from same sitemap" and source1 == source2:
                        filtered_duplicates.append(dup)
                    elif cann_filter == "Only pairs from different sitemaps" and source1 != source2:
                        filtered_duplicates.append(dup)
                
                duplicates = filtered_duplicates
            
            # Limit number of pairs to display
            displayed_duplicates = duplicates[:max_pairs] if duplicates else []
            
            if not displayed_duplicates:
                st.info(f"No potential duplicates found with current settings. Try increasing the threshold or changing filters.")
            else:
                total_count = len(duplicates)
                displayed_count = len(displayed_duplicates)
                
                if total_count > displayed_count:
                    st.success(f"Found {total_count} potential content duplicates/cannibalization! Showing top {displayed_count}.")
                else:
                    st.success(f"Found {total_count} potential content duplicates/cannibalization!")
                
                # Display in an expandable format
                for i, row in enumerate(displayed_duplicates):
                    source1 = st.session_state.url_sources.get(row['url1'], "Unknown")
                    source2 = st.session_state.url_sources.get(row['url2'], "Unknown")
                    
                    # Create a title that shows sitemap sources
                    title = f"Pair {i+1}: Distance {row['distance']:.3f}"
                    if source1 != source2:
                        title += f" (Cross-Sitemap: {source1} → {source2})"
                    
                    with st.expander(title):
                        cols = st.columns(2)
                        
                        with cols[0]:
                            st.markdown(f"**URL 1:** [{row['url1']}]({row['url1']})")
                            st.text(f"Processed path: {row['path1']}")
                            st.text(f"Source sitemap: {source1}")
                            
                            # Show content preview if available
                            if 'content_preview' in st.session_state.results_df.columns:
                                url1_content = st.session_state.results_df[
                                    st.session_state.results_df['url'] == row['url1']
                                ]['content_preview'].values
                                
                                if len(url1_content) > 0:
                                    st.text_area("Content preview:", url1_content[0], height=150)
                        
                        with cols[1]:
                            st.markdown(f"**URL 2:** [{row['url2']}]({row['url2']})")
                            st.text(f"Processed path: {row['path2']}")
                            st.text(f"Source sitemap: {source2}")
                            
                            # Show content preview if available
                            if 'content_preview' in st.session_state.results_df.columns:
                                url2_content = st.session_state.results_df[
                                    st.session_state.results_df['url'] == row['url2']
                                ]['content_preview'].values
                                
                                if len(url2_content) > 0:
                                    st.text_area("Content preview:", url2_content[0], height=150)
    
    # Tab 5: Content Inspector
    with tab5:
        st.subheader("Content Inspector")
        
        if not st.session_state.content_dict:
            st.info("No content was extracted. Please run the analysis with content extraction enabled.")
        else:
            st.markdown("""
            This tab lets you examine the raw and processed content that was extracted from each URL 
            and used for the analysis. This helps you understand what text was actually vectorized.
            """)
            
            # Add filtering options for the content inspector
            inspector_cols = st.columns([2, 1, 1])
            
            with inspector_cols[0]:
                # Add a search box for URLs
                url_search = st.text_input("Search URLs:", key="content_inspector_search")
            
            with inspector_cols[1]:
                # Filter by page type
                inspector_page_types = ["All"] + sorted(st.session_state.results_df['page_type'].unique().tolist())
                inspector_page_type = st.selectbox("Filter by Page Type:", inspector_page_types, key="inspector_page_type")
            
            with inspector_cols[2]:
                # Filter by source sitemap
                if len(set(st.session_state.url_sources.values())) > 1:
                    inspector_sitemaps = ["All"] + sorted(set(st.session_state.url_sources.values()))
                    inspector_sitemap = st.selectbox("Filter by Sitemap:", inspector_sitemaps, key="inspector_sitemap")
                else:
                    inspector_sitemap = "All"
            
            # Create a searchable dropdown of all URLs
            all_urls = list(st.session_state.content_dict.keys())
            
            # Filter URLs based on search and other filters
            filtered_urls = all_urls.copy()
            
            # Apply search filter
            if url_search:
                filtered_urls = [url for url in filtered_urls if url_search.lower() in url.lower()]
            
            # Apply page type filter
            if inspector_page_type != "All":
                filtered_urls = [
                    url for url in filtered_urls 
                    if url in st.session_state.results_df['url'].values and
                    st.session_state.results_df[st.session_state.results_df['url'] == url]['page_type'].values[0] == inspector_page_type
                ]
            
            # Apply sitemap filter
            if inspector_sitemap != "All":
                filtered_urls = [
                    url for url in filtered_urls 
                    if st.session_state.url_sources.get(url) == inspector_sitemap
                ]
            
            # Add a dropdown for selecting URLs
            if filtered_urls:
                selected_url = st.selectbox(
                    "Select URL to inspect:",
                    filtered_urls,
                    key="content_inspector_url"
                )
                
                if selected_url:
                    # Get raw and processed content
                    raw_content = st.session_state.content_dict.get(selected_url, "")
                    processed_content = st.session_state.processed_content_dict.get(selected_url, "")
                    source_sitemap = st.session_state.url_sources.get(selected_url, "Unknown")
                    
                    # Get distance from centroid if available
                    distance = None
                    page_type = None
                    if st.session_state.results_df is not None:
                        url_row = st.session_state.results_df[st.session_state.results_df['url'] == selected_url]
                        if not url_row.empty:
                            distance = url_row['distance_from_centroid'].values[0]
                            if 'page_type' in url_row.columns:
                                page_type = url_row['page_type'].values[0]
                    
                    # Display URL info
                    st.subheader("URL Information")
                    
                    # Create columns for URL details
                    info_cols = st.columns(4)
                    with info_cols[0]:
                        st.markdown(f"**URL:** [{selected_url}]({selected_url})")
                    
                    with info_cols[1]:
                        st.markdown(f"**Source Sitemap:** {source_sitemap}")
                    
                    with info_cols[2]:
                        if page_type:
                            st.markdown(f"**Page Type:** {page_type}")
                    
                    with info_cols[3]:
                        if distance is not None:
                            st.markdown(f"**Distance from Centroid:** {distance:.3f}")
                            
                            # Show whether this URL is among most focused or divergent
                            sorted_df = st.session_state.results_df.sort_values('distance_from_centroid')
                            rank = sorted_df.index[sorted_df['url'] == selected_url].tolist()
                            if rank:
                                percentile = (rank[0] / len(sorted_df)) * 100
                                st.markdown(f"**Percentile:** {percentile:.1f}%")
                                
                                if percentile < 20:
                                    st.success("This URL is among the most focused content")
                                elif percentile > 80:
                                    st.warning("This URL is among the most divergent content")
                    
                    # Display content
                    st.subheader("Content Inspector")
                    
                    # Create tabs for different content views
                    content_tabs = st.tabs(["Raw Content", "Processed Content", "Content Statistics"])
                    
                    # Tab 1: Raw Content
                    with content_tabs[0]:
                        st.markdown("**Raw extracted content** (before preprocessing):")
                        if raw_content:
                            st.text_area("Raw Content", raw_content, height=400)
                            st.info(f"Raw content length: {len(raw_content)} characters")
                        else:
                            st.warning("No raw content was extracted for this URL")
                    
                    # Tab 2: Processed Content
                    with content_tabs[1]:
                        st.markdown("**Processed content** (after preprocessing, used for vectorization):")
                        if processed_content:
                            st.text_area("Processed Content", processed_content, height=400)
                            st.info(f"Processed content length: {len(processed_content)} characters")
                            
                            # Show changes made during processing
                            if raw_content:
                                st.subheader("Processing Changes")
                                reduction_pct = 100 - (len(processed_content) / len(raw_content) * 100) if len(raw_content) > 0 else 0
                                st.markdown(f"Content was reduced by **{reduction_pct:.1f}%** during processing")
                                
                                # Display some examples of removed content
                                if len(raw_content) > 0:
                                    raw_words = set(raw_content.lower().split())
                                    processed_words = set(processed_content.split())
                                    removed_words = raw_words - processed_words
                                    
                                    if removed_words:
                                        st.markdown("**Examples of removed words:**")
                                        st.write(", ".join(list(removed_words)[:50]))
                        else:
                            st.warning("No processed content available for this URL")
                    
                    # Tab 3: Content Statistics
                    with content_tabs[2]:
                        if processed_content:
                            # Word frequency analysis
                            st.subheader("Word Frequency Analysis")
                            
                            # Count word frequencies
                            words = processed_content.split()
                            word_freq = {}
                            for word in words:
                                if len(word) > 1:  # Skip single-character words
                                    word_freq[word] = word_freq.get(word, 0) + 1
                            
                            # Create DataFrame for display
                            if word_freq:
                                freq_df = pd.DataFrame({
                                    'Word': list(word_freq.keys()),
                                    'Frequency': list(word_freq.values())
                                })
                                freq_df = freq_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
                                
                                # Display word frequency table
                                st.dataframe(
                                    freq_df.head(20),
                                    use_container_width=True,
                                    column_config={
                                        "Word": st.column_config.TextColumn("Word"),
                                        "Frequency": st.column_config.NumberColumn("Frequency")
                                    }
                                )
                                
                                # Create bar chart of top words
                                fig = px.bar(
                                    freq_df.head(15),
                                    x='Word',
                                    y='Frequency',
                                    title="Top 15 Words in Content"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Basic text statistics
                                st.subheader("Text Statistics")
                                stats_cols = st.columns(3)
                                
                                with stats_cols[0]:
                                    st.metric("Word Count", len(words))
                                    
                                with stats_cols[1]:
                                    st.metric("Unique Words", len(word_freq))
                                    
                                with stats_cols[2]:
                                    lexical_diversity = len(word_freq) / len(words) if len(words) > 0 else 0
                                    st.metric("Lexical Diversity", f"{lexical_diversity:.2f}")
                                    st.caption("(Higher values indicate more diverse vocabulary)")
                            else:
                                st.warning("Not enough content for statistical analysis")
                        else:
                            st.warning("No processed content available for statistical analysis")
                else:
                    st.info("Please select a URL to inspect its content")
            else:
                st.warning("No URLs match your search criteria")
                
            # Add bulk download options
            with st.expander("Bulk Export Options"):
                st.markdown("Download all extracted content as CSV file")
                
                if st.button("Generate Content CSV"):
                    # Create DataFrame with content data
                    content_data = []
                    for url in all_urls:
                        raw = st.session_state.content_dict.get(url, "")
                        processed = st.session_state.processed_content_dict.get(url, "")
                        source = st.session_state.url_sources.get(url, "Unknown")
                        
                        # Get distance if available
                        distance = None
                        page_type = None
                        if st.session_state.results_df is not None:
                            url_row = st.session_state.results_df[st.session_state.results_df['url'] == url]
                            if not url_row.empty:
                                distance = url_row['distance_from_centroid'].values[0]
                                if 'page_type' in url_row.columns:
                                    page_type = url_row['page_type'].values[0]
                        
                        content_data.append({
                            'URL': url,
                            'Source Sitemap': source,
                            'Page Type': page_type,
                            'Distance from Centroid': distance,
                            'Raw Content Length': len(raw),
                            'Processed Content Length': len(processed),
                            'Raw Content Preview': raw[:500] + "..." if len(raw) > 500 else raw,
                            'Processed Content': processed
                        })
                    
                    # Create DataFrame
                    content_df = pd.DataFrame(content_data)
                    
                    # Convert to CSV
                    csv = content_df.to_csv(index=False)
                    
                    # Create download link
                    import base64
                    csv_b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{csv_b64}" download="site_content_export.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Topical Focus Analyzer** | Built with Python, Streamlit, and Content Analysis")
