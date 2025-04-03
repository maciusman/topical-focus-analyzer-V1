# --- START OF FILE multi_sitemap_app.py ---

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from dotenv import load_dotenv
import numpy as np
import pickle  # Added for saving/loading
import base64 # Added for CSV download link

# Import our custom modules
# Ensure these modules exist in a 'modules' directory or adjust path as needed
try:
    from modules.sitemap_finder import find_sitemaps
    from modules.sitemap_parser import parse_sitemap
    from modules.content_extractor import batch_extract_content, preprocess_text_for_analysis, extract_main_content # Ensure extract_main_content is importable if used directly
    from modules.simple_vectorizer import vectorize_urls_and_content
    from modules.dimensionality_reducer import reduce_dimensions_and_find_centroid
    from modules.analyzer import calculate_metrics, find_potential_duplicates
    from modules.llm_summarizer import get_gemini_summary
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure the 'modules' directory and required files exist.")
    st.stop()

# Load environment variables
load_dotenv()

# --- Session State Initialization ---
# Initialize session state variables FIRST, before widgets access them
# We check if they exist because loading might populate them before widgets are drawn

# Analysis results
DEFAULT_SESSION_STATE = {
    'sitemaps': None,
    'selected_sitemaps': [],
    'urls': None,
    'url_sources': {},
    'content_dict': None,
    'processed_content_dict': {},
    'results_df': None,
    'focus_score': None,
    'radius_score': None,
    'pairwise_distances': None,
    'llm_summary': None,
    'centroid': None,
    'processed_content': None, # List of processed content strings after vectorization step
    'analysis_loaded': False, # Flag to indicate if state was loaded
    '_loading_in_progress': False, # Temp flag during loading

    # Input parameters with defaults
    'domain': "",
    'input_include_filters': ["", "", ""], # Use a distinct key for input state storage
    'input_exclude_filters': ["", "", ""], # Use a distinct key for input state storage
    'input_include_logic_any': True,
    'input_analyze_content': True,
    'input_use_urls_too': True,
    'input_url_weight': 0.3,
    'input_max_workers': 3,
    'input_request_delay': 1.0,
    'input_max_urls': 100,
    'input_perplexity': 15,
    'input_focus_k': 5.0,
    'input_radius_k': 5.0,
    'input_use_gemini': True,
    'input_google_api_key': "" # Store API key entered (but don't pre-fill password field)
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Pickle Load Function ---
def load_analysis_state(uploaded_file):
    """Loads analysis state from a .pkl file."""
    try:
        state = pickle.load(uploaded_file)

        # Restore results state first
        for key in [
            'sitemaps', 'selected_sitemaps', 'urls', 'url_sources', 'content_dict',
            'processed_content_dict', 'results_df', 'focus_score', 'radius_score',
            'pairwise_distances', 'llm_summary', 'centroid', 'processed_content'
            ]:
            st.session_state[key] = state.get(key, DEFAULT_SESSION_STATE.get(key)) # Use loaded or default

        # Restore input parameters to their specific session state keys
        input_param_keys = [
            'input_domain', 'input_include_filters', 'input_exclude_filters',
            'input_include_logic_any', 'input_analyze_content', 'input_use_urls_too',
            'input_url_weight', 'input_max_workers', 'input_request_delay', 'input_max_urls',
            'input_perplexity', 'input_focus_k', 'input_radius_k', 'input_use_gemini',
            'input_google_api_key'
        ]
        for key in input_param_keys:
             # Map loaded 'input_domain' back to 'domain' in session state etc.
             session_state_key = key # Assume mapping like 'input_domain' -> 'input_domain'
             if key == 'input_domain': session_state_key = 'domain' # Special case for domain

             loaded_value = state.get(key, DEFAULT_SESSION_STATE.get(key))

             # Ensure filter lists have correct length
             if key == 'input_include_filters' or key == 'input_exclude_filters':
                  loaded_value = (loaded_value + ["", "", ""])[:3]

             st.session_state[session_state_key] = loaded_value

        st.session_state.analysis_loaded = True # Set flag to prevent re-analysis
        st.success("Analysis state loaded successfully! Sidebar values updated.")
        # No need to rerun here, sidebar widgets will read the updated session state on the *next* natural rerun

        return True

    except pickle.UnpicklingError:
        st.error("Error: Could not load the file. It might be corrupted or not a valid analysis file.")
        st.session_state.analysis_loaded = False # Ensure flag is false on error
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        st.session_state.analysis_loaded = False # Ensure flag is false on error
        return False

# --- Pickle Save Function ---
def save_analysis_state(filename):
    """Saves the current analysis state to a .pkl file."""
    if st.session_state.results_df is None:
        st.warning("No analysis results available to save.")
        return

    try:
        # Gather current input parameters directly from session state (should be up-to-date)
        input_params = {
            'input_domain': st.session_state.domain,
            'input_include_filters': st.session_state.input_include_filters,
            'input_exclude_filters': st.session_state.input_exclude_filters,
            'input_include_logic_any': st.session_state.input_include_logic_any,
            'input_analyze_content': st.session_state.input_analyze_content,
            'input_use_urls_too': st.session_state.input_use_urls_too,
            'input_url_weight': st.session_state.input_url_weight,
            'input_max_workers': st.session_state.input_max_workers,
            'input_request_delay': st.session_state.input_request_delay,
            'input_max_urls': st.session_state.input_max_urls,
            'input_perplexity': st.session_state.input_perplexity,
            'input_focus_k': st.session_state.input_focus_k,
            'input_radius_k': st.session_state.input_radius_k,
            'input_use_gemini': st.session_state.input_use_gemini,
            # 'input_google_api_key': st.session_state.input_google_api_key # Avoid saving API key if entered in password field
        }

        # Gather results state
        results_state = {
            'sitemaps': st.session_state.sitemaps,
            'selected_sitemaps': st.session_state.selected_sitemaps,
            'urls': st.session_state.urls,
            'url_sources': st.session_state.url_sources,
            'content_dict': st.session_state.content_dict,
            'processed_content_dict': st.session_state.processed_content_dict,
            'results_df': st.session_state.results_df,
            'focus_score': st.session_state.focus_score,
            'radius_score': st.session_state.radius_score,
            'pairwise_distances': st.session_state.pairwise_distances,
            'llm_summary': st.session_state.llm_summary,
            'centroid': st.session_state.centroid,
            'processed_content': st.session_state.processed_content,
        }

        # Combine parameters and results
        analysis_state = {**input_params, **results_state}

        with open(filename, "wb") as f:
            pickle.dump(analysis_state, f)
        st.success(f"Analysis state successfully saved to `{filename}`")

    except Exception as e:
        st.error(f"Error saving analysis state: {e}")


# Page configuration (Keep as is)
st.set_page_config(
    page_title="Topical Focus Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description (Keep as is)
st.title("🔍 Topical Focus Analyzer")
st.markdown("""
This tool analyzes the topical focus of a website by examining both URL structure and page content.
It visualizes how tightly focused or widely spread the content topics are.
**New:** You can now save and load analysis results using the options in the sidebar.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Load Previous Analysis")
    st.warning("⚠️ Loading .pkl files can be risky. Only load files you trust.")
    uploaded_file = st.file_uploader("Upload .pkl analysis file:", type=["pkl"], key="file_uploader")

    if uploaded_file is not None and not st.session_state._loading_in_progress:
        # Check if this upload is different from the last processed one to avoid loops
        if uploaded_file is not st.session_state.get('_last_uploaded_file', None):
            st.session_state._loading_in_progress = True
            load_analysis_state(uploaded_file)
            st.session_state._last_uploaded_file = uploaded_file # Store reference to prevent reload on same file
            st.session_state._loading_in_progress = False
            st.rerun() # Rerun necessary to reflect loaded state in widgets

    st.divider() # Visual separator

    st.header("Analysis Parameters")

    # Domain input - Reads from and writes to st.session_state.domain
    st.session_state.domain = st.text_input(
        "Enter a domain (e.g., example.com):",
        value=st.session_state.domain, # Read from state
        key="domain_input"
    )
    # Assign to local variable AFTER widget for use in current script run
    domain = st.session_state.domain

    # --- URL Filtering ---
    st.subheader("URL Filtering")
    with st.expander("URL Filters", expanded=True):
        # Include Filters
        st.markdown("**Include URLs containing:**")
        current_includes = st.session_state.input_include_filters[:] # Work with a copy
        for i in range(3):
            current_includes[i] = st.text_input(
                f"Include filter #{i+1}:",
                value=current_includes[i], # Read from list
                key=f"include_{i}"
            )
        st.session_state.input_include_filters = current_includes # Update state list

        # Exclude Filters
        st.markdown("**Exclude URLs containing:**")
        current_excludes = st.session_state.input_exclude_filters[:] # Work with a copy
        for i in range(3):
             current_excludes[i] = st.text_input(
                 f"Exclude filter #{i+1}:",
                 value=current_excludes[i], # Read from list
                 key=f"exclude_{i}"
            )
        st.session_state.input_exclude_filters = current_excludes # Update state list

        # Filter Logic
        filter_logic_index = 1 if not st.session_state.input_include_logic_any else 0
        filter_logic = st.radio(
            "Include filter logic:",
            ["Match ANY filter (OR)", "Match ALL filters (AND)"],
            index=filter_logic_index, # Read from state
            key='filter_logic_radio',
            help="For multiple include filters, choose whether URLs should match any or all of the filters"
        )
        st.session_state.input_include_logic_any = (filter_logic == "Match ANY filter (OR)")

    # --- Content Analysis Options ---
    st.subheader("Content Analysis Options")
    st.session_state.input_analyze_content = st.checkbox(
        "Analyze Page Content (slower but more accurate)",
        value=st.session_state.input_analyze_content, # Read from state
        key='analyze_content_cb'
    )
    # Assign to local variable AFTER widget
    analyze_content = st.session_state.input_analyze_content

    if analyze_content:
        st.session_state.input_use_urls_too = st.checkbox(
            "Also use URL paths (combined analysis)",
            value=st.session_state.input_use_urls_too, # Read from state
            key='use_urls_too_cb'
        )
        st.session_state.input_url_weight = st.slider(
            "URL Path Weight vs. Content", 0.0, 1.0,
            value=st.session_state.input_url_weight, # Read from state
            key='url_weight_slider',
            help="Higher values give more importance to URL paths vs page content"
        )

        with st.expander("Advanced Content Options"):
            st.session_state.input_max_workers = st.slider(
                "Maximum Parallel Workers", 1, 10,
                value=st.session_state.input_max_workers, # Read from state
                key='max_workers_slider',
                help="Higher values scrape pages faster but may trigger rate limits"
            )
            st.session_state.input_request_delay = st.slider(
                "Delay Between Requests (seconds)", 0.1, 5.0,
                value=st.session_state.input_request_delay, # Read from state
                key='request_delay_slider',
                help="Longer delays reduce risk of rate limiting"
            )
        # Assign local variables AFTER widgets
        use_urls_too = st.session_state.input_use_urls_too
        url_weight = st.session_state.input_url_weight
        max_workers = st.session_state.input_max_workers
        request_delay = st.session_state.input_request_delay

    else: # If content analysis is off, set dependent variables accordingly for *this run*
        use_urls_too = True
        url_weight = 1.0
        # Keep the slider values in state even if hidden, read the state values
        max_workers = st.session_state.input_max_workers
        request_delay = st.session_state.input_request_delay


    # --- Advanced Analysis Options ---
    with st.expander("Advanced Analysis Options"):
        st.session_state.input_max_urls = st.slider(
            "Maximum URLs to analyze:", 10, 1000,
            value=st.session_state.input_max_urls, # Read from state
            key='max_urls_slider',
            help="Lower values are faster but less comprehensive"
        )
        st.session_state.input_perplexity = st.slider(
            "t-SNE Perplexity:", 5, 50,
            value=st.session_state.input_perplexity, # Read from state
            key='perplexity_slider',
            help="Lower values preserve local structure, higher values preserve global structure"
        )
        st.session_state.input_focus_k = st.slider(
            "Focus Score Scaling (k1):", 1.0, 20.0,
            value=st.session_state.input_focus_k, # Read from state
            key='focus_k_slider',
            help="Higher values make the focus score more sensitive to distance variations"
        )
        st.session_state.input_radius_k = st.slider(
            "Radius Score Scaling (k2):", 1.0, 20.0,
            value=st.session_state.input_radius_k, # Read from state
            key='radius_k_slider',
            help="Higher values make the radius score more sensitive to maximum distances"
        )
    # Assign local variables AFTER widgets
    max_urls = st.session_state.input_max_urls
    perplexity = st.session_state.input_perplexity
    focus_k = st.session_state.input_focus_k
    radius_k = st.session_state.input_radius_k

    # --- Gemini Options ---
    st.session_state.input_use_gemini = st.checkbox(
        "Generate AI summary with Gemini",
        value=st.session_state.input_use_gemini, # Read from state
        key='use_gemini_cb'
    )
    use_gemini = st.session_state.input_use_gemini # Assign local variable

    if use_gemini:
        st.info("Ensure you've set your GOOGLE_API_KEY in the .env file or provide it below.")
        # Read API key into session state, but don't use it as the default 'value' for security
        st.session_state.input_google_api_key = st.text_input(
            "Google API Key (optional):",
            value="", # Always start empty in the UI
            type="password",
            key='google_api_key_input',
            help="Leave empty to use the key from your .env file"
        )
        # Use the entered value OR the previously stored value if not re-entered
        google_api_key = st.session_state.input_google_api_key
    else:
        google_api_key = "" # Ensure it's empty if Gemini is off

    st.divider()

    # --- Save Analysis Section ---
    st.header("Save Analysis")
    # Suggest a filename based on the domain
    save_filename_default = f"{st.session_state.domain.replace('.', '_')}_analysis.pkl" if st.session_state.domain else "analysis.pkl"
    save_filename = st.text_input("Save analysis as:", value=save_filename_default, key="save_filename_input")

    # Only show save button if analysis is complete and not currently loading
    can_save = st.session_state.results_df is not None and not st.session_state._loading_in_progress
    if st.button("Save Analysis State", key="save_state_button", use_container_width=True, disabled=not can_save):
        if save_filename:
            save_analysis_state(save_filename)
        else:
            st.error("Please enter a filename to save the analysis.")
    elif not can_save and st.session_state.results_df is None:
        st.caption("Run an analysis first to enable saving.")

    st.divider()

    # --- Start Analysis Section ---
    st.header("Run New Analysis")
    analyze_button = st.button("Find Sitemaps", use_container_width=True, key="find_sitemaps_button")


# --- Main Application Logic ---

# Local variables for analysis derived from session state (read AFTER widgets)
# These reflect the user's current selections for a *new* analysis run
# If analysis is loaded, these values might differ from the loaded state, which is fine.
domain = st.session_state.domain
include_filters = [f for f in st.session_state.input_include_filters if f] # Use non-empty filters
exclude_filters = [f for f in st.session_state.input_exclude_filters if f] # Use non-empty filters
include_logic_any = st.session_state.input_include_logic_any
analyze_content = st.session_state.input_analyze_content
use_urls_too = st.session_state.input_use_urls_too if analyze_content else True
url_weight = st.session_state.input_url_weight if analyze_content else 1.0
max_workers = st.session_state.input_max_workers
request_delay = st.session_state.input_request_delay
max_urls = st.session_state.input_max_urls
perplexity = st.session_state.input_perplexity
focus_k = st.session_state.input_focus_k
radius_k = st.session_state.input_radius_k
use_gemini = st.session_state.input_use_gemini
google_api_key = st.session_state.input_google_api_key # The one potentially entered

# --- Analysis Execution Block ---
# Only run if the "Find Sitemaps" button was clicked AND domain is provided
# And crucially, only if an analysis wasn't just loaded
if analyze_button and domain:
    st.session_state.analysis_loaded = False # Explicitly mark as NOT loaded state
    st.session_state._last_uploaded_file = None # Clear last loaded file tracker

    # Reset previous results before starting a new analysis
    keys_to_reset = ['sitemaps', 'selected_sitemaps', 'urls', 'url_sources', 'content_dict',
                     'processed_content_dict', 'results_df', 'focus_score', 'radius_score',
                     'pairwise_distances', 'llm_summary', 'centroid', 'processed_content']
    for key in keys_to_reset:
        st.session_state[key] = DEFAULT_SESSION_STATE.get(key) # Reset to default

    # Step 1: Find Sitemaps
    with st.spinner("Finding sitemaps..."):
        st.session_state.sitemaps = find_sitemaps(domain)

    if not st.session_state.sitemaps:
        st.error(f"No sitemaps found for {domain}. Check domain or robots.txt.")
    else:
        st.success(f"Found {len(st.session_state.sitemaps)} sitemap(s)!")
        # Reset selection specifically
        st.session_state.selected_sitemaps = []
        st.rerun() # Rerun to show sitemap selection immediately

# Step 2: Select Sitemaps (Show if sitemaps found OR if loaded state has sitemaps)
if st.session_state.sitemaps:
    st.subheader("Available Sitemaps")
    sitemap_cols = st.columns(2)
    process_button_disabled = True # Default to disabled

    with sitemap_cols[0]:
        # If analysis was loaded, just display the selected ones - no checkboxes
        if st.session_state.analysis_loaded:
            st.markdown("**Selected Sitemaps (from loaded analysis):**")
            if st.session_state.selected_sitemaps:
                 for i, sitemap in enumerate(st.session_state.selected_sitemaps):
                    st.write(f"- {sitemap}")
            else:
                st.write("None selected in loaded analysis.")
        # Otherwise (new analysis), show checkboxes for selection
        else:
            all_sitemaps = st.session_state.sitemaps
            select_all = st.checkbox("Select All Sitemaps", key="select_all_sitemaps")

            # Use list from session state for persistence during selection interaction
            current_selection = st.session_state.selected_sitemaps[:]
            new_selection = []
            for i, sitemap in enumerate(all_sitemaps):
                 # Checkbox state depends on select_all OR if it's already in the current selection list
                 is_checked = select_all or sitemap in current_selection
                 if st.checkbox(f"{i+1}. {sitemap}", value=is_checked, key=f"sitemap_{i}"):
                     new_selection.append(sitemap)

            # If selection changed via checkboxes or select_all, update session state
            if new_selection != current_selection or select_all:
                 st.session_state.selected_sitemaps = list(set(new_selection)) # Use set for unique, list for order? Order matters less here.

            # Enable button only if sitemaps are selected in a non-loaded state
            process_button_disabled = len(st.session_state.selected_sitemaps) == 0

    with sitemap_cols[1]:
        # Show selected count regardless of loaded/not loaded
        st.markdown(f"**{len(st.session_state.selected_sitemaps)} sitemaps selected**")
        if st.session_state.selected_sitemaps:
            with st.expander("View selected sitemaps"):
                for i, sitemap in enumerate(st.session_state.selected_sitemaps):
                    st.write(f"{i+1}. {sitemap}")

    # "Process Sitemaps" Button - only shown if NOT loaded
    if not st.session_state.analysis_loaded:
        process_button = st.button(
            "Process Selected Sitemaps",
            use_container_width=True,
            disabled=process_button_disabled,
            key="process_sitemaps_button"
        )

        if process_button_disabled and process_button:
            st.warning("Please select at least one sitemap to process.")

        # Step 3: Process Sitemaps (Run only if Process button clicked AND sitemaps selected AND not loaded)
        if process_button and st.session_state.selected_sitemaps:
            # --- Start of Processing Block (Original Logic Preserved) ---
            with st.spinner("Parsing sitemaps..."):
                all_urls_list = [] # Use different name to avoid scope issues
                url_sources_dict = {}

                # Define filter function locally or ensure it's accessible
                def url_passes_filters_local(url, inc_filters, exc_filters, inc_logic_any):
                     for exclude in exc_filters:
                         if exclude and exclude.lower() in url.lower(): return False
                     if not inc_filters: return True # Pass if no include filters
                     if inc_logic_any:
                         return any(include.lower() in url.lower() for include in inc_filters if include)
                     else:
                         return all(include.lower() in url.lower() for include in inc_filters if include)

                for sitemap_url in st.session_state.selected_sitemaps:
                    st.info(f"Parsing sitemap: {sitemap_url}")
                    sitemap_urls = parse_sitemap(sitemap_url)
                    filtered_urls = [
                        url for url in sitemap_urls
                        if url_passes_filters_local(url, include_filters, exclude_filters, include_logic_any) # Use current filters
                    ]
                    for url in filtered_urls: url_sources_dict[url] = sitemap_url
                    all_urls_list.extend(filtered_urls)
                    st.success(f"Found {len(filtered_urls)} URLs in sitemap (after filtering)")

                seen = set()
                unique_urls_list = [url for url in all_urls_list if not (url in seen or seen.add(url))]

                if len(unique_urls_list) > max_urls: # Use current max_urls
                    st.warning(f"Limiting analysis to {max_urls} URLs out of {len(unique_urls_list)} found.")
                    unique_urls_list = unique_urls_list[:max_urls]
                    url_sources_dict = {url: source for url, source in url_sources_dict.items() if url in unique_urls_list}

                # Store results in session state
                st.session_state.urls = unique_urls_list
                st.session_state.url_sources = url_sources_dict

                # Display sample (Original Logic)
                if unique_urls_list:
                    with st.expander("Sample of URLs found (click to expand)"):
                         for i, url in enumerate(unique_urls_list[:10]):
                             source = url_sources_dict.get(url, "Unknown")
                             st.write(f"{i+1}. {url} (from: {source})")
                         if len(unique_urls_list) > 10: st.write(f"... and {len(unique_urls_list) - 10} more")
                time.sleep(0.5) # Original delay

            if not st.session_state.urls:
                st.error("No URLs found in the selected sitemaps after applying filters.")
            else:
                st.success(f"Found {len(st.session_state.urls)} unique URLs across all selected sitemaps!")

                # --- Content Extraction Step (Original Logic Preserved) ---
                if analyze_content: # Use current setting
                    with st.spinner("Extracting page content... This may take a while..."):
                        st.info(f"Extracting content from {len(st.session_state.urls)} pages with {max_workers} parallel workers...") # Use current settings
                        st.warning("This step may take several minutes...")

                        progress_bar = st.progress(0)
                        status_text = st.empty() # Placeholder for text updates

                        # --- Original Function: extract_with_progress ---
                        def extract_with_progress(urls_to_scrape):
                            results = {}
                            total = len(urls_to_scrape)
                            for i, url in enumerate(urls_to_scrape):
                                try:
                                    # content = extract_main_content(url) # Assumes this function exists
                                    # Simulate extraction if module not fully set up
                                    content = f"Simulated content for {url}"
                                    results[url] = content

                                    progress = (i + 1) / total
                                    progress_bar.progress(progress)
                                    if (i + 1) % 5 == 0 or (i + 1) == total:
                                         status_text.text(f"Processed {i + 1} of {total} URLs...") # Update status text

                                    if i < total - 1: time.sleep(request_delay) # Use current delay

                                except Exception as e:
                                    st.error(f"Error extracting content from {url}: {str(e)}")
                                    results[url] = "" # Store empty string on error
                            status_text.text("Sequential extraction complete.")
                            return results
                        # --- End Original Function ---

                        # --- Original Conditional Extraction Logic ---
                        if len(st.session_state.urls) <= 20:
                            status_text.text("Using sequential extraction for small site...")
                            content_dict_result = extract_with_progress(st.session_state.urls)
                        else:
                            status_text.text("Using parallel extraction...")
                            # Assuming batch_extract_content handles progress internally or add callback if supported
                            content_dict_result = batch_extract_content(
                                st.session_state.urls,
                                max_workers=max_workers, # Use current setting
                                delay=request_delay # Use current setting
                            )
                            progress_bar.progress(1.0) # Ensure completion
                            status_text.text("Parallel extraction complete.")
                        # --- End Original Conditional Logic ---

                        st.session_state.content_dict = content_dict_result

                        # Process content (Original Logic)
                        processed_dict = {}
                        for url, content in content_dict_result.items():
                             processed_dict[url] = preprocess_text_for_analysis(content or "") # Handle None/empty
                        st.session_state.processed_content_dict = processed_dict

                        # Show stats and preview (Original Logic)
                        content_lengths = [len(str(c)) for c in content_dict_result.values()]
                        avg_len = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                        empty_count = sum(1 for length in content_lengths if length == 0)
                        st.info(f"Content extraction complete! Average length: {avg_len:.1f} chars")
                        if empty_count > 0: st.warning(f"Could not extract content from {empty_count} URLs")

                        with st.expander("Sample of extracted content (click to expand)"):
                            for i, (url, content) in enumerate(list(content_dict_result.items())[:3]):
                                source = st.session_state.url_sources.get(url, "Unknown")
                                st.write(f"**URL:** {url} (from: {source})")
                                preview = str(content)[:500] + "..." if len(str(content)) > 500 else str(content)
                                st.text_area(f"Content preview {i+1}", preview, height=150, key=f"extract_preview_{i}")
                        time.sleep(0.5) # Original delay
                else: # If not analyzing content
                    st.session_state.content_dict = None
                    st.session_state.processed_content_dict = {}


                # --- Vectorization Step (Original Logic Preserved) ---
                with st.spinner("Vectorizing content..."):
                    st.info("Processing data through vectorization...")
                    # Determine message based on current settings
                    if analyze_content and use_urls_too: st.info("Using combined URL path and content analysis")
                    elif analyze_content: st.info("Using content-only analysis")
                    else: st.info("Using URL path-only analysis")

                    # Use processed_content_dict for vectorization if content analysis is on
                    content_input = st.session_state.processed_content_dict if analyze_content else None

                    url_list_res, processed_paths_res, processed_content_list_res, vectorizer_res, matrix_res = vectorize_urls_and_content(
                        st.session_state.urls,
                        content_dict=content_input, # Pass processed dict
                        use_url_paths=use_urls_too or not analyze_content, # Derived from current settings
                        use_content=analyze_content, # Current setting
                        url_weight=url_weight # Current setting
                    )

                    # Store the list of processed content used by the vectorizer
                    st.session_state.processed_content = processed_content_list_res

                    # Show details (Original Logic)
                    matrix_type = "TF-IDF" if hasattr(vectorizer_res, "idf_") else "Similarity/Embedding" # Check TFIDF attribute
                    st.info(f"Vectorization complete. Matrix type: {matrix_type}, Shape: {matrix_res.shape}")

                    with st.expander("Vectorization Details (click to expand)"):
                         # Explanations (Keep original markdown)
                         st.subheader("How Vectorization Works")
                         if analyze_content and use_urls_too: st.markdown("**Combined URL + Content Analysis:**...")
                         elif analyze_content: st.markdown("**Content-Only Analysis:**...")
                         else: st.markdown("**URL-Only Analysis:**...")
                         # Samples (Keep original logic)
                         st.subheader("Sample Data")
                         for i in range(min(3, len(url_list_res))):
                              url = url_list_res[i]
                              source = st.session_state.url_sources.get(url, "Unknown")
                              st.write(f"**URL:** {url} (from: {source})")
                              st.write(f"**Processed Path:** {processed_paths_res[i]}")
                              if analyze_content and i < len(processed_content_list_res):
                                   preview = processed_content_list_res[i][:200] + "..."
                                   st.write(f"**Processed Content:** {preview}")
                              st.write("---")
                    time.sleep(0.5) # Original delay


                # --- Dimensionality Reduction (Original Logic Preserved) ---
                with st.spinner("Reducing dimensions (t-SNE)... This may take time..."):
                    st.info("Starting t-SNE dimensionality reduction...")
                    st.warning("This step may take several minutes for larger datasets!")
                    progress_placeholder = st.empty()
                    progress_placeholder.text("Running t-SNE...")

                    # Ensure perplexity is valid based on current setting and matrix size
                    num_samples = matrix_res.shape[0]
                    safe_perplexity = min(perplexity, num_samples - 1) if num_samples > 1 else 1
                    if safe_perplexity != perplexity and num_samples > 1:
                         st.warning(f"Perplexity adjusted from {perplexity} to {safe_perplexity} due to dataset size ({num_samples}).")

                    coordinates_df_res, centroid_res = reduce_dimensions_and_find_centroid(
                        matrix_res,
                        perplexity=safe_perplexity
                    )
                    st.session_state.centroid = centroid_res # Store result

                    progress_placeholder.empty() # Clear progress text
                    st.info(f"t-SNE complete. Centroid: ({centroid_res[0]:.2f}, {centroid_res[1]:.2f})")
                    time.sleep(0.5) # Original delay


                # --- Calculate Metrics (Original Logic Preserved) ---
                with st.spinner("Calculating metrics..."):
                    # Pass current k values from sliders
                    results_df_res, focus_score_res, radius_score_res, pairwise_dist_matrix_res = calculate_metrics(
                        url_list=url_list_res,
                        processed_paths=processed_paths_res,
                        coordinates_df=coordinates_df_res,
                        centroid=centroid_res,
                        k1=focus_k, # Current slider value
                        k2=radius_k  # Current slider value
                    )

                    # Add content preview (Original logic, uses processed_content list from state)
                    if analyze_content and st.session_state.processed_content and len(st.session_state.processed_content) == len(results_df_res):
                        results_df_res['content_preview'] = [
                            (p[:200] + "..." if len(p) > 200 else p)
                            for p in st.session_state.processed_content
                        ]

                    # Add source sitemap (Original Logic)
                    results_df_res['source_sitemap'] = results_df_res['url'].apply(
                        lambda url: st.session_state.url_sources.get(url, "Unknown")
                    )

                    # Store results in session state
                    st.session_state.results_df = results_df_res
                    st.session_state.focus_score = focus_score_res
                    st.session_state.radius_score = radius_score_res
                    st.session_state.pairwise_distances = pairwise_dist_matrix_res

                    st.info(f"Metrics calculated. Focus Score: {focus_score_res:.1f}, Radius Score: {radius_score_res:.1f}")


                # --- Generate LLM Summary (Original Logic Preserved) ---
                if use_gemini: # Use current setting
                    with st.spinner("Generating AI summary..."):
                         # Get API key (use entered or env var)
                         api_key_to_use = google_api_key if google_api_key else os.getenv("GOOGLE_API_KEY")
                         if not api_key_to_use:
                              st.error("Gemini summary requested, but GOOGLE_API_KEY is missing.")
                              st.session_state.llm_summary = "Error: API key not provided."
                         else:
                              # Prepare data (Original logic)
                              sorted_df = st.session_state.results_df.sort_values('distance_from_centroid')
                              top_focused = sorted_df['url'].head(5).tolist()
                              top_divergent = sorted_df['url'].tail(5).tolist()
                              page_types = sorted_df['page_type'].value_counts().to_dict()

                              st.session_state.llm_summary = get_gemini_summary(
                                  api_key_to_use,
                                  st.session_state.focus_score,
                                  st.session_state.radius_score,
                                  len(url_list_res),
                                  top_focused,
                                  top_divergent,
                                  page_types
                              )
                              st.success("AI summary generated!")
            # --- End of Processing Block ---
            st.success("Analysis processing complete!")
            st.rerun() # Rerun to display results immediately

# --- Display Results ---
# This section runs if results_df exists (from new run OR loaded state)
if st.session_state.results_df is not None:
    if not st.session_state.analysis_loaded :
         st.success("Displaying new analysis results.")
    else:
         st.success("Displaying loaded analysis results.")

    # Determine tabs based on available data (Original logic for Content Inspector)
    tab_titles = ["Overview", "URL Details", "Visual Map (t-SNE)", "Cannibalization/Clusters"]
    content_available = st.session_state.content_dict is not None or st.session_state.processed_content_dict # Check both raw and processed
    if content_available:
         tab_titles.append("Content Inspector")

    tabs = st.tabs(tab_titles)

    # Tab 1: Overview (Keep original content)
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1: st.metric("Site Focus Score", f"{st.session_state.focus_score:.1f}/100", help="...")
        with col2: st.metric("Site Radius Score", f"{st.session_state.radius_score:.1f}/100", help="...")

        # Sitemap Distribution (Original logic)
        unique_sources = set(st.session_state.url_sources.values())
        if len(unique_sources) > 1 and 'source_sitemap' in st.session_state.results_df.columns:
            st.subheader("Sitemap Distribution")
            sitemap_counts = st.session_state.results_df['source_sitemap'].value_counts()
            fig_sitemap = px.pie(values=sitemap_counts.values, names=sitemap_counts.index, title="URLs by Source Sitemap")
            st.plotly_chart(fig_sitemap, use_container_width=True)

        # LLM Summary (Original logic)
        if st.session_state.llm_summary:
            st.subheader("Analysis")
            st.markdown(st.session_state.llm_summary)

        # Page Type Distribution (Original logic)
        if 'page_type' in st.session_state.results_df.columns:
             st.subheader("Page Type Distribution")
             page_type_counts = st.session_state.results_df['page_type'].value_counts()
             if not page_type_counts.empty:
                 fig_pie = px.pie(values=page_type_counts.values, names=page_type_counts.index, title="Content Distribution by Page Type")
                 st.plotly_chart(fig_pie, use_container_width=True)
             else:
                 st.info("No page type information available.")


        # Distance Distribution (Original logic)
        if 'distance_from_centroid' in st.session_state.results_df.columns:
             st.subheader("Distance Distribution")
             fig_hist = px.histogram(st.session_state.results_df, x="distance_from_centroid", nbins=30, title="Distribution of URL Distances", labels={"distance_from_centroid": "Distance"})
             st.plotly_chart(fig_hist, use_container_width=True)

        # Focused/Divergent URLs (Original logic)
        col1, col2 = st.columns(2)
        if 'distance_from_centroid' in st.session_state.results_df.columns:
             with col1:
                  st.subheader("Most Focused URLs")
                  focused_df = st.session_state.results_df.nsmallest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']].rename(columns={'distance_from_centroid': 'distance'})
                  st.dataframe(focused_df, hide_index=True, use_container_width=True)
             with col2:
                  st.subheader("Most Divergent URLs")
                  divergent_df = st.session_state.results_df.nlargest(10, 'distance_from_centroid')[['url', 'page_type', 'source_sitemap', 'distance_from_centroid']].rename(columns={'distance_from_centroid': 'distance'})
                  st.dataframe(divergent_df, hide_index=True, use_container_width=True)

    # Tab 2: URL Details (Keep original content)
    with tabs[1]:
        st.subheader("URL Analysis Details")
        # Filters (Original logic)
        filter_cols = st.columns([2, 1, 1])
        with filter_cols[0]: search_term = st.text_input("Search URLs:", key="details_search")
        with filter_cols[1]:
            # Check if 'page_type' column exists before creating filter options
            all_page_types = ["All"]
            if 'page_type' in st.session_state.results_df.columns:
                all_page_types.extend(sorted(st.session_state.results_df['page_type'].unique().tolist()))
            selected_page_type = st.selectbox("Filter by Page Type:", all_page_types, key="details_type_filter")
        with filter_cols[2]:
             # Check if 'source_sitemap' column exists
            all_sitemaps = ["All"]
            sitemap_filter_disabled = True
            if 'source_sitemap' in st.session_state.results_df.columns:
                 unique_sitemaps = sorted(st.session_state.results_df['source_sitemap'].unique().tolist())
                 all_sitemaps.extend(unique_sitemaps)
                 sitemap_filter_disabled = len(unique_sitemaps) <= 1
            selected_sitemap = st.selectbox("Filter by Sitemap:", all_sitemaps, key="details_sitemap_filter", disabled=sitemap_filter_disabled)

        # Apply Filters (Original logic - ensure columns exist before filtering)
        filtered_df = st.session_state.results_df.copy()
        if search_term and 'url' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['url'].str.contains(search_term, case=False)]
        if selected_page_type != "All" and 'page_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['page_type'] == selected_page_type]
        if selected_sitemap != "All" and 'source_sitemap' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['source_sitemap'] == selected_sitemap]

        # --- MODIFICATION START ---
        # Prepare for display: Add 'x' and 'y' to the list of desired columns
        display_columns = ['url', 'page_type', 'source_sitemap', 'page_depth', 'distance_from_centroid', 'x', 'y']
        if 'content_preview' in filtered_df.columns:
             display_columns.append('content_preview') # Add preview if it exists

        # Ensure all desired display columns actually exist in the filtered dataframe before selection
        existing_display_columns = [col for col in display_columns if col in filtered_df.columns]
        # --- MODIFICATION END ---

        if existing_display_columns:
             display_df = filtered_df[existing_display_columns] # Select only existing columns

             # Rename columns for display
             rename_map = {
                 'distance_from_centroid': 'distance',
                 'page_depth': 'depth',
                 'source_sitemap': 'sitemap'
             }
             # Only rename columns that actually exist in display_df
             actual_rename_map = {k: v for k, v in rename_map.items() if k in display_df.columns}
             display_df = display_df.rename(columns=actual_rename_map)

             # --- MODIFICATION START ---
             # Update column config to include 'x' and 'y'
             column_config_dict = {
                 "url": st.column_config.TextColumn("URL", help="The URL analyzed"),
                 "page_type": st.column_config.TextColumn("Page Type", help="Categorization based on URL structure"),
                 "sitemap": st.column_config.TextColumn("Sitemap", help="Source sitemap the URL was found in"),
                 "depth": st.column_config.NumberColumn("Depth", help="URL path depth"),
                 "distance": st.column_config.NumberColumn("Distance", help="Euclidean distance from the topic centroid in the t-SNE map", format="%.3f"),
                 "x": st.column_config.NumberColumn("Coord X", help="X-coordinate from the t-SNE visualization", format="%.3f"),
                 "y": st.column_config.NumberColumn("Coord Y", help="Y-coordinate from the t-SNE visualization", format="%.3f"),
                 "content_preview": st.column_config.TextColumn("Content Preview", help="First 200 characters of processed content (if available)"),
             }

             # Filter column config to only include columns present in the final display_df
             final_column_config = {col: config for col, config in column_config_dict.items() if col in display_df.columns}
             # --- MODIFICATION END ---

             st.dataframe(
                 # Sort by distance if the column exists after potential renaming
                 display_df.sort_values('distance') if 'distance' in display_df.columns else display_df,
                 use_container_width=True,
                 column_config=final_column_config # Use the filtered config
             )
             st.info(f"Showing {len(filtered_df)} URLs out of {len(st.session_state.results_df)} total.")
        else:
             st.warning("No columns available for display based on current data or filters.")


    # Tab 3: Visual Map (t-SNE) (Keep original content)
    with tabs[2]:
        st.subheader("Topical Map Visualization")
        # Options (Original logic)
        visual_cols = st.columns([2, 1, 1])
        with visual_cols[0]:
            color_options = {"Distance from Centroid": "distance_from_centroid", "Page Type": "page_type", "Page Depth": "page_depth", "Source Sitemap": "source_sitemap"}
            # Filter available color options based on columns present in results_df
            available_color_options = {k: v for k, v in color_options.items() if v in st.session_state.results_df.columns}
            if not available_color_options:
                st.warning("No data columns available for coloring points.")
                color_by_selection = None
            else:
                color_by_selection = st.selectbox("Color points by:", list(available_color_options.keys()), key="viz_color_select")

        with visual_cols[1]: point_size = st.slider("Point Size:", 3, 15, 8, key="viz_size_slider")
        with visual_cols[2]:
             all_sitemaps_viz = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist()) if 'source_sitemap' in st.session_state.results_df.columns else ["All"]
             viz_sitemap = st.selectbox("Show sitemap:", all_sitemaps_viz, key="viz_sitemap_filter", disabled=len(all_sitemaps_viz) <= 1)

        # Filter data (Original logic)
        viz_df = st.session_state.results_df.copy()
        if viz_sitemap != "All" and 'source_sitemap' in viz_df.columns: viz_df = viz_df[viz_df['source_sitemap'] == viz_sitemap]

        # Create plot if possible (Original logic + checks)
        if color_by_selection and 'x' in viz_df.columns and 'y' in viz_df.columns:
            color_column = available_color_options[color_by_selection]
            hover_cols = [col for col in ["page_type", "distance_from_centroid", "page_depth", "source_sitemap"] if col in viz_df.columns]

            if color_column in ["page_type", "source_sitemap"]: # Categorical coloring
                 fig = px.scatter(viz_df, x="x", y="y", color=color_column, hover_name="url", hover_data=hover_cols, title="t-SNE Clustering", size_max=point_size)
            else: # Continuous coloring
                 fig = px.scatter(viz_df, x="x", y="y", color=color_column, color_continuous_scale="Viridis", hover_name="url", hover_data=hover_cols, title="t-SNE Clustering", size_max=point_size)

            # Add centroid (Original logic)
            if st.session_state.centroid is not None:
                 fig.add_trace(go.Scatter(x=[st.session_state.centroid[0]], y=[st.session_state.centroid[1]], mode="markers", marker=dict(symbol="star", size=15, color="red"), name="Centroid"))

            fig.update_layout(height=700, hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
            st.info("""
            **How to interpret this visualization:**
            * Each point represents a URL from the sitemap
            * Points that cluster together have similar topics
            * The star marker represents the "topic centroid" - the center of all topics
            * Distances from the centroid reflect how focused or divergent each URL is
            * Colors help identify patterns in content structure
            """) # Keep original info text
        elif 'x' not in viz_df.columns or 'y' not in viz_df.columns:
            st.warning("t-SNE coordinates (x, y) not found in results. Cannot generate map.")
        else:
            st.warning("Select a valid coloring option.")


    # Tab 4: Cannibalization/Clusters (Keep original content)
    with tabs[3]:
        st.subheader("Content Cannibalization Analysis")
        st.markdown("This tab helps identify potentially duplicate content...")
        # Filters (Original logic)
        cann_cols = st.columns([2, 1, 1])
        with cann_cols[0]: threshold = st.slider("Distance Threshold:", min_value=0.01, max_value=2.0, value=0.5, step=0.01, help="...", key="cann_threshold") # Adjusted min/step potentially
        with cann_cols[1]: max_pairs = st.number_input("Max pairs to display:", 5, 100, 20, key="cann_max_pairs")
        with cann_cols[2]:
             all_sitemaps_cann = ["All pairs", "Same sitemap", "Different sitemaps"]
             cann_filter = st.selectbox("Filter pairs:", all_sitemaps_cann, key="cann_sitemap_filter", disabled=len(st.session_state.results_df['source_sitemap'].unique()) <= 1 if 'source_sitemap' in st.session_state.results_df.columns else True)

        # Analysis (Original logic + Preserved Expander)
        if st.session_state.results_df is not None and st.session_state.pairwise_distances is not None:

             # --- Start: Preserved Distance Matrix Statistics ---
             with st.expander("Distance Matrix Statistics"):
                 dist_matrix = st.session_state.pairwise_distances
                 st.write(f"Matrix shape: {dist_matrix.shape}")
                 # Calculate stats safely for potentially non-square matrices or if empty
                 if dist_matrix.size > 0:
                      non_zero_distances = dist_matrix[dist_matrix > 1e-6] # Avoid floating point zeros
                      st.write(f"Min non-zero distance: {non_zero_distances.min():.4f}" if non_zero_distances.size > 0 else "N/A")
                      st.write(f"Max distance: {dist_matrix.max():.4f}")
                      st.write(f"Mean non-zero distance: {non_zero_distances.mean():.4f}" if non_zero_distances.size > 0 else "N/A")
                      st.write(f"Number of close-to-zero distances: {(dist_matrix <= 1e-6).sum()}")

                      # Histogram of non-zero distances
                      if non_zero_distances.size > 0:
                          fig_dist_hist = px.histogram(
                              x=non_zero_distances.flatten(),
                              nbins=50,
                              title="Distribution of Pairwise Distances (excluding self-comparisons)",
                              labels={"x": "Distance"}
                          )
                          st.plotly_chart(fig_dist_hist)
                      else:
                           st.write("No non-zero distances to plot.")
                 else:
                      st.write("Distance matrix is empty.")
            # --- End: Preserved Distance Matrix Statistics ---

             duplicates = find_potential_duplicates(
                 st.session_state.results_df,
                 st.session_state.pairwise_distances,
                 threshold
             )

             # Filter duplicates (Original logic, check column exists)
             if cann_filter != "All pairs" and duplicates and 'source_sitemap' in st.session_state.results_df.columns:
                filtered_duplicates = []
                # Need url_sources mapping for this filter
                url_to_source = pd.Series(st.session_state.results_df.source_sitemap.values, index=st.session_state.results_df.url).to_dict()
                for dup in duplicates:
                     url1, url2 = dup['url1'], dup['url2']
                     source1 = url_to_source.get(url1, "Unknown")
                     source2 = url_to_source.get(url2, "Unknown")
                     if cann_filter == "Same sitemap" and source1 == source2: filtered_duplicates.append(dup)
                     elif cann_filter == "Different sitemaps" and source1 != source2: filtered_duplicates.append(dup)
                duplicates = filtered_duplicates

             # Display (Original logic)
             displayed_duplicates = duplicates[:max_pairs] if duplicates else []
             if not displayed_duplicates: st.info("No potential duplicates found with current settings.")
             else:
                 total_count = len(duplicates)
                 st.success(f"Found {total_count} potential duplicates! Showing top {len(displayed_duplicates)}.")
                 for i, row in enumerate(displayed_duplicates):
                      url1, url2 = row['url1'], row['url2']
                      # Need url_sources mapping again
                      url_to_source = pd.Series(st.session_state.results_df.source_sitemap.values, index=st.session_state.results_df.url).to_dict()
                      source1 = url_to_source.get(url1, "Unknown")
                      source2 = url_to_source.get(url2, "Unknown")

                      title = f"Pair {i+1}: Distance {row['distance']:.3f}"
                      if source1 != source2: title += f" (Cross-Sitemap: {source1} -> {source2})"
                      with st.expander(title):
                           # Display pair details (Original layout)
                           cols=st.columns(2)
                           with cols[0]:
                                st.markdown(f"**URL 1:** [{url1}]({url1})")
                                if 'processed_path' in row: st.text(f"Processed path: {row['path1']}") # Check if path info present
                                st.text(f"Source: {source1}")
                                if 'content_preview' in st.session_state.results_df.columns:
                                     content1 = st.session_state.results_df.loc[st.session_state.results_df['url'] == url1, 'content_preview'].values
                                     if len(content1)>0: st.text_area("Content preview 1:", content1[0], height=150, key=f"cann_p1_{i}")
                           with cols[1]:
                                st.markdown(f"**URL 2:** [{url2}]({url2})")
                                if 'processed_path' in row: st.text(f"Processed path: {row['path2']}") # Check if path info present
                                st.text(f"Source: {source2}")
                                if 'content_preview' in st.session_state.results_df.columns:
                                     content2 = st.session_state.results_df.loc[st.session_state.results_df['url'] == url2, 'content_preview'].values
                                     if len(content2)>0: st.text_area("Content preview 2:", content2[0], height=150, key=f"cann_p2_{i}")


    # Tab 5: Content Inspector (Keep original content, check availability more robustly)
    if "Content Inspector" in tab_titles:
        with tabs[4]:
            st.subheader("Content Inspector")
            # Check if *any* content dictionary has data
            if not st.session_state.content_dict and not st.session_state.processed_content_dict:
                st.info("No content was extracted or loaded. Run analysis with content extraction or load a file containing content.")
            else:
                st.markdown("Examine raw and processed content...")
                # Filters (Original logic)
                inspector_cols = st.columns([2, 1, 1])
                with inspector_cols[0]: url_search = st.text_input("Search URLs:", key="inspector_search")
                with inspector_cols[1]:
                     all_page_types_insp = ["All"] + sorted(st.session_state.results_df['page_type'].unique().tolist()) if 'page_type' in st.session_state.results_df.columns else ["All"]
                     inspector_page_type = st.selectbox("Filter by Page Type:", all_page_types_insp, key="inspector_type_filter")
                with inspector_cols[2]:
                     all_sitemaps_insp = ["All"] + sorted(st.session_state.results_df['source_sitemap'].unique().tolist()) if 'source_sitemap' in st.session_state.results_df.columns else ["All"]
                     inspector_sitemap = st.selectbox("Filter by Sitemap:", all_sitemaps_insp, key="inspector_sitemap_filter", disabled=len(all_sitemaps_insp) <= 1)

                # Determine list of URLs with *any* content
                urls_with_content = set(st.session_state.content_dict.keys() if st.session_state.content_dict else [])
                urls_with_content.update(st.session_state.processed_content_dict.keys() if st.session_state.processed_content_dict else [])
                all_inspectable_urls = sorted(list(urls_with_content))

                # Filter URLs (Original logic, use the combined list)
                filtered_inspect_urls = all_inspectable_urls[:]
                if url_search: filtered_inspect_urls = [url for url in filtered_inspect_urls if url_search.lower() in url.lower()]
                # Ensure results_df exists and has the columns before filtering
                if st.session_state.results_df is not None:
                    if inspector_page_type != "All" and 'page_type' in st.session_state.results_df.columns:
                         urls_matching_type = st.session_state.results_df[st.session_state.results_df['page_type'] == inspector_page_type]['url'].tolist()
                         filtered_inspect_urls = [url for url in filtered_inspect_urls if url in urls_matching_type]
                    if inspector_sitemap != "All" and 'source_sitemap' in st.session_state.results_df.columns:
                         urls_matching_sitemap = st.session_state.results_df[st.session_state.results_df['source_sitemap'] == inspector_sitemap]['url'].tolist()
                         filtered_inspect_urls = [url for url in filtered_inspect_urls if url in urls_matching_sitemap]


                if filtered_inspect_urls:
                    selected_url_inspect = st.selectbox("Select URL to inspect:", filtered_inspect_urls, key="inspector_url_select")
                    if selected_url_inspect:
                        # Get data (Original logic, handle missing dicts)
                        raw_content = st.session_state.content_dict.get(selected_url_inspect, "N/A") if st.session_state.content_dict else "N/A"
                        processed_content = st.session_state.processed_content_dict.get(selected_url_inspect, "N/A")
                        source_sitemap = st.session_state.url_sources.get(selected_url_inspect, "Unknown") # Use original url_sources

                        # Get info from results_df if available
                        distance, page_type, percentile = None, None, None
                        if st.session_state.results_df is not None and selected_url_inspect in st.session_state.results_df['url'].values:
                            url_row = st.session_state.results_df[st.session_state.results_df['url'] == selected_url_inspect].iloc[0]
                            distance = url_row.get('distance_from_centroid')
                            page_type = url_row.get('page_type')
                            # Percentile calculation (Original logic)
                            if distance is not None:
                                sorted_df = st.session_state.results_df.sort_values('distance_from_centroid').reset_index()
                                rank_list = sorted_df.index[sorted_df['url'] == selected_url_inspect].tolist()
                                if rank_list: percentile = (rank_list[0] / (len(sorted_df)-1)) * 100 if len(sorted_df) > 1 else 0


                        # Display info (Original logic)
                        st.subheader("URL Information")
                        info_cols=st.columns(4)
                        with info_cols[0]: st.markdown(f"**URL:** [{selected_url_inspect}]({selected_url_inspect})")
                        with info_cols[1]: st.markdown(f"**Sitemap:** {source_sitemap}")
                        with info_cols[2]: st.markdown(f"**Type:** {page_type or 'N/A'}")
                        with info_cols[3]:
                            if distance is not None: st.markdown(f"**Distance:** {distance:.3f}")
                            if percentile is not None: st.markdown(f"**Focus Rank:** {percentile:.1f}%")
                            if percentile is not None:
                                 if percentile < 20: st.success("Highly Focused")
                                 elif percentile > 80: st.warning("Highly Divergent")

                        # Display content tabs (Original logic)
                        st.subheader("Content Inspector")
                        content_tabs = st.tabs(["Raw Content", "Processed Content", "Content Statistics"])
                        with content_tabs[0]:
                            st.markdown("**Raw content**:")
                            st.text_area("Raw Content", str(raw_content), height=300, key="inspect_raw")
                            st.info(f"Length: {len(str(raw_content))} chars")
                        with content_tabs[1]:
                            st.markdown("**Processed content** (used for vectorization):")
                            st.text_area("Processed Content", str(processed_content), height=300, key="inspect_proc")
                            st.info(f"Length: {len(str(processed_content))} chars")
                            # Processing changes (Original logic)
                            if raw_content != "N/A" and processed_content != "N/A":
                                 st.subheader("Processing Changes")
                                 len_raw, len_proc = len(str(raw_content)), len(str(processed_content))
                                 reduc_pct = 100 - (len_proc / len_raw * 100) if len_raw > 0 else 0
                                 st.markdown(f"Reduced by **{reduc_pct:.1f}%**")
                                 # Removed words example (Original logic)
                                 if len_raw > 0 :
                                      raw_words = set(str(raw_content).lower().split())
                                      proc_words = set(str(processed_content).split())
                                      removed = raw_words - proc_words
                                      if removed: st.write("**Removed examples:** " + ", ".join(list(removed)[:50]))
                        with content_tabs[2]:
                            st.subheader("Content Statistics")
                            if processed_content != "N/A" and len(str(processed_content)) > 0:
                                 # Word Freq (Original logic)
                                 words = str(processed_content).split()
                                 word_freq = {}
                                 for w in words:
                                      if len(w) > 1: word_freq[w] = word_freq.get(w, 0) + 1

                                 if word_freq:
                                      freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values('Frequency', ascending=False).reset_index(drop=True)
                                      st.dataframe(freq_df.head(20), use_container_width=True)
                                      # Chart (Original logic)
                                      fig_freq = px.bar(freq_df.head(15), x='Word', y='Frequency', title="Top 15 Words")
                                      st.plotly_chart(fig_freq, use_container_width=True)
                                      # Stats (Original logic)
                                      stats_cols = st.columns(3)
                                      unique_count = len(word_freq)
                                      word_count = len(words)
                                      diversity = unique_count / word_count if word_count > 0 else 0
                                      with stats_cols[0]: st.metric("Word Count", word_count)
                                      with stats_cols[1]: st.metric("Unique Words", unique_count)
                                      with stats_cols[2]: st.metric("Lexical Diversity", f"{diversity:.2f}")
                                 else: st.warning("Not enough words for frequency analysis.")
                            else: st.warning("No processed content for statistics.")

                else: # No URLs match filter
                    st.warning("No URLs match your filter criteria or no content available to inspect.")

                # Bulk export (Original logic, use combined URL list)
                with st.expander("Bulk Export Options"):
                    st.markdown("Download extracted content and analysis data as CSV")
                    if st.button("Generate Content CSV", key="export_csv_button"):
                        if not all_inspectable_urls:
                             st.warning("No content URLs available to export.")
                        else:
                             export_data = []
                             for url in all_inspectable_urls:
                                  raw = st.session_state.content_dict.get(url, "") if st.session_state.content_dict else ""
                                  processed = st.session_state.processed_content_dict.get(url, "")
                                  source = st.session_state.url_sources.get(url, "Unknown")
                                  distance, page_type = None, None
                                  if st.session_state.results_df is not None and url in st.session_state.results_df['url'].values:
                                      url_row = st.session_state.results_df[st.session_state.results_df['url'] == url].iloc[0]
                                      distance = url_row.get('distance_from_centroid')
                                      page_type = url_row.get('page_type')

                                  export_data.append({
                                     'URL': url, 'Source': source, 'Type': page_type, 'Distance': distance,
                                     'Raw Len': len(str(raw)), 'Proc Len': len(str(processed)),
                                     'Raw Preview': str(raw)[:500]+"...", 'Processed Text': processed
                                  })
                             export_df = pd.DataFrame(export_data)
                             csv = export_df.to_csv(index=False).encode('utf-8')
                             b64 = base64.b64encode(csv).decode()
                             href = f'<a href="data:file/csv;base64,{b64}" download="site_content_analysis.csv">Download Content Analysis CSV</a>'
                             st.markdown(href, unsafe_allow_html=True)


# Footer (Keep as is)
st.markdown("---")
st.markdown("**Topical Focus Analyzer** | Built with Python, Streamlit, and Content Analysis")
# --- END OF FILE ---
