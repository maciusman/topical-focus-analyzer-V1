import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()

def format_url_list(urls, max_urls=5):
    """Format a list of URLs for inclusion in the prompt."""
    formatted_list = ""
    for i, url in enumerate(urls[:max_urls], 1):
        formatted_list += f"{i}. {url}\n"
    
    if len(urls) > max_urls:
        formatted_list += f"... and {len(urls) - max_urls} more\n"
    
    return formatted_list

def get_gemini_summary(api_key, focus_score, radius_score, total_urls, top_focused_urls, top_divergent_urls, page_type_distribution=None):
    """
    Generate a descriptive summary using Gemini.
    
    Args:
        api_key (str): Google API key for Gemini
        focus_score (float): Site focus score (0-100)
        radius_score (float): Site radius score (0-100)
        total_urls (int): Total number of URLs analyzed
        top_focused_urls (list): List of the most focused URLs (lowest distance from centroid)
        top_divergent_urls (list): List of the most divergent URLs (highest distance from centroid)
        page_type_distribution (dict, optional): Distribution of page types
        
    Returns:
        str: Generated text summary
    """
    # Use provided API key or get from environment
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: No API key provided. Please set GOOGLE_API_KEY in your environment or provide it as a parameter."
    
    try:
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Select the model - using Gemini 2.5 Pro 
        # Note: Model name might need updating based on availability, e.g., 'gemini-1.5-pro-latest'
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
        
        # *** REVISED PROMPT START ***
        prompt = f"""
        **Act as an expert Website Content Strategist and SEO Analyst.**

        Your task is to analyze the provided data about a website's content structure and topical relevance. Your audience is the **website owner**, who may not be an SEO expert. Therefore, your explanation must be **clear, concise, and easy to understand**, avoiding overly technical jargon whenever possible. If technical terms are necessary, explain them simply.

        **Your goal is to provide a comprehensive summary that includes:**
        1.  An explanation of the key metrics (Focus Score, Radius Score).
        2.  Insights derived from the data (what the scores and URL examples mean for *this specific site*).
        3.  A practical, actionable plan the site owner can follow.

        **DATA FOR ANALYSIS:**

        *   **Total URLs Analyzed:** {total_urls}
            *   *Explanation: This is the number of pages on the site included in this analysis.*

        *   **Site Focus Score:** {focus_score:.2f} / 100
            *   *Explanation: This score measures how tightly connected your site's content is to a central theme. A higher score (closer to 100) means your content is very consistent and focused on a specific topic or niche. A lower score suggests the content might be scattered across different, less related topics.*

        *   **Site Radius Score:** {radius_score:.2f} / 100
            *   *Explanation: This score indicates the breadth or range of topics covered on your site, relative to the main theme. A higher score suggests your site explores a wider variety of related sub-topics. A lower score means the site sticks very closely to the core topic with less variation.*

        *   **Most Focused URLs (Examples closest to the site's core theme):**
            {format_url_list(top_focused_urls)}
            *   *Insight: These pages likely represent the very heart of what your website is about.*

        *   **Most Divergent URLs (Examples furthest from the site's core theme):**
            {format_url_list(top_divergent_urls)}
            *   *Insight: These pages are the least related to your site's main topic. They might be necessary (like 'Contact Us') or could indicate content that's off-topic.*
        """

        if page_type_distribution:
            prompt += "\n*   **Page Type Distribution (How content is categorized):**\n"
            for page_type, count in page_type_distribution.items():
                percentage = (count / total_urls) * 100 if total_urls > 0 else 0
                prompt += f"    - {page_type}: {count} pages ({percentage:.1f}%)\n"
            prompt += "    *   *Insight: This shows the balance of different types of content (e.g., products, blog posts, informational pages) on your site.*\n"

        # Optional: Add cluster info if available (ensure cluster_info is defined earlier)
        if cluster_info: # Check if cluster_info was passed and is not None/empty
             prompt += f"\n*   **Content Clustering Insights:** {cluster_info}\n"
             prompt += "    *   *Insight: This indicates how the site's content groups together based on semantic similarity. Refer to the interactive visualization if provided.*\n"

        prompt += """
        **ANALYSIS AND ACTION PLAN REQUIRED:**

        Provide the analysis DIRECTLY using the following structure.

        **1. Executive Summary (Understand Your Site's Topical Health):**
            *   Start with a simple, high-level sentence summarizing the site's overall topical focus and breadth based on the scores.
            *   Explain what the **Site Focus Score ({focus_score:.2f})** specifically means for *this* website in plain English. Is the content tightly knit or more varied?
            *   Explain what the **Site Radius Score ({radius_score:.2f})** specifically means for *this* website. Does it cover a wide range of related subjects, or is it narrowly specialized?
            *   Briefly describe how the focus and radius scores work together for this site (e.g., "Highly focused on a narrow topic," "Covers a broad area but stays well-connected," "Seems a bit scattered across different themes").

        **2. Key Insights from Your Content:**
            *   Based on the **Most Focused URLs**, what appears to be the core subject matter or purpose of this website?
            *   What do the **Most Divergent URLs** suggest? Are these expected outliers (like 'About Us', 'Privacy Policy'), or do they represent content potentially diluting the site's focus? Could they be opportunities to link back to core topics?
            *   (If Page Type Distribution provided) How does the mix of page types relate to the focus and radius scores? Does the distribution support the site's apparent goals (e.g., lots of product pages for an e-commerce site, lots of blog posts for an informational site)?

        **3. Action Plan (Steps to Improve or Maintain Focus):**
            *   Provide **2-4 specific, actionable recommendations** based *directly* on the data analysis above.
            *   For each recommendation, clearly state **what** the site owner should do and **why** it's suggested based on the scores or URL examples (e.g., "Recommendation: Strengthen internal links from [Divergent URL example type] back to core [Focused URL example type] pages. Why: This can help tie potentially stray content back to your main theme, improving overall focus.").
            *   Phrase recommendations as clear steps (e.g., "Review content on divergent pages...", "Consider creating new content on sub-topics related to...", "Audit internal linking between...").
            *   Ensure the recommendations are practical for a typical website owner.

        **IMPORTANT:**
        *   **Output ONLY the structured analysis below (Executive Summary, Key Insights, Action Plan).** Do NOT include any introductory sentences, meta-commentary about the analysis process, explanations of how you are generating the response, or concluding conversational remarks (like "Let me know if you have questions").
        *   Go straight to generating section "1. Executive Summary".
        *   Use straightforward language. Pretend you are explaining this directly to the business owner over coffee.
        *   Base your entire analysis *strictly* on the data provided. Do not invent information or assume external context.
        *   The goal is to be helpful, insightful, and provide clear direction within the requested structure.
        """
        # *** REVISED PROMPT END ***
        # Generate the summary
        response = model.generate_content(prompt)

        # Handle potential safety blocks or empty responses
        if not response.parts:
             # Check candidate for block reason if available
            if response.candidates and response.candidates[0].finish_reason != 'STOP':
                return f"Error generating summary: Content blocked due to {response.candidates[0].finish_reason}. Prompt text might need adjustment."
            else:
                return "Error generating summary: Received an empty response from the model."

        summary = response.text.strip()
        return summary

    except Exception as e:
        # Provide more context on the error if possible
        # Be cautious about logging sensitive parts of the prompt or API key in production
        print(f"DEBUG: Error encountered during Gemini API call. Error: {str(e)}")
        return f"Error generating summary: An unexpected error occurred ({type(e).__name__}). Please check API key validity, model availability, and prompt content."


# Optional function to test the summarizer
def test_summarizer():
    # Test data
    test_data = {
        'focus_score': 78.5,
        'radius_score': 45.2,
        'total_urls': 125,
        'top_focused_urls': [
            'https://example.com/products/widgets/blue-widget',
            'https://example.com/products/widgets/red-widget',
            'https://example.com/products/widgets/heavy-duty-widget',
            'https://example.com/services/widget-installation',
            'https://example.com/products/widgets/'
        ],
        'top_divergent_urls': [
            'https://example.com/blog/company-picnic-2023',
            'https://example.com/about-us/team/ceo',
            'https://example.com/contact',
            'https://example.com/careers/open-positions',
            'https://example.com/blog/unrelated-industry-news'
        ],
        'page_type_distribution': {
            'Product': 80,
            'Service': 5,
            'Blog': 15,
            'Informational': 10, # (e.g., About, Contact, Careers)
            'Category': 15 # Added for more context
        }
    }
    
    # Calculate total URLs from distribution if more accurate
    total_urls_from_dist = sum(test_data['page_type_distribution'].values())
    test_data['total_urls'] = total_urls_from_dist # Update total_urls

    # Try to get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        print("Testing with API key from environment...")
        summary = get_gemini_summary(
            api_key,
            test_data['focus_score'],
            test_data['radius_score'],
            test_data['total_urls'],
            test_data['top_focused_urls'],
            test_data['top_divergent_urls'],
            test_data['page_type_distribution']
        )
        print("\n--- Generated Summary ---")
        print(summary)
        print("--- End of Summary ---")
    else:
        print("\n--- WARNING ---")
        print("No GOOGLE_API_KEY found in environment variables (.env file).")
        print("Please set the GOOGLE_API_KEY to test the summarizer function.")
        print("---------------")


if __name__ == "__main__":
    test_summarizer()