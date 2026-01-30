import streamlit as st
import pandas as pd
from ddgs import DDGS
import openai
import google.generativeai as genai
import os
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Page Config
st.set_page_config(page_title="Real-Time B2B Lead Scraper", page_icon="🔍", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for search history
if 'previous_queries' not in st.session_state:
    st.session_state.previous_queries = []
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🔧 Configuration")
    
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Google Gemini"])
    api_key = st.text_input(f"{llm_provider} API Key", type="password")
    
    st.markdown("---")
    industry = st.text_input("Industry / Niche", value="Hotels")
    location = st.text_input("Location", value="Madrid")
    
    st.markdown("---")
    
    # Search Focus - helps LLM choose appropriate platforms
    search_focus_options = [
        "Local Businesses (Restaurants, Gyms, Shops, etc.)",
        "Tourism & Hospitality (Hotels, Tours, Activities)",
        "B2B Professionals (Executives, Consultants)",
        "Service Providers (Agencies, Contractors)",
        "Wholesale & Distribution",
        "General / Not Sure"
    ]
    search_focus = st.selectbox(
        "Search Focus", 
        search_focus_options,
        index=5,
        help="This helps the AI choose the best search platforms (e.g., Google Maps for local businesses, LinkedIn for professionals)"
    )
    
    target_persona_options = [
        "Not Specified", "Decision Maker", "General Manager", "Procurement/Purchasing", 
        "Marketing Director", "Wholesaler", "Distributor", "Custom/Other"
    ]
    target_persona = st.selectbox("Target Persona", target_persona_options, index=0)
    
    if target_persona == "Custom/Other":
        target_persona = st.text_input("Specify Target Persona", value="CEO")
    
    user_context = st.text_input("User Context / Intent", 
                               help="Who is using this data? (e.g., 'A sales rep looking for cold calls')")
    
    additional_notes = st.text_area("Additional Instructions / Notes", 
                                  help="Specific Constraints (e.g., 'Ignore chains, only independent businesses')")
    
    expected_results = st.number_input("Expected Results Count", min_value=5, max_value=100, value=20, step=5,
                                       help="Number of qualified leads you want to find")
    
    save_path = st.text_input("Local Save Path", value="./leads_output/")
    
    search_btn = st.button("Search & Scrape", type="primary")
    
    # Show re-search button only after first search
    if st.session_state.search_count > 0:
        st.markdown("---")
        st.info(f"Previous searches: {st.session_state.search_count}")
        research_btn = st.button("🔄 Search Again (Different Criteria)", type="secondary")
    else:
        research_btn = False

# --- Backend Logic ---

def scrape_url_content(url, timeout=8):
    """
    Scrapes a single URL and extracts contact information.
    Returns dict with emails, phones, and other contact data.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract all text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Find emails using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(email_pattern, text_content)))
        
        # Find phone numbers (international and Spanish formats)
        phone_pattern = r'(?:\+34|0034)?\s*[6-9]\d{2}\s*\d{2}\s*\d{2}\s*\d{2}|(?:\+\d{1,3})?\s*\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}'
        phones = list(set(re.findall(phone_pattern, text_content)))
        # Clean phone numbers
        phones = [re.sub(r'\s+', ' ', p.strip()) for p in phones if len(p.replace(' ', '')) >= 9]
        
        # Try to find contact page link
        contact_links = []
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            if any(word in href for word in ['contact', 'contacto', 'about', 'acerca']):
                full_url = urljoin(url, link['href'])
                contact_links.append(full_url)
        
        # If no contact info found on main page but contact link exists, try scraping that
        if (not emails and not phones) and contact_links:
            try:
                contact_url = contact_links[0]
                contact_response = requests.get(contact_url, headers=headers, timeout=5)
                contact_soup = BeautifulSoup(contact_response.text, 'lxml')
                contact_text = contact_soup.get_text(separator=' ', strip=True)
                
                emails = list(set(re.findall(email_pattern, contact_text)))
                phones = list(set(re.findall(phone_pattern, contact_text)))
                phones = [re.sub(r'\s+', ' ', p.strip()) for p in phones if len(p.replace(' ', '')) >= 9]
            except:
                pass
        
        return {
            'emails': emails[:3],  # Limit to 3 emails
            'phones': phones[:3],  # Limit to 3 phones
            'success': True
        }
        
    except requests.Timeout:
        return {'emails': [], 'phones': [], 'success': False, 'error': 'timeout'}
    except requests.RequestException as e:
        return {'emails': [], 'phones': [], 'success': False, 'error': str(e)[:50]}
    except Exception as e:
        return {'emails': [], 'phones': [], 'success': False, 'error': str(e)[:50]}

def configure_llm(provider, key):
    if not key:
        return False
    if provider == "OpenAI":
        openai.api_key = key
        return True
    elif provider == "Google Gemini":
        genai.configure(api_key=key)
        try:
            # Debug: List available models
            models = [m.name for m in genai.list_models()]
            st.sidebar.write("Available Models:", models) # Debugging enabled
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini: {e}")
            return False
    return False

def generate_search_queries(industry, location, persona, context, notes, expected_results, search_focus, previous_queries, provider):
    # Calculate number of queries needed (estimate 3-5 results per query)
    num_queries = max(3, min(10, (expected_results // 4) + 1))
    
    previous_queries_text = ""
    if previous_queries:
        previous_queries_text = f"\n\nIMPORTANT: These queries were ALREADY USED in previous searches. DO NOT repeat them or similar variations:\n{json.dumps(previous_queries, indent=2)}\n\nGenerate COMPLETELY DIFFERENT search strategies, using different platforms, operators, or keyword combinations."
    
    # Platform recommendations based on search focus
    platform_guidance = ""
    if "Local Businesses" in search_focus:
        platform_guidance = "\n\nPLATFORM FOCUS: Prioritize local business directories, Google Maps, Yelp, local review sites, business associations, and local government directories. AVOID LinkedIn unless explicitly needed."
    elif "Tourism" in search_focus:
        platform_guidance = "\n\nPLATFORM FOCUS: Prioritize TripAdvisor, Booking.com, Expedia, GetYourGuide, Viator, tourism boards, and local tourism websites. AVOID LinkedIn unless explicitly needed."
    elif "B2B Professionals" in search_focus:
        platform_guidance = "\n\nPLATFORM FOCUS: Prioritize LinkedIn, corporate directories, industry associations, professional networks, and company websites."
    elif "Service Providers" in search_focus:
        platform_guidance = "\n\nPLATFORM FOCUS: Prioritize service directories, industry-specific platforms, Google Business, yellow pages, and professional associations."
    elif "Wholesale" in search_focus:
        platform_guidance = "\n\nPLATFORM FOCUS: Prioritize wholesale directories, B2B marketplaces, trade associations, and industry-specific platforms."
    else:
        platform_guidance = "\n\nPLATFORM FOCUS: Use a diverse mix of platforms - include both professional networks and local business directories."
    
    # System prompt for Phase 1: Query Orchestration
    system_prompt = """You are a Senior Commercial Intelligence Analyst specialized in B2B market research and data extraction.
Your role is to generate strategic search queries that will discover high-quality business leads."""
    
    user_prompt = f"""PHASE 1: QUERY ORCHESTRATION

Generate {num_queries} strategic search queries to find leads with the following parameters:
- Industry/Niche: {industry}
- Location: {location}
- Target Persona: {persona}
- User Context: {context}
- Special Notes: {notes}
- Expected Results: {expected_results}
- Search Focus: {search_focus}
{platform_guidance}
{previous_queries_text}

SEARCH MATRIX STRATEGY:
Your queries should include a mix of:
1. Direct Search: [{industry}] [{location}]
2. Authority Search: site:.org OR site:.gov "{industry}" {location} association OR chamber
3. Persona Search: site:linkedin.com "{persona}" {industry} {location} (only if B2B focus)
4. Deep Web Footprints: "{industry}" {location} "contact us" OR "about" OR "team"

CRITICAL RULES:
- Generate DIVERSE queries using different operators (site:, intitle:, inurl:)
- Vary platforms based on Search Focus
- Do NOT repeat previous queries or similar variations
- Output ONLY a valid JSON array of strings

Example output format:
["site:tripadvisor.com {industry} {location}", "{industry} directory {location}", "intitle:contact {industry} {location}"]
"""
    
    try:
        if provider == "OpenAI":
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content
        elif provider == "Google Gemini":
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            content = response.text
            
        # Extract JSON list from text
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            # Fallback
            return [f"{industry} {persona} {location} contact email", f"{industry} directory {location}"]
            
    except Exception as e:
        st.error(f"Error formulating queries: {e}")
        return []

def perform_search(queries):
    results = []
    for query in queries:
        try:
            st.info(f"🔍 Searching: '{query}'")
            # Fetching generic results (titles, snippets, urls)
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=10, region='wt-wt')
            
            count = 0
            for res in search_results:
                res['query_used'] = query
                results.append(res)
                count += 1
            
            st.success(f"✓ Found {count} results for this query")
            time.sleep(2)  # Increased delay to avoid rate limiting
            
        except Exception as e:
            st.warning(f"⚠️ Error searching for '{query}': {str(e)}")
            time.sleep(3)  # Longer delay after error
    
    return results

def scrape_websites(search_results, max_urls=15):
    """
    Takes search results and scrapes actual website content.
    Returns enriched results with contact information.
    """
    scraped_data = []
    urls_to_scrape = []
    
    # Collect unique URLs
    seen_urls = set()
    for result in search_results:
        url = result.get('href') or result.get('link')
        if url and url not in seen_urls:
            urls_to_scrape.append({'url': url, 'title': result.get('title', ''), 'snippet': result.get('body', '')})
            seen_urls.add(url)
            if len(urls_to_scrape) >= max_urls:
                break
    
    st.info(f"🌐 Found {len(urls_to_scrape)} candidate websites. Starting deep scraping...")
    
    progress_bar = st.progress(0)
    for idx, item in enumerate(urls_to_scrape):
        url = item['url']
        domain = urlparse(url).netloc
        
        st.caption(f"🔍 Scraping {idx+1}/{len(urls_to_scrape)}: {domain}")
        
        scraped = scrape_url_content(url)
        
        if scraped['success']:
            if scraped['emails'] or scraped['phones']:
                scraped_data.append({
                    'url': url,
                    'title': item['title'],
                    'snippet': item['snippet'],
                    'emails': scraped['emails'],
                    'phones': scraped['phones']
                })
                st.success(f"✓ {domain}: Found {len(scraped['emails'])} emails, {len(scraped['phones'])} phones")
            else:
                st.caption(f"⚠️ {domain}: No contact info found")
        else:
            st.caption(f"❌ {domain}: Failed ({scraped.get('error', 'unknown')})")
        
        progress_bar.progress((idx + 1) / len(urls_to_scrape))
        time.sleep(1)  # Delay between requests
    
    progress_bar.empty()
    return scraped_data

def extract_and_filter(scraped_data, context, notes, industry, provider):
    """
    Uses LLM to organize and filter scraped data into structured leads.
    Now works with REAL scraped data that includes emails and phones.
    """
    if not scraped_data:
        return pd.DataFrame()
        
    # Prepare data for LLM - already has emails and phones from scraping
    data_summary = []
    for item in scraped_data[:20]:  # Limit to prevent token overflow
        data_summary.append({
            'url': item['url'],
            'title': item['title'],
            'emails': item['emails'],
            'phones': item['phones'],
            'description': item['snippet'][:200]  # Truncate
        })
    
    results_json = json.dumps(data_summary, indent=2)
    
    # System prompt for Phases 2-4
    system_prompt = """You are a Senior Commercial Intelligence Analyst specialized in B2B market research and lead qualification.

Your mission is to transform raw scraped web data into high-quality, verified, strategically-scored business leads.

CORE PRINCIPLES:
1. ANTI-HALLUCINATION: Never invent data. If information is not present, leave field empty.
2. EVIDENCE-BASED: All data must be traceable to the source snippet.
3. STRATEGIC FILTERING: Prioritize leads that match user context and notes.
4. QUALITY OVER QUANTITY: Better 5 perfect leads than 20 mediocre ones."""
    
    user_prompt = f"""PHASES 2-4: EXTRACTION, SCORING & VERIFICATION

User Parameters:
- Industry: {industry}
- User Context: {context}
- Special Notes: {notes}

Scraped Website Data (with REAL contact information):
{results_json}

YOUR TASKS:

PHASE 2: DATA HARVESTING
For each website, extract:
- Organization Name: Official business name
- Contact Info: Use ACTUAL scraped emails/phones (DO NOT INVENT!)
- Category: Business type (Restaurant, Hotel, Tour Operator, etc.)

PHASE 3: RELEVANCE SCORING (0-100 points)
Evaluate each lead:
- Niche Fit (+40 pts): Exactly matches industry/search intent?
- Persona Match (+30 pts): Is target decision-maker identifiable?
- Notes Compliance (+30 pts): Meets special requirements?
- PENALTY (-50 pts): Chain/franchise when independent was requested

PHASE 4: ANTI-HALLUCINATION VERIFICATION
CRITICAL RULES:
- If email not in scraped data → leave "Email" empty or "Not found"
- If phone not in scraped data → leave "Phone" empty or "Not found"
- Cite evidence from snippet to validate the lead exists
- Verify URL domain matches organization name

OUTPUT FORMAT (JSON only):
Return a JSON array of objects with these EXACT keys:
- "Organization": Company/business name
- "Contact Name": Person name (leave empty if not available)
- "Job Title": Role/position (leave empty if not available)  
- "Email": FIRST email from scraped data (or empty string)
- "Phone": FIRST phone from scraped data (or empty string)
- "Why Good Fit": Brief justification based on context ({context})
- "Category": Business type
- "Website": Source URL
- "Relevance Score": Numeric score 0-100
- "Evidence": Text snippet validating this lead

PRIORITY RULES:
- If "Special Notes" provided, they override all other logic
- If User Context = "Sales/Vendedor", prioritize leads with direct contact info
- If User Context = "Analyst/Analista", prioritize accuracy and category precision
- Only include leads with score ≥ 60

OUTPUT: Return ONLY valid JSON, no markdown, no explanations."""
    
    try:
        content = ""
        if provider == "OpenAI":
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2  # Lower temperature for factual, non-creative output
            )
            content = response.choices[0].message.content
        elif provider == "Google Gemini":
            model = genai.GenerativeModel(
                'gemini-2.5-flash',
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            content = response.text
            
        # Clean and parse JSON
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        
        # Keep only essential columns for display (hide scoring internals)
        df = pd.DataFrame(data)
        if 'Relevance Score' in df.columns and 'Evidence' in df.columns:
            # Optionally drop these for cleaner UI, or keep for transparency
            # df = df.drop(columns=['Relevance Score', 'Evidence'])
            pass
        
        return df
        
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return pd.DataFrame()

# --- Main App Logic ---
st.title("🚀 Real-Time B2B Lead Scraper & Analyzer")
st.markdown(f"**Targeting:** `{target_persona}` in `{industry}` ({location})")

# Main execution - triggered by either button
if search_btn or research_btn:
    if not api_key:
        st.error("Please provide an API Key.")
        st.stop()
        
    if not configure_llm(llm_provider, api_key):
        st.error("Failed to configure LLM provider.")
        st.stop()
    
    # Show which mode we're in
    if research_btn:
        st.info("🔄 Re-searching with different criteria to find new leads...")
        
    st.status("🤖 Formulating Search Queries...", expanded=True)
    with st.spinner("Asking LLM for best search strategies..."):
        queries = generate_search_queries(
            industry, location, target_persona, user_context, 
            additional_notes, expected_results, search_focus,
            st.session_state.previous_queries,  # Pass previous queries
            llm_provider
        )
        st.write("Queries Generated:", queries)
        
        # Save queries to history
        st.session_state.previous_queries.extend(queries)
        st.session_state.search_count += 1
        
    st.status("🌐 Searching the Web (DuckDuckGo)...", expanded=True)
    with st.spinner("Scraping real-time data..."):
        raw_results = perform_search(queries)
        st.write(f"Found {len(raw_results)} raw results.")
    
    # NEW: Deep scraping phase
    st.status("🕷️ Deep Scraping Websites...", expanded=True)
    scraped_websites = scrape_websites(raw_results, max_urls=15)
    st.write(f"Successfully scraped {len(scraped_websites)} websites with contact data.")
        
    st.status("🧠 Analyzing & Filtering Leads...", expanded=True)
    with st.spinner("LLM is processing results..."):
        df = extract_and_filter(scraped_websites, user_context, additional_notes, industry, llm_provider)
        
    if not df.empty:
        st.success(f"Successfully extracted {len(df)} qualified leads!")
        
        # Display
        st.dataframe(
            df,
            column_config={
                "Website": st.column_config.LinkColumn("Website"),
                "Email": st.column_config.TextColumn("Email"),
            },
            use_container_width=True
        )
        
        # Auto-Save
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"leads_{industry}_{location}_{timestamp}.csv"
        full_path = os.path.join(save_path, filename)
        
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        st.toast(f"Saved to {full_path}", icon="💾")
        
        # Download Button
        csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "📥 Download CSV",
            csv,
            filename,
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("No valid leads found after filtering. Try adjusting your inputs.")
