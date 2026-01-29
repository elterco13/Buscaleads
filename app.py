import streamlit as st
import pandas as pd
from duckduckgo_search import DDGS
import openai
import google.generativeai as genai
import os
import time
import json
import re

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

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🔧 Configuration")
    
    llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Google Gemini"])
    api_key = st.text_input(f"{llm_provider} API Key", type="password")
    
    st.markdown("---")
    industry = st.text_input("Industry / Niche", value="Hotels")
    location = st.text_input("Location", value="Madrid")
    
    target_persona_options = [
        "Decision Maker", "General Manager", "Procurement/Purchasing", 
        "Marketing Director", "Wholesaler", "Distributor", "Custom/Other"
    ]
    target_persona = st.selectbox("Target Persona", target_persona_options)
    
    if target_persona == "Custom/Other":
        target_persona = st.text_input("Specify Target Persona", value="CEO")
    
    user_context = st.text_input("User Context / Intent", 
                               help="Who is using this data? (e.g., 'A sales rep looking for cold calls')")
    
    additional_notes = st.text_area("Additional Instructions / Notes", 
                                  help="Specific Constraints (e.g., 'Ignore chains, only independent businesses')")
    
    save_path = st.text_input("Local Save Path", value="./leads_output/")
    
    search_btn = st.button("Search & Scrape", type="primary")

# --- Backend Logic ---

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

def generate_search_queries(industry, location, persona, context, notes, provider):
    prompt = f"""
    Act as a search query expert. I need to find contact information for '{persona}' in the '{industry}' industry in '{location}'.
    Context: {context}
    Constraints: {notes}
    
    Generate 3 distinct, high-quality search queries optimized for DuckDuckGo to find specific leads. 
    Focus on finding directories, company lists, or direct contact pages. 
    Do NOT include specific company names unless they are examples. 
    Format the output as a valid JSON list of strings.
    Example: ["site:linkedin.com {persona} {industry} {location}", "{industry} directory {location} contact"]
    """
    
    try:
        if provider == "OpenAI":
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response.choices[0].message.content
        elif provider == "Google Gemini":
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
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
    with DDGS() as ddgs:
        for query in queries:
            try:
                # Fetching generic results (titles, snippets, urls)
                search_results = list(ddgs.text(query, max_results=5))
                for res in search_results:
                    res['query_used'] = query
                    results.append(res)
                time.sleep(1) # Respect rate limits
            except Exception as e:
                st.warning(f"Error searching for '{query}': {e}")
    return results

def extract_and_filter(raw_results, context, notes, provider):
    if not raw_results:
        return pd.DataFrame()
        
    results_json = json.dumps(raw_results[:15]) # Limit payload size
    
    prompt = f"""
    I have a list of web search results. I need you to extract valid B2B leads and filter out irrelevant ones.
    
    User Context: {context}
    Special Notes: {notes}
    
    Raw Search Data:
    {results_json}
    
    Task:
    1. Analyze each search result snippet/title.
    2. Extract: Entity Name, Role/Contact (if found), and verify if it matches the criteria.
    3. Return a JSON list of objects with these keys:
       - "Entity Name": Name of the business or person.
       - "Role/Contact": Any contact info or role mentioned (e.g., "Manager", "email@example.com").
       - "Source URL": The URL from the result.
       - "Context/Notes": Why is this a good lead based on the user context?
       - "Verified Information": The specific snippet text that validates this lead.
    
    Strictly return ONLY valid JSON.
    """
    
    try:
        content = ""
        if provider == "OpenAI":
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            content = response.choices[0].message.content
        elif provider == "Google Gemini":
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            content = response.text
            
        # Clean and parse JSON
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return pd.DataFrame()

# --- Main App Execution ---
st.title("🚀 Real-Time B2B Lead Scraper & Analyzer")
st.markdown(f"**Targeting:** `{target_persona}` in `{industry}` ({location})")

if search_btn:
    if not api_key:
        st.error("Please provide an API Key.")
        st.stop()
        
    if not configure_llm(llm_provider, api_key):
        st.error("Failed to configure LLM provider.")
        st.stop()
        
    st.status("🤖 Formulating Search Queries...", expanded=True)
    with st.spinner("Asking LLM for best search strategies..."):
        queries = generate_search_queries(industry, location, target_persona, user_context, additional_notes, llm_provider)
        st.write("Queries Generated:", queries)
        
    st.status("🌐 Searching the Web (DuckDuckGo)...", expanded=True)
    with st.spinner("Scraping real-time data..."):
        raw_results = perform_search(queries)
        st.write(f"Found {len(raw_results)} raw results.")
        
    st.status("🧠 Analyzing & Filtering Leads...", expanded=True)
    with st.spinner("LLM is processing results..."):
        df = extract_and_filter(raw_results, user_context, additional_notes, llm_provider)
        
    if not df.empty:
        st.success(f"Successfully extracted {len(df)} qualified leads!")
        
        # Display
        st.dataframe(
            df,
            column_config={
                "Source URL": st.column_config.LinkColumn("Source URL"),
            }
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
