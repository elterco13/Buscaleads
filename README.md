# 🚀 Real-Time B2B Lead Scraper & Analyzer

A powerful Streamlit application that finds **real, verified B2B leads** using DuckDuckGo Search and processes them with LLMs (OpenAI or Google Gemini) to ensure high-quality, hallucination-free data.

## 🌟 Features

*   **Real-Time Scrubbing:** Uses `duckduckgo-search` to fetch live web results. No stagnant databases.
*   **AI-Powered Filtering:** Integrates OpenAI (GPT) or Google Gemini to parse snippet data and filter out irrelevant results.
*   **Smart Query Generation:** Automatically converts your industry/persona inputs into optimized search queries.
*   **Targeted Search:** Specific inputs for Industry, Location, Role, and Context.
*   **Instant Export:** Auto-saves results to CSV and provides a direct download button.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/b2b-lead-scraper.git
    cd b2b-lead-scraper
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

1.  **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Configure the Sidebar:**
    *   **LLM Provider:** Choose OpenAI or Google Gemini.
    *   **API Key:** Enter your valid API key.
    *   **Industry & Location:** e.g., "Solar Panel Installers" in "Austin, TX".
    *   **Persona:** e.g., "Owner" or "Procurement Manager".
    *   **Context:** Explain your goal (e.g., "Selling wholesale components").

3.  **Click "Search & Scrape"**:
    *   The app will generate queries, search the web, and extract data.
    *   Results will appear in the main table.
    *   Data is automatically saved to the `leads_output/` folder.

## 📦 Dependencies

*   `streamlit` - UI Framework
*   `pandas` - Data manipulation
*   `duckduckgo-search` - Search API
*   `openai` - LLM Client
*   `google-generativeai` - LLM Client

## ⚠️ Disclaimer

This tool is for educational and legitimate business research purposes only. Respect `robots.txt` and terms of service of websites you visit.
