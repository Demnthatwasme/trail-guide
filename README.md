🚵 Trail & Gear Advisor

A Conversational Retrieval-Augmented Generation (RAG) application built to answer complex mountain bike maintenance, component torque spec, and trail guide questions using custom domain manuals.

🌟 Key Features

Smart Retrieval Architecture: Powered by ChromaDB vector storage using local Hugging Face embeddings (all-MiniLM-L6-v2).

Two-Stage Retrieval (Re-ranking): Employs a Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) to re-rank vector search results, ensuring maximum semantic precision before passing context to the LLM.

Conversational Memory: Built with create_history_aware_retriever to intelligently reformulate follow-up user queries based on context.

Source Transparency: Displays expandable inline citations with exact PDF source file names and page numbers for answer verification.

Streamlit UI: Clean, responsive chat interface with session reset capabilities and quick-start prompts.

🛠️ Tech Stack

Language: Python 3.13

LLM Orchestration: LangChain (langchain-classic, langchain-google-genai)

LLM: Google Gemini

Vector Store: ChromaDB

Embeddings & Re-ranking: HuggingFace Transformers (sentence-transformers, cross-encoder)

Frontend: Streamlit

🚀 Quickstart Guide

1. Prerequisites

Ensure you have Python 3.10+ and a Google Gemini API Key.

2. Installation

Clone the repository:

git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME


Set up a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


3. Environment Setup

Create a .env file in the root directory and add your Google API key:

GOOGLE_API_KEY=your_actual_gemini_api_key_here


4. Process Manuals & Run App

Place your PDF manuals in the designated folder, process them into vector embeddings, and launch the UI:

# Process PDFs and generate vector storage
python process_manuals.py

# Launch the Streamlit application
streamlit run app.py


🔒 Security Note

This repository uses .gitignore to prevent secret keys (.env) and local database files (chroma_db/) from being exposed publicly.

⚠️ Disclaimer

Copyright & Asset Ownership: The PDF manuals, component guides, and trail maps used to populate the database for this project are the property of their respective manufacturers, authors, and copyright holders. They are utilized in this repository strictly for educational, non-commercial, and demonstration purposes to showcase RAG architecture. No ownership is claimed over the original documents.