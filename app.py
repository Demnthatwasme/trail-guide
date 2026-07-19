import os
import streamlit as st
from dotenv import load_dotenv  # <-- Add this import
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever

# --- 1. Setup & Load Secrets ---
# This looks for the hidden .env file and loads its variables into your system environment
load_dotenv()

# Check to ensure the key was loaded properly before running the app
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Error: GOOGLE_API_KEY not found. Please make sure your .env file is set up correctly.")
    st.stop()

# ... (The rest of your app.py code remains exactly the same!)

# --- 1. Page Configuration ---
st.set_page_config(page_title="Trail & Gear Advisor", page_icon="🚵", layout="centered")
st.title("🚵 Mountain Bike Trail & Gear Advisor")
st.markdown("Ask me anything about bike maintenance, torque specs, or trail guides!")

# --- 2. Cache the AI Setup ---
# @st.cache_resource ensures the DB and Model only load once, making the web app fast
@st.cache_resource
def load_rag_pipeline():
    
    # Load Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # UPGRADE: Better AI Retrieval using MMR (Maximal Marginal Relevance)
    # MMR fetches diverse documents instead of just the closest ones, 
    # preventing the AI from reading the exact same paragraph 3 times.
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    
    # Load LLM
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    system_prompt = """
    You are an expert mountain bike mechanic and trail guide. 
    Use the provided retrieved context to answer the user's question. 
    If the context does not contain the answer, say "I don't have that information in my manuals." 
    Do not guess torque specs or trail directions. Answer in a friendly, helpful tone.

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

rag_chain = load_rag_pipeline()

# --- 3. Chat Interface State ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle User Input ---
if user_query := st.chat_input("E.g., How do I bleed Shimano XT brakes?"):
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Digging through the manuals..."):
            # Fetch the answer from your RAG chain
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            # Display the answer
            st.markdown(answer)
            
            # Display the sources hidden inside an expander
            with st.expander("Sources used"):
                for doc in response["context"]:
                    source_name = doc.metadata.get('source', 'Unknown')
                    page_num = doc.metadata.get('page', 'Unknown')
                    st.caption(f"- {source_name} (Page {page_num})")
                    
    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})