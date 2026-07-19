import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 1. Setup & Load Secrets ---
# This looks for the hidden .env file and loads its variables into your system environment
load_dotenv()

# Check to ensure the key was loaded properly before running the app
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Error: GOOGLE_API_KEY not found. Please make sure your .env file is set up correctly.")
    st.stop()

# --- 2. Page Configuration ---
st.set_page_config(page_title="Trail & Gear Advisor", page_icon="🚵", layout="centered")
st.title("🚵 Mountain Bike Trail & Gear Advisor")
st.markdown("Ask me anything about bike maintenance, torque specs, or trail guides!")

# --- 3. Cache the AI Setup ---
# @st.cache_resource ensures the DB and Model only load once, making the web app fast
@st.cache_resource
def load_rag_pipeline():
    
    # A. Load Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # B. Better AI Retrieval using a Re-ranker
    base_retriever = db.as_retriever(search_kwargs={"k": 10})
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

    # C. Combine them: Chroma fetches 10, the compressor shrinks it to 3
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )    

    # D. Load LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # E. Define the Contextualizer Prompt (Memory Step 1)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # F. Wrap the Re-ranker retriever with the memory logic
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # G. The Final Answer Prompt (Memory Step 2)
    qa_system_prompt = """
    You are an expert mountain bike mechanic and trail guide. 
    Use the provided retrieved context to answer the user's question. 
    If the context does not contain the answer, say "I don't have that information in my manuals." 
    Do not guess torque specs or trail directions. Answer in a friendly, helpful tone.

    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # H. Build the Final RAG Chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

rag_chain = load_rag_pipeline()

# --- 4. Chat Interface State ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# We need a separate history list just for LangChain's internal memory format
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Handle User Input ---
if user_query := st.chat_input("E.g., How do I bleed Shimano XT brakes?"):
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Save user message to UI history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Digging through the manuals..."):
            
            # Fetch the answer from your RAG chain, passing the accumulated history
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.langchain_history
            })
            answer = response["answer"]
            
            # Display the answer
            st.markdown(answer)
            
            # Display the sources hidden inside an expander
            with st.expander("Sources used"):
                for doc in response["context"]:
                    source_name = doc.metadata.get('source', 'Unknown')
                    page_num = doc.metadata.get('page', 'Unknown')
                    st.caption(f"- {source_name} (Page {page_num})")
                    
    # Save assistant message to UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Update LangChain's internal memory format 
    st.session_state.langchain_history.extend([
        ("human", user_query),
        ("assistant", answer)
    ])