from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# load the data , loads all pages
loader = PyPDFDirectoryLoader("bike_data")
documents = loader.load()
print(f"Success! Loaded {len(documents)} pages from the bike_data folder.") # returns total number of pages(i.e, from all documents)



# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=50,     
    separators=["\n\n", "\n", " ", ""] # priority for splitting : Paragraph first
)


chunks = text_splitter.split_documents(documents)
print(f"Success! Split the manuals into {len(chunks)} searchable chunks.") # validation test

"""
example validation  
if chunks:
    print("\n--- Example Chunk ---")
    print(chunks[0].page_content)
    print("\n--- Chunk Metadata (Source Info) ---")
    print(chunks[0].metadata)
"""



# create a vector database
# initialised free model
print("\nDownloading embedding model and converting text to vectors...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store the chunks in a local Chroma database
print("Saving vectors to ChromaDB...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db" # Though This folder will remain hidden
)