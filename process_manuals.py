from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the data , loads all pages
loader = PyPDFDirectoryLoader("bike_data")
documents = loader.load()
# returns total number of pages from all document
print(f"Success! Loaded {len(documents)} pages from the bike_data folder.")



# split the pages into chunks of 500 char with 50 char overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=50,     
    separators=["\n\n", "\n", " ", ""] # priority for splitting : Paragraph first
)


chunks = text_splitter.split_documents(documents)
print(f"Success! Split the manuals into {len(chunks)} searchable chunks.")


# example 
# if chunks:
#     print("\n--- Example Chunk ---")
#     print(chunks[0].page_content)
#     print("\n--- Chunk Metadata (Source Info) ---")
#     print(chunks[0].metadata)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# create a vector database
print("\nDownloading embedding model and converting text to vectors...")
# initialised free model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Store the chunks in a local Chroma database
print("Saving vectors to ChromaDB...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db" # Creates a folder in your current directory
)

print(f"Success! Database created in the 'chroma_db' folder.")