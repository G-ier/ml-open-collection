import seaborn as sns
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from custom_tokenizer import custom_4500_token_encoder

# Define the searching tool
def search_tool(query: str, vector_db: Chroma):

    # Get and downlaod relevant documents from the web

    pass

# Download the specified documents from the web
def download_relevant_documents(link: str):
    pass

def longer_context_encoder(text: str):
    
    embeddings = []
    for chunk in chunks:
        embedding = custom_4500_token_encoder(chunk.page_content)
        embeddings.append(embedding)

    return embeddings

def encode_chunks(chunks: list):

    # Define the embedding model# Define the embedding model
    embedding_model = SentenceTransformer("intfloat/e5-base-v2")

    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk.page_content)
        embeddings.append(embedding)

    return embeddings

# Define the RAG pipeline
# define client --> search --> chunk --> add to db
def rag_insertion(query: str, file_path: str, collection: chromadb.Collection, custom_token_encoder: bool = False):

    # Chunk the document for the database using a stream
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    if custom_token_encoder:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)

        # Encode each chunk and retrieve the embeddings list
        encoded_chunks = encode_chunks(chunks)

        # Add the chunks to the collection
        collection.add(encoded_chunks)
    else:
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Encode each chunk and retrieve the embeddings list
        encoded_chunks = encode_chunks(chunks)

        # Add the chunks to the collection
        collection.add(encoded_chunks)
        
# Retrieve the chunks from the collection???
def rag_retrieval(query: str, collection: chromadb.Collection):
    
    # Retrieve the chunks from the collection
    chunks = collection.query(query_texts=[query], n_results=15)
    print(chunks)

# Rerank shit separately???
def rerank():
    pass

def rag_pipeline(query: str, file_path: str):

    # Initialize the vector db
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="/Users/gier/projects/implementations/db/"
    ))

    # Create or get the collection
    collection = client.get_or_create_collection(name="my_collection")

    rag_insertion(query, file_path, collection)
    rag_retrieval(query, collection)