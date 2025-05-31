from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from custom_tokenizer import custom_4500_token_encoder

# Define the searching tool
def search_tool(urls, vector_db: Chroma):

    # Get and downlaod relevant documents from the web
    loader = WebBaseLoader(web_paths=urls)
    documents = loader.load()
    
    return documents


def longer_context_encoder(chunks: list):
    
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
        encoded_chunks = longer_context_encoder(chunks)

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
        
# Retrieve the chunks from the collection with MMR ranker
def rag_retrieval(query: str, collection: chromadb.Collection, custom_token_encoder: bool = False):
    
    # Define the result list size based on the custom token encoder
    if custom_token_encoder:
        elements_to_return = 6
        elements_to_search = 10
    else:
        elements_to_return = 10
        elements_to_search = 20

    # Wrap the chromadb into a chroma vectorstore
    vectordb = Chroma(collection_name=collection.name, embedding_function=custom_4500_token_encoder, persist_directory="/Users/gier/projects/implementations/db/")

    # Use MMR search directly on the vectorstore
    results = vectordb.max_marginal_relevance_search(query, k=elements_to_return, fetch_k=elements_to_search, lambda_mult=0.5)

    # Return the reranked chunks
    return results
    

# Define the RAG stateful pipeline
def rag_pipeline(query: str, file_path: str):

    # Initialize the vector db
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="/Users/gier/projects/implementations/db/"
    ))

    # Create or get the collection
    collection = client.get_or_create_collection(name="qa_rag_database")

    # Define the custom token encoder
    custom_token_encoder = True

    # Insert the document into the collection
    rag_insertion(query, file_path, collection, custom_token_encoder=custom_token_encoder)

    # Retrieve the chunks from the collection
    rag_results = rag_retrieval(query, collection, custom_token_encoder=custom_token_encoder)

    # Pass to LLM

if __name__ == "__main__":
    
    search_tool(urls=[
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://docs.python.org/3/tutorial/introduction.html"
    ], vector_db=None)