import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Load book data
books = pd.read_csv("P:/book_recommender-main/book_recommender-main/book_recommender/data/books_cleaned.csv")

# Function to save descriptions as a text file for embeddings
def save_descriptions_to_txt():
    books['tagged_description'].to_csv(r"P:/book_recommender-main/book_recommender-main/book_recommender/data/tagged_description.txt", sep='\n', index=False, header=False)

# Function to load and split documents
# def load_and_split_documents(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
#     # raw_documents = TextLoader(file_path, encoding="utf-8").load()
#     # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='\n')
#     # return text_splitter.split_documents(raw_documents)
#     with open(file_path, "r", encoding="utf-8") as f:
#         raw_text = f.read()

#     text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='\n')
#     return text_splitter.split_text(raw_text)

def load_and_split_documents(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='\n')
    chunks = text_splitter.split_text(raw_text)
    
    # Convert each chunk into a Document object
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

# Function to create Chroma vector database from documents
def create_chroma_db(documents: list) -> Chroma:
    return Chroma.from_documents(
        documents,
        embedding=OllamaEmbeddings(model='gemma2:2b'),
    )

# Function to retrieve semantic recommendations
def retrieve_semantic_recommend(query: str, db_books: Chroma, books: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    return books[books['isbn13'].isin(books_list)].head(top_k)

# Save descriptions to text file
save_descriptions_to_txt()

# Load and split documents into chunks
documents = load_and_split_documents("P:/book_recommender-main/book_recommender-main/book_recommender/data/tagged_description.txt")

# Create Chroma vector database from the documents
db_books = create_chroma_db(documents)

# Query for recommendations
query = "A book that talks about war"
df = retrieve_semantic_recommend(query, db_books, books)

# Display the results
print(df)
