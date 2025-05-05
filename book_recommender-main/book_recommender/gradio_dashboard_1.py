import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import gradio as gr
from langchain_ollama import OllamaEmbeddings
import os

load_dotenv()

# Load books and set thumbnails
# books = pd.read_csv("P:/book_recommender-main/book_recommender-main/book_recommender/data/books_with_emotions.csv")
books = pd.read_csv(r"P:\book_recommender-main\book_recommender-main\book_recommender\data\books_with_emotions.csv")
# books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = books["thumbnail"].fillna("book_recommender/data/OnePageBookCoverImage.jpg") + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), 
    "book_recommender/data/OnePageBookCoverImage.jpg", 
    books['large_thumbnail']
)

CHROMA_DB_PATH = "P:/book_recommender-main/book_recommender-main/book_recommender/chroma_db"

# def get_or_create_db():
#     if os.path.exists(CHROMA_DB_PATH):
#         try:
#             print("Loading existing Chroma database...")
#             return Chroma(
#                 persist_directory=CHROMA_DB_PATH,
#                 embedding_function= OllamaEmbeddings(model='gemma2:2b')
#             )
#         except Exception as e:
#             print(f"Error loading database: {e}")
    
#     print("Creating new Chroma database...")
#     raw_documents = TextLoader("P:/book_recommender-main/book_recommender-main/book_recommender/data/tagged_description.txt",encoding="utf-8").load()
#     print("Sample text:", raw_documents[:5])
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
#     documents = text_splitter.split_documents(raw_documents)
    
#     return Chroma.from_documents(
#         documents=documents,
#         embedding=OllamaEmbeddings(model='gemma2:2b'),
#         persist_directory=CHROMA_DB_PATH
#     )

# db_books = get_or_create_db()

import chardet  
import sys

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
import chardet
import sys

import sys
import chardet
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm  # For the progress bar

def get_or_create_db():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Creating a new Chroma database...")

    # File path to the tagged description text file
    file_path = "P:/book_recommender-main/book_recommender-main/book_recommender/data/tagged_description.txt"

    # Detect encoding of the file
    with open(file_path, "rb") as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)["encoding"]
        print(f"Detected file encoding: {detected_encoding}")

    # Read and clean the file line by line
    cleaned_lines = []
    with open(file_path, "r", encoding=detected_encoding, errors="replace") as f:
        for line in f:
            line = "".join(char if char.isprintable() else " " for char in line)
            if line.strip():  # skip empty lines
                cleaned_lines.append(line.strip())

    # Save the cleaned lines for reference (optional)
    cleaned_file = "cleaned_tagged_description.txt"
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    # Create one document per line
    raw_documents = [Document(page_content=line) for line in cleaned_lines]
    print(f"Total raw documents loaded: {len(raw_documents)}")

    # Optional: further split documents into smaller chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Total split documents: {len(documents)}")

    # Create Chroma vector store instance with OllamaEmbeddings
    embedding_model = OllamaEmbeddings(model="gemma2:2b")  # Initialize OllamaEmbeddings object
    persist_directory = "P:/book_recommender-main/book_recommender-main/book_recommender/chroma_db"
    
    # Update Chroma import and usage
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model.embed_query)

    # Batch processing with a progress bar
    batch_size = 100  # Adjust batch size based on memory constraints
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents", unit="batch"):
        chunk = documents[i:i + batch_size]
        
        try:
            # Add documents to Chroma database
            db.add_documents(chunk)
            
            # Persist after every batch to manage memory
            db.persist()
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            continue  # Continue processing even if an error occurs in a batch

    # Final persist to ensure all documents are saved
    db.persist()

    print(f"New Chroma database created with {db._collection.count()} documents.")
    return db

# Run the function and print the document count
db_books = get_or_create_db()
print("Chroma database document count:", db_books._collection.count())



def retrieve_book_recommendation(query: str, category: str = None, tone: str = None,
                                 initial_top_k=50, final_top_k=16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    print("recs",recs)
    book_list = [int(rec.page_content.strip('"\'').split()[0]) for rec in recs]
    matching_books = books[books['isbn13'].isin(book_list)]
    
    book_recs = matching_books.copy()
    
    if category and category != "All":
        book_recs = book_recs[book_recs['simple_categories'] == category]
    
    if tone and tone != "All":
        tone_mapping = {
            "Happy": "joy", "Surprising": "surprise",
            "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"
        }
        if tone in tone_mapping:
            book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)
    print(book_recs)
    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_book_recommendation(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row.get('description', 'No description available')
        truncated_desc = ' '.join(str(description).split()[:30]) + '...'
        
        authors = str(row.get('authors', 'Unknown Author')).split(';')
        if len(authors) >= 2:
            authors_str = f"{', '.join(authors[:-1])} and {authors[-1]}"
        else:
            authors_str = authors[0]
        
        caption = f"📖 {row['title']}\n✍️ {authors_str}\n📝 {truncated_desc}"
        results.append((row['large_thumbnail'], caption))
    print(results)
    return results

# UI Configuration
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

css = """
.gallery {
    gap: 20px !important;
}
.gallery-item {
    height: 400px !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #1e293b !important;
    border: 1px solid #334155 !important;
}
.gallery-item img {
    height: 300px !important;
    object-fit: cover !important;
}
.gallery-item .caption {
    padding: 12px !important;
    color: white !important;
    font-size: 0.9em !important;
    line-height: 1.4 !important;
}
"""

with gr.Blocks(css=css) as dashboard:
    gr.Markdown("# 📚 Discover Your Next Favorite Book")
    
    with gr.Row():
        user_query = gr.Textbox(
            label="Describe your reading mood",
            placeholder="e.g. 'A thrilling mystery with complex characters'",
            max_lines=3
        )
        
    with gr.Row():
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Filter by Category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Filter by Mood",
            value="All"
        )
        submit_btn = gr.Button("Find Books 🔍", variant="primary")
    
    gallery = gr.Gallery(
        label="Recommended Books",
        columns=5,
        rows=2,
        object_fit="cover",
        height="auto"
    )

    submit_btn.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=gallery
    )

if __name__ == "__main__":
    dashboard.launch(share=True)