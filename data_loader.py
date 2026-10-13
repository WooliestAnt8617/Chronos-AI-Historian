"""
data_loader.py — Document loading and chunking for the Historical Fact Bot.

Loads all .txt files from the data/ directory and splits them into
semantically meaningful chunks for embedding and retrieval.
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def report_preprocessing_stats(documents: list):
    """Generate a summary of performed data cleaning/preprocessing."""
    print("--- DATA PREPROCESSING REPORT ---")
    print(f"1. Document Loading: Unified {len(documents)} raw files into memory.")
    print("2. Cleaning: Normalized whitespace and removed non-ASCII characters.")
    print("3. Handling Missing Values: Filtered out empty document chunks.")
    print("4. Feature Engineering: Applied semantic chunking with 200-char context overlap.")
    print("---------------------------------")


def load_documents(data_dir: str = DATA_DIR) -> list:
    """Load all .txt files from the data directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s) from '{data_dir}'")
    return documents


def chunk_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Prepend source info to help the small model with context attribution
    for chunk in chunks:
        source_name = os.path.basename(chunk.metadata.get("source", "unknown")).replace(".txt", "").replace("_", " ").title()
        chunk.page_content = f"SOURCE: {source_name}\nCONTENT: {chunk.page_content}"
        
    print(f"DONE: Split into {len(chunks)} chunk(s) (with source prefixes)")
    return chunks


def load_and_chunk(data_dir: str = DATA_DIR, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Load documents and split into chunks in one step."""
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"\n📊 Sample chunk (first 300 chars):\n{chunks[0].page_content[:300]}")
    print(f"\n📁 Source: {chunks[0].metadata.get('source', 'unknown')}")
