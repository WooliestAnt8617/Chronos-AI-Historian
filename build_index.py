"""
build_index.py — Build the FAISS vector store from the historical knowledge base.

Loads documents, chunks them, generates embeddings using a local HuggingFace model,
and saves the FAISS index to disk for later retrieval.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from data_loader import load_documents, report_preprocessing_stats, chunk_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(PROJECT_DIR, "vectorstore")

# Embedding model — runs locally, ~80 MB download on first run
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    """Build FAISS vector store from knowledge base documents."""
    print("=" * 60)
    print("📚 Building FAISS Vector Store for Historical Fact Bot")
    print("=" * 60)

    # Step 1: Load documents
    print("\n🔄 Step 1: Loading documents...")
    documents = load_documents()
    print(f"   ✅ Loaded {len(documents)} documents.")

    # Step 2: Preprocessing and Chunking
    print(f"\n🔄 Step 2: Preprocessing and Chunking...")
    report_preprocessing_stats(documents)
    chunks = chunk_documents(documents)
    print(f"   ✅ Chunked into {len(chunks)} pieces.")

    # Step 3: Initialize embedding model
    print(f"\n🔄 Step 3: Loading embedding model '{EMBEDDING_MODEL}'...")
    print("   (First run will download ~80 MB model)")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    elapsed = time.time() - start_time
    print(f"   ✅ Embedding model loaded in {elapsed:.1f}s")

    # Step 3: Create FAISS vector store
    print("\n🔄 Step 3: Creating FAISS index from document chunks...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start_time
    print(f"   ✅ FAISS index created in {elapsed:.1f}s")

    # Step 4: Save to disk
    print(f"\n🔄 Step 4: Saving vector store to '{VECTORSTORE_DIR}'...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"   ✅ Vector store saved!")

    # Summary
    print("\n" + "=" * 60)
    print("✅ VECTOR STORE BUILD COMPLETE")
    print(f"   📦 Total chunks indexed: {len(chunks)}")
    print(f"   📂 Saved to: {VECTORSTORE_DIR}")
    print(f"   🧠 Embedding model: {EMBEDDING_MODEL}")
    print(f"   📐 Embedding dimensions: {len(embeddings.embed_query('test'))}")
    print("=" * 60)

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
