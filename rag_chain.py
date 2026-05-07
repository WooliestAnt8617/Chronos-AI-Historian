"""
rag_chain.py — RAG chain with factual QA prompt and conversation memory.

Uses Ollama (local LLM) + FAISS retriever + conversation memory
to answer historical questions based on the knowledge base.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


# ─── Configuration ───────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(PROJECT_DIR, "vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:0.5b"
RETRIEVER_K = 5  # Increased to provide more context for accuracy


# ─── Factual QA Prompt ───────────────────────────────────────
SYSTEM_TEMPLATE = """You are a highly accurate and concise history expert. Your task is to answer the user's question using ONLY the provided context below.

STRICT GUIDELINES:
1. Identify the specific historical event or figure in the user's question.
2. Search the CONTEXT for that specific event. 
3. If the context contains multiple different events (e.g., different wars or eras), ONLY use the information directly related to the question.
4. If the context does not contain the answer, say: "I don't have enough information about that specific topic in my local knowledge base."
5. Do NOT combine unrelated facts from different parts of the context.
6. Provide dates and names exactly as they appear in the context.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER (ensure accuracy based on context):"""

QA_PROMPT = PromptTemplate(
    template=SYSTEM_TEMPLATE,
    input_variables=["context", "chat_history", "question"],
)


def load_vectorstore():
    """Load the persisted FAISS vector store."""
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'. "
            "Run 'python build_index.py' first to build the index."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"DONE: Loaded FAISS vector store from '{VECTORSTORE_DIR}'")
    return vectorstore


def create_rag_chain():
    """Create the conversational RAG chain."""

    # Load vector store and create retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )

    # Initialize local LLM via Ollama
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,  # Low temperature for factual accuracy
        num_predict=1024,  # Max tokens in response
    )
    print(f"DONE: Initialized Ollama LLM (model: {OLLAMA_MODEL})")

    # Conversation memory — keeps last 2 exchanges to save context space
    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Build the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    print("DONE: RAG chain created successfully!")
    return chain


def ask(chain, question: str) -> dict:
    """Ask a question and get the answer with source documents. Auto-fetches from Wikipedia if unknown."""
    result = chain.invoke({"question": question})
    answer = result["answer"]

    # Check if the LLM says it doesn't know
    if "I don't have enough information" in answer or "I do not have enough information" in answer:
        print(f"INFO: Knowledge missing for: '{question}'")
        print("INFO: Attempting to fetch dynamically from Wikipedia...")
        try:
            import wikipedia
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            
            # Search wikipedia for the question
            search_results = wikipedia.search(question)
            if search_results:
                best_match = search_results[0]
                # FETCH ONLY THE SUMMARY to prevent massive CPU embedding delays
                summary_content = wikipedia.summary(best_match, auto_suggest=False)
                
                # Save to data folder
                filename = os.path.join(PROJECT_DIR, "data", f"{best_match.lower().replace(' ', '_')}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(summary_content)
                    
                # Chunk and add to vectorstore
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(summary_content)
                docs = [Document(page_content=c, metadata={"source": filename}) for c in chunks]
                
                vectorstore = chain.retriever.vectorstore
                vectorstore.add_documents(docs)
                vectorstore.save_local(VECTORSTORE_DIR)
                
                print(f"DONE: Successfully learned about '{best_match}'. Retrying question...")
                
                # Clean up the failed memory interaction so it doesn't pollute context
                if hasattr(chain, 'memory') and hasattr(chain.memory, 'chat_memory'):
                    if len(chain.memory.chat_memory.messages) >= 2:
                        chain.memory.chat_memory.messages = chain.memory.chat_memory.messages[:-2]
                
                # Ask again!
                result = chain.invoke({"question": question})
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "fetched_new": best_match
                }
        except Exception as e:
            print(f"ERROR: Failed to fetch dynamically: {e}")
            pass # Return original answer if fetching fails

    return {
        "answer": answer,
        "source_documents": result.get("source_documents", []),
    }


if __name__ == "__main__":
    print("🤖 Historical Fact Bot — CLI Mode")
    print("Type 'quit' to exit.\n")

    chain = create_rag_chain()

    while True:
        question = input("\n📜 Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue

        result = ask(chain, question)
        print(f"\n📖 Answer:\n{result['answer']}")

        if result["source_documents"]:
            print(f"\n📚 Sources ({len(result['source_documents'])} documents):")
            for i, doc in enumerate(result["source_documents"], 1):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                print(f"   {i}. {source}: {doc.page_content[:100]}...")
