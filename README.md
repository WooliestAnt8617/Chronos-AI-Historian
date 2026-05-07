# 🏛️ Chronos: The AI Historian
### An End-to-End Local AI System for Historical Research

Chronos is a professional-grade, locally-run AI system designed to provide accurate historical facts using **Retrieval-Augmented Generation (RAG)** and **Dynamic Machine Learning**. It meets all "Major Project" requirements for AI/ML design, development, and deployment.

---

## 🛰️ System Architecture Overview
The system is built using a **Client-Server Architecture**, separating the user interface from the heavy-lifting AI backend.

### 1. Frontend: Gradio UI (`app.py`)
- **Role**: The entry point for the user.
- **Function**: Captures historical questions, sends them as JSON requests to the API, and renders the AI's response with sources and "learning" notifications.
- **Theme**: Custom "Soft" theme with historical aesthetics (🏛️ icons).

### 2. Backend: FastAPI Service (`server.py`)
- **Role**: The "Brain" of the operation.
- **Function**: Hosts the `/predict` API endpoint. It manages the lifecycle of the AI model and ensures data schema consistency.
- **Port**: Runs on `http://127.0.0.1:8000`.

### 3. AI Core: Modular RAG System (`rag_system.py`)
- **Role**: Intelligent retrieval and generation.
- **Components**:
    - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Converts text into 384-dimensional mathematical vectors).
    - **Vector Store**: `FAISS` (Local database for ultra-fast similarity searches).
    - **LLM**: `Llama 3.2 (1B Parameters)` via **Ollama** (Generates human-like, detailed history lessons).
    - **RAG Chain**: LangChain `ConversationalRetrievalChain` with memory.

---

## 🔄 The Data Flow: Step-by-Step

1.  **User Inquiry**: You type a question (e.g., *"Why did the Crusades start?"*) in the UI.
2.  **API Request**: The Frontend sends a POST request with the query to the Backend.
3.  **Local Research**: 
    - The Backend uses the **Embedding Model** to turn your question into a "vector."
    - It searches the **FAISS Index** for the closest matching 5 paragraphs from our historical text files.
4.  **Inference**:
    - The retrieved paragraphs + your question + conversation history are sent to **Llama 3.2**.
    - The model acts as a "History Professor" to synthesize a detailed answer.
5.  **Dynamic Learning Fallback**:
    - If the local files lack the data, the system automatically triggers a **Wikipedia search**.
    - It fetches the latest content and **creates a new `.txt` file** in the `data/` folder.
6.  **Response**: The Backend returns a JSON object containing the `answer`, `sources`, and any `learned_topic` flag.
7.  **Final Display**: The UI shows you the detailed answer and marks exactly which files were used for the research.

---

## 🛠️ Technical Specifications
- **Model**: Llama-3.2-1B-Instruct (4-bit quantization).
- **Inference Engine**: Ollama (Windows/Local).
- **Preprocessing**: Custom cleaning pipeline in `data_loader.py` (normalization, non-ASCII stripping).
- **Chunking**: 1000 characters with 200 character overlap for high-context retrieval.

---

## 🚀 How to Run the Entire System
1.  **Start Ollama**: Make sure the Ollama app is running in your taskbar.
2.  **Run the Batch File**: Double-click `run_chronos.bat`.
3.  **Interact**: Open the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

