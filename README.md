Chronos: The AI Historian
An End-to-End Local AI System for Historical Research
Chronos is a professional-grade, locally-run AI system designed to provide accurate historical facts using Retrieval-Augmented Generation (RAG) and Dynamic Machine Learning. This project meets the requirements for AI/ML design, development, and deployment within a local environment.

System Architecture Overview
The system is built using a Client-Server Architecture, separating the user interface from the heavy-lifting AI backend to ensure stability and scalability.

1. Frontend: Gradio UI (app.py)
Role: The user interface entry point.

Function: Captures historical questions, sends JSON requests to the API, and renders the AI's response including sources and learning notifications.

Theme: Custom "Soft" theme with a historical aesthetic.

2. Backend: FastAPI Service (server.py)
Role: The system orchestration layer.

Function: Hosts the /predict API endpoint, manages the AI model lifecycle, and ensures data schema consistency.

Port: Runs on http://127.0.0.1:8000.

3. AI Core: Modular RAG System (rag_system.py)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional mathematical vectors).

Vector Store: FAISS (Local database for fast similarity searches).

LLM: Llama 3.2 (1B Parameters) via Ollama.

RAG Chain: LangChain ConversationalRetrievalChain with memory.

The Data Flow: Step-by-Step
User Inquiry: The user enters a question in the UI.

API Request: The Frontend sends a POST request with the query to the FastAPI Backend.

Local Research: The Backend uses the Embedding Model to vectorize the question and searches the FAISS Index for the 5 most relevant document chunks.

Inference: The retrieved paragraphs, question, and conversation history are sent to Llama 3.2 for synthesis.

Dynamic Learning Fallback: If local files lack sufficient data, the system triggers a Wikipedia search, fetches content, and creates a new .txt file in the data/ folder for future use.

Response: The Backend returns a JSON object containing the answer, sources, and learning flags.

Final Display: The UI displays the detailed answer and cites the specific files used.

Technical Specifications
Model: Llama-3.2-1B-Instruct (4-bit quantization).

Inference Engine: Ollama.

Preprocessing: Custom cleaning pipeline (normalization, non-ASCII stripping).

Chunking: 1000 characters with 200 character overlap.

Setup and Installation
1. Prerequisites
Python 3.10 or higher.

Ollama: Installed and running.

Llama 3.2 Model: Run 'ollama run llama3.2:1b' in your terminal.

2. Install Dependencies
Install the required libraries using pip:

Bash
pip install fastapi uvicorn gradio langchain langchain-community sentence-transformers faiss-cpu ollama requests wikipedia
3. Create the Launcher (run_chronos.bat)
Since the batch file is not provided in the repository, create a new file named run_chronos.bat in the project root directory and paste the following:

Code snippet
@echo off
echo Starting Chronos: The AI Historian...

:: Start the FastAPI Backend
start /min cmd /c "python server.py"
echo [1/2] Backend Service Starting on http://127.0.0.1:8000...

:: Wait for initialization
timeout /t 5 /nobreak > NUL

:: Start the Gradio Frontend
echo [2/2] Launching Gradio UI...
python app.py

pause
How to Run
Ensure the Ollama application is active.

Double-click run_chronos.bat.

Access the interface via the local URL provided by Gradio (typically http://127.0.0.1:7860).
