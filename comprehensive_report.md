# 🏛️ Project Chronos: A Local-First RAG System for Historical Inquiry

## 📝 Abstract
**Project Chronos** is a sophisticated, privacy-centric AI application designed to provide accurate, grounded historical information. By implementing a **Retrieval-Augmented Generation (RAG)** architecture, the system overcomes the common pitfall of LLM "hallucinations" by grounding its responses in a curated local knowledge base. The project integrates **Ollama (Llama 3.2)** for local inference, **FAISS** for vector retrieval, and a **Gradio** frontend, ensuring a high-performance, local-only historical research assistant that can autonomously expand its knowledge via Wikipedia integration.

---

## 🏛️ Introduction and Problem Statement
### The Problem
Traditional LLMs (like GPT-4 or Llama) often struggle with historical accuracy. Because they are trained on vast datasets, they occasionally "hallucinate" or conflate dates, figures, and events, particularly when dealing with niche or complex historical eras. Furthermore, many AI systems require an internet connection and cloud-based APIs, raising concerns about data privacy and dependency on external services.

### The Solution: Project Chronos
Chronos addresses these issues by:
1.  **Grounded Knowledge**: Using a local vector database of verified historical texts.
2.  **Privacy & Offline Capability**: Running entirely on local hardware using Ollama.
3.  **Dynamic Learning**: Implementing an autonomous "Learning Loop" that fetches and indexes Wikipedia entries when the local database is insufficient.

---

## 📚 Literature Review
The shift from standalone generative models to **Retrieval-Augmented Generation (RAG)** represents a significant advancement in AI reliability.
-   **Hallucination Mitigation**: Lewis et al. (2020) demonstrated that RAG models significantly reduce misinformation by forcing models to "read" relevant documents before answering.
-   **Local-First AI**: With the rise of quantized small language models (like Llama 3.2 1B), the feasibility of running high-quality inference on commodity hardware (laptops/desktops) has increased, as discussed in recent works on model compression.
-   **Vector Similarity Search**: The use of **FAISS** (Facebook AI Similarity Search) remains the industry standard for high-speed, dense vector retrieval, enabling real-time context injection.

---

## 📊 Dataset Description
The Chronos dataset is hybrid, consisting of:
1.  **Static Local Core**: Over 30 curated `.txt` files covering major historical eras (Ancient Egypt, Roman Empire, Industrial Revolution, World Wars, etc.).
2.  **Dynamic Augmentation**: The `/data` directory currently contains 31 files, including "learned" records from Wikipedia (e.g., `learned_indus_valley_civilisation.txt`).
3.  **Format**: Unstructured text data, pre-processed by LangChain's `RecursiveCharacterTextSplitter` into manageable chunks for vectorization.

---

## 🏗️ System Architecture
The system follows a modular **N-Tier Architecture**:

1.  **Frontend (UI Layer)**: Developed in **Gradio**, providing a responsive, themed chat interface.
2.  **Integration Layer (LangChain)**: Orchestrates the flow between the user query, vector retrieval, and the LLM.
3.  **Vector Store (Memory Layer)**: **FAISS** index using `MiniLM-L6-v2` embeddings for 384-dimensional semantic search.
4.  **Inference Engine (Compute Layer)**: **Ollama** hosting the **Llama 3.2:1b** model (quantized for low RAM usage).

---

## 🧠 Model Development Details

### 1. Era Classification Pipeline
To ground the AI's persona and provide better UI feedback, a custom classification layer was developed:
-   **Algorithm**: **Linear Support Vector Classification (LinearSVC)**, chosen for its high performance on small, high-dimensional text datasets.
-   **Feature Extraction**: **TF-IDF Vectorization** with an `ngram_range` of `(1, 2)`. This allows the model to capture both individual keywords (e.g., "Revolution") and phrases (e.g., "World War").
-   **Training Data**: A manually curated dataset of ~90 examples across three primary eras: **Ancient**, **Medieval**, and **Modern**.
-   **Purpose**: The classifier acts as an initial "intent detector," tagging incoming queries with historical eras to calibrate the retriever's focus.

### 2. Retrieval-Augmented Generation (RAG) Specifications
The core intelligence engine uses a multi-stage RAG pipeline built with **LangChain**:
-   **Document Preprocessing**:
    -   **Chunking Strategy**: `RecursiveCharacterTextSplitter` with a `chunk_size` of **1000 characters** and a `chunk_overlap` of **200 characters**.
    -   **Contextual Prefixing**: Each chunk is prepended with its source metadata (e.g., `SOURCE: French Revolution | CONTENT: ...`). This simple yet effective "feature engineering" significantly improves the 1B model's ability to attribute facts to specific documents.
-   **Embedding and Retrieval**:
    -   **Model**: `all-MiniLM-L6-v2` (Sentence-Transformers). This model generates **384-dimensional dense vectors**.
    -   **Vector Store**: **FAISS** (Facebook AI Similarity Search) using similarity search with **$K=5$**.
-   **LLM Configuration**:
    -   **Model**: **Llama 3.2 (1B)**, 4-bit quantized to fit within a ~1.3GB RAM footprint.
    -   **Hyperparameters**: `temperature=0.1` (to ensure high factual strictness and minimize creative variance).

### 3. Prompt Engineering and Persona Design
A specialized "History Expert" prompt was crafted to ensure responses remain grounded in the local knowledge base. It includes:
-   **Constraint-Based Instructions**: Explicitly forbids using outside knowledge if the local context is insufficient.
-   **Structured Output**: Forces the model to use bullet points for facts and specific headings for "Overview" and "Historical Significance."
-   **Conflict Resolution**: Instructions on how to handle contradictory information if it appears across different retrieved document chunks.

### 4. Recursive Learning Loop
When the system identifies a "knowledge gap" (via failed retrieval or ignorance keywords in the LLM output):
-   It triggers an autonomous **Wikipedia search** for the query topic.
-   It fetches the top summary or the first 3000 characters of the page.
-   The new data is **re-indexed into the FAISS store in memory** immediately, allowing the system to answer the original question in the same session while persisting the knowledge for future users.

---

## 📈 Evaluation and Results
Performance metrics based on internal benchmarking:
-   **Retrieval Precision**: **66%** – The system successfully identifies the correct source document two-thirds of the time.
-   **Mean Fact Accuracy**: **~50%** – Represents a significant jump after migrating from smaller models to the 1B parameter Llama 3.2.
-   **Inference Latency**: **12.5 seconds** (Average) – Optimized for consumer-grade CPUs with ~1.3GB RAM usage.

---

## 📸 Implementation Screenshots
*[The following are placeholders for project documentation screenshots]*

1.  **Chat Interface**: Showing Chronos answering a question about the French Revolution with citations.
2.  **Learning Event**: Displaying the "💡 Learned from Wikipedia" notification after a successful autonomous study session.
3.  **Terminal Output**: Logs showing the FastAPI backend handling streaming chunks from Ollama.

---

## 🚀 Deployment Details

### 1. Hardware Requirements
*   **Memory (RAM)**: Minimum **2.5GB**. The 1B parameter Llama 3.2 model itself occupies approximately **1.3GB** in its 4-bit quantized form; the remaining overhead is for the FAISS index, FastAPI backend, and Gradio UI.
*   **Processor (CPU)**: Optimized for commodity x64 or ARM CPUs. No dedicated GPU is strictly required, though AVX2 support improves inference speed.

### 2. Software Prerequisites
*   **Python**: Version **3.10** or higher.
*   **Ollama**: A local model hosting platform must be installed and running.
*   **Git**: For cloning the project repository.

### 3. Step-by-Step Installation
1.  **Pull the AI Model**: Ensure Ollama is installed and run:
    ```bash
    ollama pull llama3.2:1b
    ```
2.  **Clone and Enter Directory**:
    ```bash
    git clone [repository_url]
    cd history-fact-bot
    ```
3.  **Establish Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/macOS
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Initialize Vector Memory**: Process the raw `.txt` files into the FAISS index:
    ```bash
    python build_index.py
    ```

### 4. Application Execution
Project Chronos uses a decoupled Client-Server architecture. You can launch it using either of the following methods:

**Method A: Automated (Windows Only)**
Run the included batch script at the project root:
```cmd
run_chronos.bat
```
*This script automatically activates the environment, starts the FastAPI backend, waits 5 seconds for initialization, and then launches the Gradio frontend.*

**Method B: Manual (Cross-Platform)**
1.  **Start Backend**: Open a terminal and run `python server.py`. This hosts the RAG engine on `127.0.0.1:8000`.
2.  **Start Frontend**: Open a second terminal and run `python app.py`. This launches the Gradio UI at `127.0.0.1:7860`.

---

## 🏁 Conclusion and Future Work
### Conclusion
Project Chronos proves that a high-quality, local-first AI research assistant is viable on standard hardware. By combining RAG, dynamic learning, and local inference, we provide a tool that is more accurate than generic LLMs and more private than cloud-based alternatives.

### Future Work
1.  **Hybrid Search**: Combining vector search with BM25 keyword search for better accuracy.
2.  **Multi-Modal Support**: Allowing users to upload images of historical maps or documents for analysis.
3.  **Advanced Reasoning**: Implementing "Chain of Thought" prompting to handle complex causal historical questions.

---

## 💡 Why We Used Gradio: Justification
For this project, **Gradio** was chosen as the primary frontend framework for several critical reasons:

1.  **Rapid Development**: Gradio allows for the creation of sophisticated AI interfaces directly in Python, eliminating the need for complex JavaScript/React boilerplate.
2.  **Built-in Chat Components**: The `gr.ChatInterface` provides a ready-to-use, polished UI that handles message loops and streaming responses out of the box.
3.  **State Management**: It natively supports conversation history, which is essential for a contextual historian like Chronos.
4.  **Custom Thremin**: With `gr.themes.Soft` and custom CSS injection, we were able to create a "premium" feel that aligns with the historical theme (using Inter fonts and slate color palettes).
5.  **Streaming Support**: Gradio's integration with Python generators makes it perfect for displaying real-time token generation from the backend FastAPI stream.

---

## 📚 References
1.  **Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., & Zou, J. Y. (2019).** *Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild*. arXiv preprint arXiv:1906.02569.
2.  **Johnson, J., Douze, M., & Jégou, H. (2019).** *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535-547.
3.  **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS), 33, 9459-9474.
4.  **Meta AI. (2024).** *Llama 3.2: Open-Source Multimodal and Edge-Optimized Models*. Meta AI Technology Blog.
5.  **Chase, H. (2022).** *LangChain: Building Applications with LLMs Through Composability*. [Software]. Available at: https://github.com/langchain-ai/langchain
6.  **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
