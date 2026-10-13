import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import joblib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError
except ImportError:
    wikipedia = None
    DisambiguationError = Exception
    PageError = Exception
    print("⚠️ Wikipedia package not installed. Wikipedia features will be disabled.")

class HistoryRAGSystem:
    def __init__(self, model_name="llama3.2:1b"):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.vectorstore_dir = os.path.join(self.project_dir, "vectorstore")
        self.data_dir = os.path.join(self.project_dir, "data")
        # Ensure required directories exist
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_name = model_name
        self.retriever_k = 5
        
        # Load the custom-trained classifier
        self.classifier_path = os.path.join(self.project_dir, 'history_classifier.joblib')
        self.classifier = self._load_classifier()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        self.prompt = PromptTemplate(
            template="""You are a History Expert. Write a LONG and DETAILED answer using ONLY the context provided. 
            Include all names, dates, and causes mentioned in the text.
            
            STRUCTURE:
            1. Overview
            2. Detailed Facts (Use bullet points)
            3. Historical Significance

            If you don't know the answer based on the context, say "I don't know."

            CONTEXT: {context}
            HISTORY: {chat_history}
            QUESTION: {question}
            ANSWER:""",
            input_variables=["context", "chat_history", "question"],
        )
        
        self.vectorstore = self._load_vectorstore()
        self._check_ollama()
        self.chain = self._create_chain()

    def _check_ollama(self):
        """Verify if Ollama is running and responsive."""
        import requests
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            if response.status_code != 200:
                raise Exception
        except:
            raise RuntimeError("❌ Ollama is NOT running! Please start the Ollama application on your computer and try again.")

    def _load_classifier(self):
        if os.path.exists(self.classifier_path):
            try:
                return joblib.load(self.classifier_path)
            except:
                return None
        return None

    def _load_vectorstore(self):
        return FAISS.load_local(self.vectorstore_dir, self.embeddings, allow_dangerous_deserialization=True)

    def _create_chain(self):
        llm = ChatOllama(model=self.model_name, temperature=0.1, num_predict=1000)
        memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True, output_key="answer")
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def stream_query(self, question: str):
        print(f"INFO: [STREAM] Querying AI for: '{question}'")
        
        # 🧠 Predict the Era (Fast, no streaming needed)
        detected_era = "General History"
        if self.classifier:
            try:
                detected_era = self.classifier.predict([question])[0]
            except: pass

        # 🔍 Step 1: Retrieval (Non-streaming)
        # We find the documents first so we can yield them to the UI
        docs = self.vectorstore.similarity_search(question, k=self.retriever_k)
        sources = list(set([os.path.basename(doc.metadata.get("source", "")) for doc in docs]))
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Initial Payload
        yield {"type": "metadata", "era": detected_era, "sources": sources}

        # 🗣️ Step 2: Stream Generation
        llm = ChatOllama(model=self.model_name, temperature=0.1, num_predict=1000)
        
        full_answer = ""
        # Create a simple prompt for the LLM
        prompt_text = self.prompt.format(context=context, chat_history="", question=question)
        
        for chunk in llm.stream(prompt_text):
            chunk_content = chunk.content
            full_answer += chunk_content
            yield {"type": "content", "delta": chunk_content}

        # 📚 Step 3: Check for ignorance and Wikipedia fallback with robust handling
        ignorance_keywords = ["don't know", "dont know", "no information", "insufficient", "not mentioned", "not found"]
        is_ignorant = any(k in full_answer.lower() for k in ignorance_keywords) or len(full_answer) < 60

        if is_ignorant:
            # Attempt Wikipedia search
            try:
                search_results = wikipedia.search(question)
                if not search_results:
                    print("INFO: No Wikipedia results found for the query.")
                else:
                    # Choose the most appropriate result, skip generic entries like 'Batman' unless specific
                    wiki_topic = search_results[0]
                    if "batman" in wiki_topic.lower() and len(search_results) > 1:
                        wiki_topic = search_results[1]

                    # Sanitize topic name for filename usage
                    sanitized_topic = "".join(ch for ch in wiki_topic if ch.isalnum() or ch in " -_").strip()
                    filename = f"learned_{sanitized_topic.replace(' ', '_').lower()}.txt"
                    filepath = os.path.join(self.data_dir, filename)

                    # Avoid redundant learning
                    if os.path.exists(filepath):
                        print(f"INFO: Topic '{wiki_topic}' already exists locally. Skipping fetch.")
                    else:
                        # Try fetching the page, handling possible Wikipedia errors gracefully
                        try:
                            page = wikipedia.page(wiki_topic, auto_suggest=False)
                        except (DisambiguationError, PageError) as page_err:
                            # If the first attempt fails, attempt the next search result if available
                            if len(search_results) > 1:
                                fallback_topic = search_results[1]
                                print(f"WARN: Falling back to alternative Wikipedia result '{fallback_topic}' due to error: {page_err}")
                                page = wikipedia.page(fallback_topic, auto_suggest=False)
                                wiki_topic = fallback_topic
                            else:
                                raise page_err

                        summary = getattr(page, "summary", page.content[:500])
                        content_to_save = page.content[:3000]

                        # Save the fetched knowledge locally
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(f"LEARNED KNOWLEDGE: {wiki_topic}\n\n{content_to_save}")

                        # Update in‑memory FAISS index instantly
                        from langchain_core.documents import Document
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(content_to_save)
                        new_docs = [Document(page_content=c, metadata={"source": filepath}) for c in chunks]
                        self.vectorstore.add_documents(new_docs)
                        self.vectorstore.save_local(self.vectorstore_dir)
                        print(f"DONE: Successfully learned and indexed topic: '{wiki_topic}'")
                        yield {"type": "wiki", "topic": wiki_topic, "summary": summary}
            except Exception as e:
                print(f"ERROR: Failed to learn from Wikipedia: {e}")
                # Continue without Wikipedia fallback
                pass

    def query(self, question: str):
        generator = self.stream_query(question)
        answer = ""
        sources = []
        era = "General History"
        fetched_new = None
        
        for chunk in generator:
            if chunk["type"] == "metadata":
                sources = chunk["sources"]
                era = chunk["era"]
            elif chunk["type"] == "content":
                answer += chunk["delta"]
            elif chunk["type"] == "wiki":
                fetched_new = chunk["topic"]
                # Append the learning note to the end as requested
                answer += f"\n\n---\n💡 **Learned from Wikipedia**: I added '{fetched_new}' to my local knowledge base for future research."
                sources.append(f"Wikipedia (New Study: {fetched_new})")
                
        return {"answer": answer, "sources": sources, "detected_era": era, "fetched_new": fetched_new}
