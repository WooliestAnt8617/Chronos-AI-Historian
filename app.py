"""
app.py — Gradio chatbot interface for the Historical Fact Bot.

Provides a beautiful, themed chat UI with source document display,
example questions, and conversation history.
"""

import os
import re
import gradio as gr
import requests
import json


# --- Configuration -------------------------------------------
API_URL = "http://127.0.0.1:8000"


# --- Chat Function (Client) -----------------------------------
def chat(message: str, history: list) -> str:
    """Process a chat message by sending it to the FastAPI backend."""
    message = message.strip()
    
    # Input Validation: Reject empty or purely punctuation inputs
    if not message or not re.search('[a-zA-Z]', message):
        return "INFO: Please ask a valid historical question using words."
        
    # Input Validation: Reject obvious keyboard smashes/gibberish
    if any(len(word) > 20 for word in message.split()):
        yield "INFO: That doesn't look like a valid word. Could you double-check your spelling and try again?"
        return
        
    # Remove excessive punctuation that might confuse the model
    clean_message = re.sub(r'([!?,.\'"])\1+', r'\1', message)

    try:
        # Connect to our FastAPI streaming endpoint
        response = requests.post(
            f"{API_URL}/stream", 
            json={"message": clean_message}, # Changed 'question' to 'message' to match existing backend expectation
            stream=True,
            timeout=60
        )
        
        full_answer = ""
        era = "General History"
        sources = []
        
        # Check for non-streaming errors first
        if response.status_code != 200:
            yield f"ERROR: Backend returned status {response.status_code}: {response.text}\n\nPlease ensure 'server.py' is running and accessible."
            return

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    
                    if chunk["type"] == "metadata":
                        era = chunk["era"]
                        sources = chunk["sources"]
                    elif chunk["type"] == "content":
                        full_answer += chunk["delta"]
                        # Prepend Era and current answer for real-time update
                        display_text = f"📜 **Era**: {era}\n\n{full_answer}"
                        yield display_text
                    elif chunk["type"] == "wiki":
                        # Append the study note to the answer instead of replacing it
                        topic = chunk["topic"]
                        summary = chunk["summary"]
                        # If the answer was just "I don't know", replace it with the summary
                        if "know" in full_answer.lower() and len(full_answer) < 150:
                            full_answer = summary
                        
                        full_answer += f"\n\n---\n💡 **Learned from Wikipedia**: I just studied '{topic}' and added it to my local records for future research."
                        yield f"📜 **Era**: {era}\n\n{full_answer}"
                except json.JSONDecodeError:
                    # Handle cases where a line might not be valid JSON (e.g., server error messages)
                    yield f"ERROR: Failed to decode JSON from stream: {line.decode('utf-8')}"
                    return

        # Final update with sources after all chunks are processed
        if sources:
            # Clean up source names: 'learned_topic.txt' -> 'Local Knowledge (Topic)'
            clean_sources = []
            for s in sources:
                if s.startswith("learned_"):
                    name = s.replace("learned_", "").replace(".txt", "").replace("_", " ").title()
                    clean_sources.append(f"Local Records ({name})")
                else:
                    clean_sources.append(s)
            
            final_output = f"📜 **Era**: {era}\n\n{full_answer}\n\n---\n**Sources**: {', '.join(sorted(list(set(clean_sources))))}"
            yield final_output
        elif not full_answer: # If no content was streamed, but no error, provide a default message
            yield "INFO: I didn't get a response from the AI. Please try again."
            
    except requests.exceptions.ConnectionError:
        yield "ERROR: Could not connect to the AI Backend. Please ensure 'server.py' is running and accessible at " + API_URL
    except Exception as e:
        yield f"ERROR: I encountered an unexpected error while looking into that: {str(e)}\n\nPlease try rephrasing your question or check your connection."


# ─── Custom CSS ──────────────────────────────────────────────
custom_css = """
/* Global */
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
    font-family: 'Inter', sans-serif;
}

/* Header */
.header-section {
    text-align: center;
    padding: 30px 0 20px 0;
}
.header-section h1 {
    font-size: 2.5em;
    color: #2c3e50;
    margin-bottom: 5px;
    font-weight: 700;
}
.header-section p {
    color: #7f8c8d;
    font-size: 1.1em;
    margin-top: 0;
}

/* Footer */
.footer-text {
    text-align:center;
    padding:15px;
    color:#95a5a6;
    font-size:0.85em;
    border-top: 1px solid #ecf0f1;
    margin-top:20px;
}
"""

# ─── Example Questions ───────────────────────────────────────
example_questions = [
    "What caused the fall of the Roman Empire?",
    "Tell me about the Industrial Revolution and its impact",
    "Who was Emperor Ashoka and what were his achievements?",
    "What were the main causes of World War I?",
    "Describe the Indus Valley Civilization",
    "What happened during the Crusades?",
]


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(
    title="Chronos: The AI Historian",
) as app:

    # Header
    gr.HTML("""
        <div class="header-section">
            <h1>🏛️ Chronos</h1>
            <p>Your Infinite AI Historian — powered strictly by local data</p>
        </div>
    """)

    # Chat interface
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=example_questions,
        cache_examples=False,
        chatbot=gr.Chatbot(
            height=550,
            placeholder="<div style='text-align:center; color:#7f8c8d; padding:40px;'>"
                        "<h3>Welcome to Chronos!</h3>"
                        "<p>I am a locally run historian. Ask me about ancient civilizations, <br>"
                        "world wars, historical eras, or any historical figures.</p>"
                        "<p><i>(If I don't know the answer, I will dynamically learn it from Wikipedia!)</i></p>"
                        "</div>",
        ),
        textbox=gr.Textbox(
            placeholder="Ask Chronos a historical question... (e.g., 'What caused the French Revolution?')",
            container=False,
            scale=7,
        ),
        submit_btn="Ask Chronos",
    )

    # Footer
    gr.HTML("""
        <div class="footer-text">
            <p>🧠 Powered by Ollama (Llama 3.2:1b) · FAISS (K=5) · LangChain · 100% Local Inference & Auto-Learning</p>
        </div>
    """)


# ─── Launch ──────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css=custom_css,
    )
