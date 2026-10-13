import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_system import HistoryRAGSystem
import uvicorn

app = FastAPI(title="Chronos History AI Backend")

# Initialize the AI System
print("INFO: Loading History AI System...")
history_bot = HistoryRAGSystem()
print(f"DONE: AI System Ready using {history_bot.model_name}!")

from typing import List, Optional

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    detected_era: Optional[str] = None
    fetched_new: Optional[str] = None

@app.get("/")
def home():
    return {"status": "Chronos API is running", "model": history_bot.model_name}

from fastapi.responses import StreamingResponse
import json

@app.post("/stream")
async def stream_predict(request: QueryRequest):
    def generate():
        for chunk in history_bot.stream_query(request.message):
            yield json.dumps(chunk) + "\n"
            
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/predict", response_model=QueryResponse)
def predict(request: QueryRequest):
    """Fulfills the /predict (or equivalent) API requirement."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty query provided")
    
    try:
        result = history_bot.query(request.message)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            detected_era=result.get("detected_era"),
            fetched_new=result.get("fetched_new")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
