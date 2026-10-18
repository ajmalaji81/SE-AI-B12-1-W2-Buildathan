import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from rag_service import ask_question, build_knowledge_base, reset_knowledge_base, UPLOAD_FOLDER

app = FastAPI(title="TN AgriBOT API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def home():
    return {"message": "TN AgriBOT API is running."}

@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        dest_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
    return {"uploaded": saved_files}

@app.post("/build_index")
def trigger_build():
    result = build_knowledge_base()
    return result

@app.post("/reset_index")
def trigger_reset():
    return reset_knowledge_base()

@app.post("/ask", response_model=QueryResponse)
def query_schemes(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = ask_question(payload.question)
    return QueryResponse(question=payload.question, answer=result["answer"])