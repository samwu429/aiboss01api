import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import uvicorn

# 1. 配置
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def read_root(): return {"status": "Live", "message": "AI Boss Universal Engine Ready"}

# --- 路由 1: HR 简历审核 & 兼职拦截 ---
@app.post("/screen_resume_v2")
async def screen_resume_v2(job_description: str = Form(...), resume_file: UploadFile = File(...)):
    try:
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        resume_text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])
        prompt = f"""ROLE: Elite HR Director. EVALUATE: Job: {job_description} vs Resume: {resume_text}. 
        INSTRUCTIONS: If this is a gig check, you MUST start with [PASS] or [FAIL]. 
        Then provide Match Score, Strengths, Weaknesses, and 3 tough interview questions."""
        response = model.generate_content(prompt)
        return {"review_result": response.text}
    except Exception as e: return {"review_result": f"[FAIL] Error: {str(e)}"}

# --- 路由 2: 智能记账 (双重摘要 + 分类 + 金额) ---
class LedgerRequest(BaseModel):
    text: str

@app.post("/ledger_ai")
async def ledger_ai(req: LedgerRequest):
    try:
        prompt = f"""Act as a Top Accountant. Analyze: "{req.text}". 
        Output Format: Category|Short_Summary|Detailed_Explanation|Amount_Number
        Short_Summary: max 8 words. Detailed: 2 professional sentences. Amount: number only."""
        res = model.generate_content(prompt)
        return {"data": res.text}
    except Exception: return {"data": "Misc|Error|Processing failed|0"}

# --- 路由 3: 财报对话 (QA 模式) ---
class FinanceChatRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_chat")
async def finance_chat(req: FinanceChatRequest):
    try:
        prompt = f"Role: Wall Street CFO. Context: {req.report_type} report. Data: {req.history}. Task: If data is missing for a 1500-word report, ask ONE sharp question with [PROGRESS: X/Y]. If enough, reply ONLY [READY]."
        res = model.generate_content(prompt)
        return {"reply": res.text}
    except Exception as e: return {"reply": f"Error: {str(e)}"}

# --- 路由 4: 长篇专业财报生成 ---
@app.post("/finance_report")
async def finance_report(req: FinanceChatRequest):
    try:
        prompt = f"Role: Senior Auditor. Based on: {req.history}, generate an exhaustive 1500-word {req.report_type} Financial Report. Use US GAAP standards, EBITDA, FCF, CapEx analysis. Format with ALL-CAPS headers. No markdown code blocks."
        res = model.generate_content(prompt, generation_config={"max_output_tokens": 4000})
        return {"report": res.text}
    except Exception as e: return {"report": f"Gen Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
