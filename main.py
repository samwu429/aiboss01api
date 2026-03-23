import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import uvicorn

# ==========================================
# 1. 核心配置与初始化
# ==========================================
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 【核心更新】使用 Gemini 2.5 Flash 模型，火力全开
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# 究极跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render 健康检查 (必须保留)
@app.get("/")
def read_root():
    return {"status": "Live", "engine": "Gemini 2.5 Flash", "version": "U-OS 3.0"}


# ==========================================
# 路由 1：AI 智能记账解析 (分类 + 双摘要)
# ==========================================
class LedgerRequest(BaseModel):
    text: str

@app.post("/ledger_ai")
async def ledger_ai(req: LedgerRequest):
    try:
        prompt = f"""
        Role: Senior Personal Wealth Accountant.
        Context: The user input is: "{req.text}"
        
        Task:
        1. Classify the spending (e.g., Food, Travel, Tech, Living, Investment, Entertainment).
        2. Short_Summary: A punchy 5-8 word summary for a list view.
        3. Detailed_Explanation: A professional 2-sentence financial narrative.
        4. Amount: Extract numbers only. Assume USD if not specified.
        
        OUTPUT FORMAT (Strictly Pipe-separated):
        Category|Short_Summary|Detailed_Explanation|Amount_Number
        """
        response = model.generate_content(prompt)
        return {"data": response.text}
    except Exception as e:
        return {"data": f"Misc|Processing Error|Failed to analyze: {str(e)}|0"}


# ==========================================
# 路由 2：AI 简历审核 & 兼职资格拦截
# ==========================================
@app.post("/screen_resume_v2")
async def screen_resume_v2(job_description: str = Form(...), resume_file: UploadFile = File(...)):
    try:
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        resume_text = "\n".join([p.extract_text() for p in pdf_reader.pages if p.extract_text()])

        if not resume_text.strip():
            return {"review_result": "[FAIL] System: Could not extract text from PDF."}

        prompt = f"""
        ROLE: Ruthless Technical HR & Hiring Manager.
        EVALUATE:
        Job Description: {job_description}
        Candidate Resume: {resume_text}
        
        STRICT INSTRUCTION:
        If JD asks for eligibility (Gig Check), start with EXACTLY [PASS] or [FAIL].
        Then provide:
        1. Match Score (0-100%).
        2. Core Strengths.
        3. Critical Weaknesses.
        4. 3 Hard-core technical interview questions.
        """
        response = model.generate_content(prompt)
        return {"review_result": response.text}
    except Exception as e:
        return {"review_result": f"[FAIL] Server Error: {str(e)}"}


# ==========================================
# 路由 3：财报追问对话 (CFO 模式)
# ==========================================
class FinanceChatRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_chat")
async def finance_chat(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Wall Street CFO. Creating a {req.report_type} Financial Report.
        HISTORY: {req.history}
        
        TASK:
        1. Analyze if more data is needed for a 1500-word deep report.
        2. If missing info, ask ONE sharp question. Append [PROGRESS: X/5] at the end.
        3. If data is exhaustive, reply with ONLY the word: [READY]
        """
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"CFO Connection Lost: {str(e)}"}


# ==========================================
# 路由 4：长篇专业财报生成 (1500字极致版)
# ==========================================
@app.post("/finance_report")
async def finance_report(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Top-tier Senior Auditor.
        Based on: {req.history}
        
        GENERATE A MASTERPIECE {req.report_type} FINANCIAL REPORT.
        
        STANDARDS:
        - Use US GAAP/SEC standards.
        - Include metrics: EBITDA, YoY/QoQ Growth, Free Cash Flow, CapEx, Burning Rate.
        - Length: MUST be exhaustive and detailed (aim for 1500+ words).
        - Format: Use ALL-CAPS headers, bullet points, NO markdown code blocks.
        """
        # 2.5 Flash 的生成上限很高，可以轻松处理长文
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 4096})
        return {"report": response.text}
    except Exception as e:
        return {"report": f"Gen Error: {str(e)}"}


# ==========================================
# 启动
# ==========================================
if __name__ == "__main__":
    # Render 会自动注入 PORT 环境变量
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
