import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import uvicorn

# 1. 配置 Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 使用 2.5 Flash 模型
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. 启动 FastAPI 实例
app = FastAPI()

# 配置跨域 (CORS) - 允许前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# [防坑关键] 根目录心跳检测 - 专门给 Render 看的
# ==========================================
@app.get("/")
def read_root():
    return {"status": "200 OK", "message": "AI Boss Backend is running smoothly!"}


# ==========================================
# 接口 1：AI 简历与兼职审核
# ==========================================
@app.post("/screen_resume_v2")
async def screen_resume_v2(job_description: str = Form(...), resume_file: UploadFile = File(...)):
    try:
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        resume_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + "\n"

        if not resume_text.strip():
            return {"review_result": "[FAIL] System Error: 无法从 PDF 中提取有效文字，请确保 PDF 不是纯图片扫描件。"}

        prompt = f"""
        You are an elite, ruthless Technical HR and Hiring Manager.
        Here is the Job Description or Gig Requirement:
        {job_description}
        
        Here is the Candidate's Resume Text:
        {resume_text}
        
        INSTRUCTION:
        If the prompt requires [PASS] or [FAIL] for a gig, you MUST start your response with EXACTLY [PASS] or [FAIL], followed by a professional, concise justification.
        Otherwise, provide a structured evaluation:
        1. Match Score (0-100%)
        2. Core Strengths
        3. Critical Weaknesses
        4. Provide 3 highly difficult technical interview questions.
        """
        
        response = model.generate_content(prompt)
        return {"review_result": response.text}
        
    except Exception as e:
        return {"review_result": f"[FAIL] Server processing error: {str(e)}"}


# ==========================================
# 接口 2：财务数据追问 (CFO 审查模式)
# ==========================================
class FinanceChatRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_chat")
async def finance_chat(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Wall Street CFO with 20 years of experience at a Big 4 accounting firm.
        The user is generating a {req.report_type} financial report.
        
        Here is the conversation and data history so far:
        {req.history}
        
        YOUR TASK:
        Analyze the data provided. If crucial financial metrics are missing, ask EXACTLY ONE sharp, professional question to extract the missing data. 
        Do NOT generate the report yet. Just ask the question.
        
        MANDATORY FORMAT:
        At the very end of your response, you MUST append a progress tag like [PROGRESS: X/Y]. E.g., [PROGRESS: 1/3].
        
        IF you believe you have enough data to generate a highly professional, 1500-word report, you MUST output ONLY the exact word: [READY]
        """
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}


# ==========================================
# 接口 3：长篇专业财报生成 (CFO 撰写模式)
# ==========================================
class FinanceReportRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_report")
async def finance_report(req: FinanceReportRequest):
    try:
        prompt = f"""
        You are a top-tier Wall Street CFO and Auditor.
        Based on the following data gathered from the user:
        {req.history}
        
        GENERATE A HIGHLY PROFESSIONAL {req.report_type} FINANCIAL REPORT.
        
        REQUIREMENTS:
        1. Length: MUST be exhaustive, detailed, and extremely professional (simulate a 1500+ word document).
        2. Terminology: Use advanced financial terminology (e.g., EBITDA, YoY Growth, CapEx, Amortization, Burn Rate, FCF).
        3. Structure for Corporate: Include Executive Summary, Income Statement Analysis, Balance Sheet Overview, Cash Flow & Liquidity, Risk Factors, and Forward-Looking Statements.
        4. Structure for Personal: Include Cash Flow Analysis, Net Worth Estimation, Asset Allocation, Burn Rate, and Wealth Management Advisory.
        5. Tone: Objective, analytical, cold, and strictly professional.
        6. Format using clear ALL-CAPS headers and bullet points. Do NOT use markdown code blocks.
        """
        response = model.generate_content(prompt)
        return {"report": response.text}
    except Exception as e:
        return {"report": f"Generation Error: {str(e)}"}


# ==========================================
# [防坑关键] 引擎启动模块
# ==========================================
if __name__ == "__main__":
    # 动态获取 Render 分配的端口，如果获取不到默认用 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
