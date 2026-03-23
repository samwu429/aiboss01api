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
# 请确保在 Render 的 Environment Variables 中正确配置了 GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 严格遵从老板指示：使用 2.5 Flash 模型
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# 究极跨域配置，确保前端调用不被拦截
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render 健康检查接口，防止服务器休眠后启动失败
@app.get("/")
def read_root():
    return {"status": "Live", "engine": "Gemini 2.5 Flash", "message": "AI Boss Universal Backend is running smoothly."}


# ==========================================
# 路由 1：智能记账解析 (解决多笔账单混合、正负数不分的问题)
# ==========================================
class LedgerRequest(BaseModel):
    text: str

@app.post("/ledger_ai")
async def ledger_ai(req: LedgerRequest):
    try:
        # 核心升级：强迫 AI 拆分多行，并严格区分正负数
        prompt = f"""
        You are a highly precise AI Personal Accountant.
        Analyze the user's input: "{req.text}"
        
        CRITICAL INSTRUCTION:
        If the input contains MULTIPLE distinct financial transactions (e.g., withdrawing money AND spending money), you MUST output EACH transaction on a completely NEW LINE.
        Do NOT combine income and expenses mathematically.
        
        For EACH transaction, use EXACTLY this format separated by a pipe (|):
        Category|Short_Summary|Detailed_Explanation|Amount_Number
        
        RULES for Amount_Number:
        - If it's INCOME, receiving money, or withdrawing cash to hold, make it a POSITIVE number (e.g., 1200).
        - If it's an EXPENSE, spending, or paying, make it a NEGATIVE number (e.g., -100).
        - Output the raw number only, no currency symbols.
        
        Example Input: "Withdraw 1200 for living expenses, but already spent 100 on mahjong."
        Example Output:
        Income|Cash Withdrawal|Withdrew 1200 cash for upcoming monthly living expenses.|1200
        Entertainment|Mahjong Expense|Pre-spent 100 on entertainment (mahjong) from the budget.|-100
        """
        response = model.generate_content(prompt)
        return {"data": response.text.strip()}
    except Exception as e:
        return {"data": f"Error|System Error|Failed to parse data: {str(e)}|0"}


# ==========================================
# 路由 2：AI 简历审核 & 兼职资格拦截 (HR 模块)
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
            return {"review_result": "[FAIL] 无法读取 PDF 内容，请检查文件是否为纯图片扫描件。"}

        prompt = f"""
        ROLE: Elite Technical HR Director.
        TASK: Evaluate candidate based on JD.
        
        JD: {job_description}
        Resume: {resume_text}
        
        CRITICAL RULE:
        1. If the JD requires a pass/fail check (e.g., for a gig), you MUST start your response with EXACTLY [PASS] or [FAIL].
        2. Then, provide a Match Score (0-100%).
        3. Highlight 3 core strengths and 2 critical weaknesses.
        4. Provide 3 tough technical interview questions.
        """
        response = model.generate_content(prompt)
        return {"review_result": response.text}
    except Exception as e:
        return {"review_result": f"[FAIL] Server Error: {str(e)}"}


# ==========================================
# 路由 3：财报追问对话 (CFO 审计模式)
# ==========================================
class FinanceChatRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_chat")
async def finance_chat(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Wall Street CFO.
        The user is generating a {req.report_type} Financial Report.
        
        Data History:
        {req.history}
        
        Task:
        1. Evaluate if the data is sufficient for a highly professional, 1500-word report.
        2. If NOT sufficient, ask EXACTLY ONE professional question to gather missing data. 
        3. You MUST append a progress tag at the end of your question, like [PROGRESS: X/5].
        4. If you have enough data, respond with ONLY the word: [READY]
        """
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"CFO Connection Error: {str(e)}"}


# ==========================================
# 路由 4：长篇专业财报生成 (1500字长文模式)
# ==========================================
@app.post("/finance_report")
async def finance_report(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Top-tier Senior Auditor.
        Based on the following context: {req.history}
        
        GENERATE A HIGHLY PROFESSIONAL {req.report_type} FINANCIAL REPORT.
        
        Requirements:
        1. Length: Must be exhaustive and extremely detailed (simulate 1500+ words).
        2. Tone: Objective, analytical, and compliant with US GAAP/SEC standards.
        3. Terminology: Use EBITDA, FCF, CapEx, YoY Growth, and Amortization where applicable.
        4. Format: Use clear ALL-CAPS headers. DO NOT use markdown code blocks (```).
        """
        # 设置 max_output_tokens 确保长文不会被截断
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 4096})
        return {"report": response.text}
    except Exception as e:
        return {"report": f"Generation Error: {str(e)}"}


# ==========================================
# 引擎启动模块
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
