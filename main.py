import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import uvicorn

# 1. 配置核心引擎
# 请确保在 Render 的 Environment Variables 中配置了 GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 使用 Gemini 1.5 Flash 模型 (平衡速度与逻辑能力)
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

# 2. 究极跨域配置 (允许你的 GitHub Pages 前端访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render 健康检查接口
@app.get("/")
def read_root():
    return {"status": "Live", "message": "AI Boss Backend for Financial Hub is active."}

# ==========================================
# 接口 1：AI 简历审核 (对应 screen_resume_v2)
# ==========================================
@app.post("/screen_resume_v2")
async def screen_resume_v2(job_description: str = Form(...), resume_file: UploadFile = File(...)):
    try:
        # 解析 PDF
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        resume_text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + "\n"

        if not resume_text.strip():
            return {"review_result": "[FAIL] 无法读取 PDF 内容，请检查文件格式。"}

        # 构造 HR 指令
        prompt = f"""
        Role: Elite Technical HR Director.
        Task: Evaluate the candidate based on JD: {job_description}
        Resume Content: {resume_text}
        
        Requirements:
        1. If this is a skill check, start your answer with [PASS] or [FAIL].
        2. Provide a match score (0-100%).
        3. Highlight 3 strengths and 2 fatal weaknesses.
        4. Provide 3 high-level interview questions.
        """
        
        response = model.generate_content(prompt)
        return {"review_result": response.text}
    except Exception as e:
        return {"review_result": f"Processing Error: {str(e)}"}

# ==========================================
# 接口 2：财报追问对话 (对应 finance_chat)
# ==========================================
class FinanceChatRequest(BaseModel):
    history: str
    report_type: str

@app.post("/finance_chat")
async def finance_chat(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Wall Street CFO and Senior Auditor. 
        Context: The user is generating a {req.report_type} Financial Report.
        
        Data History:
        {req.history}
        
        Task:
        1. Check if the provided data is enough for a deep, 1500-word professional report.
        2. If NOT enough, ask EXACTLY ONE sharp, professional question to get missing info.
        3. You MUST end your response with a progress tag: [PROGRESS: X/5].
        4. If you have enough data, respond with ONLY one word: [READY]
        """
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"CFO Connection Error: {str(e)}"}

# ==========================================
# 接口 3：长篇专业财报生成 (对应 finance_report)
# ==========================================
@app.post("/finance_report")
async def finance_report(req: FinanceChatRequest):
    try:
        prompt = f"""
        You are a Top-tier CFO. Based on the following data:
        {req.history}
        
        GENERATE A PROFESSIONAL {req.report_type} FINANCIAL REPORT.
        
        Strict Requirements:
        1. Professional Tone: Use EBITDA, YoY, CapEx, FCF terms.
        2. Length: Must be exhaustive and extremely detailed (simulate 1500 words).
        3. Format: Use ALL-CAPS headers. Do NOT use markdown code blocks (```).
        4. Structure: Include Executive Summary, Analysis, Risks, and Future Guidance.
        """
        # 设置较大的 max_output_tokens 以保证财报长度
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 4096})
        return {"report": response.text}
    except Exception as e:
        return {"report": f"Generation Error: {str(e)}"}

# ==========================================
# 启动模块
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
