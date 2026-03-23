import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import uvicorn

# 1. 配置核心引擎
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 使用 Gemini 1.5 Flash
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# 2. 究极跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render 健康检查
@app.get("/")
def read_root():
    return {"status": "Live", "message": "AI Boss Universal Engine is active."}

# ==========================================
# 【新增】接口 0：智能记账解析 (解决你之前的报错)
# ==========================================
class LedgerRequest(BaseModel):
    text: str

@app.post("/ledger_ai")
async def ledger_ai(req: LedgerRequest):
    try:
        # 这个 Prompt 是专门为你前端那种“分类显示+详情展开”设计的
        prompt = f"""
        Act as a professional personal accountant. 
        Analyze this spending description: "{req.text}"
        
        Output format: Category|Short_Summary|Detailed_Explanation|Amount_Number
        
        Requirements:
        1. Category: One word (e.g., Food, Transport, Tech, Entertainment).
        2. Short_Summary: Max 8 words for the list view.
        3. Detailed_Explanation: 2 professional sentences for the expanded view.
        4. Amount_Number: Extract only the number. Default to 0 if not found.
        
        Example: "Spent 50 bucks on a nice steak dinner" -> Food|Steak Dinner|A high-quality dining expense for personal nutrition and leisure.|50
        """
        response = model.generate_content(prompt)
        return {"data": response.text}
    except Exception as e:
        return {"data": f"Misc|Error|Processing failed: {str(e)}|0"}

# ==========================================
# 接口 1：AI 简历审核 (对应 screen_resume_v2)
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
            return {"review_result": "[FAIL] 无法读取 PDF 内容。"}

        prompt = f"Role: Elite HR. Evaluate JD: {job_description} with Resume: {resume_text}. If skill check, start with [PASS] or [FAIL]."
        response = model.generate_content(prompt)
        return {"review_result": response.text}
    except Exception as e:
        return {"review_result": f"Error: {str(e)}"}

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
        You are a Wall Street CFO. The user is generating a {req.report_type} Financial Report.
        History: {req.history}
        Task: Ask ONE sharp question to get missing info. End with [PROGRESS: X/5]. If ready, reply ONLY [READY].
        """
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# ==========================================
# 接口 3：长篇专业财报生成 (对应 finance_report)
# ==========================================
@app.post("/finance_report")
async def finance_report(req: FinanceChatRequest):
    try:
        prompt = f"Role: Top CFO. Based on: {req.history}, generate an exhaustive 1500-word {req.report_type} Report. Use ALL-CAPS headers."
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 4096})
        return {"report": response.text}
    except Exception as e:
        return {"report": f"Gen Error: {str(e)}"}

# 启动
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
