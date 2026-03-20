from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import google.generativeai as genai
import os
import io
import PyPDF2
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if my_api_key:
    genai.configure(api_key=my_api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

# 验证系统是否升级成功的标志
@app.get("/")
def read_root():
    return {"message": "引擎运行正常！系统已升级支持 PDF 文件解析。CORS 已开启！"}

# 兼容旧版本
class ScreenRequest(BaseModel):
    job_description: str
    resume_text: str

@app.post("/screen_resume")
def screen_resume(data: ScreenRequest):
    prompt = f"HR要求：{data.job_description}。简历：{data.resume_text}。请简短评价并打分。"
    response = model.generate_content(prompt)
    return {"review_result": response.text}

# 🚀 全新的 V2 接口：专门吃 PDF！
@app.post("/screen_resume_v2")
async def screen_resume_v2(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        # 提取 PDF 文字
        pdf_bytes = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        resume_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"
                
        if not resume_text.strip():
            return {"review_result": "[WARNING] 解析失败：这份 PDF 可能是纯图片扫描件，未能提取到文字信息。"}

        # 极客风格的 Prompt
        prompt = f"""
        > SYSTEM_OVERRIDE_ACTIVE
        > ROLE: 资深技术招聘专家 (10年+ 经验)
        
        [岗位核心要求]
        {job_description}
        
        [候选人简历数据]
        {resume_text}
        
        请执行深度匹配分析，并严格按照以下结构输出报告（保持语言极客、专业）：
        【匹配指数】：(给出0-100分的精准分数，并标注评级 S/A/B/C)
        【核心能力锚点】：(提取1-3个候选人最符合岗位的亮点)
        【潜在风险/缺失项】：(冷酷地指出候选人欠缺或描述模糊的地方)
        【最终决策】：(明确结论：强烈推荐 / 建议面试 / 储备 / 不匹配)
        """
        
        response = model.generate_content(prompt)
        return {"review_result": response.text}
        
    except Exception as e:
        return {"review_result": f"[FATAL ERROR] PDF 解析模块崩溃，详情：{str(e)}"}
