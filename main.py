from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # 👈 导入保安工具

load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# 🛡️ 这是最重要的“跨域通行证”！没有这段，网页永远连不上
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有网站（包括你的 GitHub Pages）
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有请求方式
    allow_headers=["*"],           # 允许所有请求头
)

if my_api_key:
    genai.configure(api_key=my_api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

class ScreenRequest(BaseModel):
    job_description: str
    resume_text: str

@app.get("/")
def read_root():
    return {"message": "我的 AI 招聘后台终于跑通啦！CORS 已经开启！"}

@app.post("/screen_resume")
def screen_resume(data: ScreenRequest):
    prompt = f"HR要求：{data.job_description}。简历：{data.resume_text}。请简短评价并打分。"
    response = model.generate_content(prompt)
    return {"review_result": response.text}
