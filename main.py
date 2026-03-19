from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 加载 .env 里的密码
load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

# 初始化后台
app = FastAPI()

# 配置密码
if my_api_key:
    genai.configure(api_key=my_api_key)

# 召唤模型
model = genai.GenerativeModel('gemini-2.5-flash')

class ScreenRequest(BaseModel):
    job_description: str
    resume_text: str

# 👇 重点在这里！这是让网页显示接口的魔法
@app.get("/")
def read_root():
    return {"message": "我的 AI 招聘后台终于跑通啦！"}

@app.post("/screen_resume")
def screen_resume(data: ScreenRequest):
    prompt = f"HR要求：{data.job_description}。简历：{data.resume_text}。请简短评价并打分。"
    response = model.generate_content(prompt)
    return {"review_result": response.text}