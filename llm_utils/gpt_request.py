import os
import base64
import cv2
import numpy as np
from mimetypes import guess_type
from openai import OpenAI

# -----------------------------------------------------------------------------
# 初始化 OpenRouter Client
# 確保你已經設定了 export OPENROUTER_API_KEY="你的金鑰"
# -----------------------------------------------------------------------------
openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# 在 OpenRouter 中，模型名稱需要加上供應商前綴
# 你可以使用 'openai/gpt-4o'，或者替換成 'meta-llama/llama-3.2-90b-vision-instruct' 來進行對比
MODEL_NAME = "openai/gpt-4o"

# Function to encode a local image into data URL 
def local_image_to_data_url(image):
    if isinstance(image, str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image, np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

def gptv_response(text_prompt, image_prompt, system_prompt=""):
    """處理帶有影像的 Vision 請求"""
    prompt = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': text_prompt},
            {'type': 'image_url', 'image_url': {'url': local_image_to_data_url(image_prompt)}}
        ]}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prompt,
        max_tokens=1000,
        timeout=30.0   # 加上這行：如果 30 秒沒回應就強制拋出 Exception，進入你的 Retry 迴圈
    )
    return response.choices[0].message.content

# def gpt_response(text_prompt, system_prompt=""):
#     """處理純文字請求"""
#     prompt = [
#         {'role': 'system', 'content': system_prompt},
#         {'role': 'user', 'content': [{'type': 'text', 'text': text_prompt}]}
#     ]

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=prompt,
#         max_tokens=1000
#     )
#     return response.choices[0].message.content