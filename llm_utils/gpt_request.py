import os
import base64
import cv2
import numpy as np
from mimetypes import guess_type
from openai import OpenAI

# -----------------------------------------------------------------------------
# export OPENROUTER_API_KEY="<YOUR KEYS>"
# -----------------------------------------------------------------------------
openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

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
        timeout=30.0
    )
    return response.choices[0].message.content