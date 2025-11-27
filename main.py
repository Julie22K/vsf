import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from framework.processor import Processor

app = FastAPI()
processor = Processor()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vsf2.onrender.com"],  # фронтенд
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Frame(BaseModel):
    image: str

@app.post("/detect")
async def detect(frame: Frame):
    img_bytes = base64.b64decode(frame.image.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detections = processor.process_frame(img)
    return detections
