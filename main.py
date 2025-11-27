from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from framework.processor import Processor

app = FastAPI()
processor = Processor()

class Frame(BaseModel):
    image: str  # base64 string

@app.post("/detect")
async def detect(frame: Frame):
    img_bytes = base64.b64decode(frame.image.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detections = processor.process_frame(img)
    return detections
