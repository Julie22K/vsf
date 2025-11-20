import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from framework.processor import Processor

app = FastAPI()
processor = Processor()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            detections = processor.process_frame(frame)

            await ws.send_json(detections)

        except Exception as e:
            print("WebSocket error:", e)
            break
