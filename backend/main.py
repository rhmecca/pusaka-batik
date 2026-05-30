from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

app = FastAPI()

# Mengizinkan komunikasi lintas port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model AI
MODEL_PATH = "15_motives.tflite"
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    print(f"Warning: Model tidak ditemukan atau gagal dimuat. Error: {e}")
    model_loaded = False

MOTIVES_LIST = [
    "Batik Bali", "Batik Betawi", "Batik Cendrawasih", "Batik Dayak", 
    "Batik Geblek Renteng", "Batik Ikat Celup", "Batik Insang", "Batik Kawung", 
    "Batik Lasem", "Batik Megamendung", "Batik Pala", "Batik Parang", 
    "Batik Poleng", "Batik Sekar Jagad", "Batik Tambal"
]

@app.post("/predict")
async def predict_batik(file: UploadFile = File(...)):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"message": "Model AI belum tersedia di server."})
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((300, 300))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        pred_idx = np.argmax(prediction[0])
        pred_motive = MOTIVES_LIST[pred_idx]
        confidence = float(prediction[0][pred_idx]) * 100
        
        return {"motif": pred_motive, "confidence": round(confidence, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Menyajikan file Frontend secara statis
# Endpoint ini diletakkan di bawah agar tidak menabrak rute /predict
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")