from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
from train_model import predict_sign

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

@app.post("/translate-sign")
async def translate_sign(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB as MediaPipe requires
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks:
        return {"text": "No hand detected"}

    # Extract landmarks for the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1, 1)

    # Predict sign from landmarks
    translated_text = predict_sign(landmarks)

    return {"text": translated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
