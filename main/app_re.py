import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# ------------------------------------------------------------------- config
MODEL_PATH   = Path("../models/cnn_model_LIBRAS_note_re_20250421_2257.h5")
THRESHOLD    = 0.50
ROI_TOPLEFT  = (425, 100)
ROI_BOTRIGHT = (625, 300)
IMG_SIZE     = (64, 64)
# --------------------------------------------------------------------------

model = load_model(MODEL_PATH)
print("[INFO] Modelo carregado:", MODEL_PATH)

def predict_re(roi_bgr):
    img = cv2.resize(roi_bgr, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = img[np.newaxis, ...]                 # (1, 64, 64, 3)
    prob = model.predict(img, verbose=0)[0][0]
    return prob, prob >= THRESHOLD

# --------- **AQUI** criamos a webcam antes do loop ------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam (id 0).")

while True:
    ret, frame = cam.read()                    # agora 'cam' existe
    if not ret:
        print("Frame vazio — encerrando.")
        break

    frame = cv2.flip(frame, 1)
    x1, y1 = ROI_TOPLEFT
    x2, y2 = ROI_BOTRIGHT
    cv2.rectangle(frame, ROI_TOPLEFT, ROI_BOTRIGHT, (0, 255, 0), 2)

    roi = frame[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]
    prob, is_re = predict_re(roi)

    label = f"re ({prob:.2f})" if is_re else f"--- ({prob:.2f})"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 255) if is_re else (200, 200, 200), 2)

    cv2.imshow("ROI", roi)
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:            # ESC para sair
        break

cam.release()
cv2.destroyAllWindows()
