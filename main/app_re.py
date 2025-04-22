#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web‑cam detector  –  versão 2  (modelo binário re × bg)

• O modelo salvo em 2025‑04‑21_2324 foi treinado com:
    classes = {'Re': 0, 'bg': 1}      (Keras ordena pastas por nome)
    saída   = Dense(1, sigmoid)

⇒ O escalar que a rede devolve é **P(bg)**.  
   A probabilidade do gesto **re** é, portanto, `1 ‑ P(bg)`.
"""

import cv2, numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# ------------------------------- CONFIG -------------------------------------
MODEL_PATH  = Path("../models/cnn_model_LIBRAS_re_vs_bg_20250421_2325.h5")
THRESHOLD   = 0.50                     # prob_re ≥ 0.5  ⇒  “re”
ROI_TL      = (425, 100)               # (x1, y1)
ROI_BR      = (625, 300)               # (x2, y2)
IMG_SIZE    = (64, 64)                 # esperado pelo modelo
# ----------------------------------------------------------------------------

model = load_model(MODEL_PATH)
print("[INFO] Modelo carregado:", MODEL_PATH)

def predict_re(roi_bgr):
    """Recebe ROI BGR, devolve probabilidade de ser 're' e bool."""
    img = cv2.resize(roi_bgr, IMG_SIZE).astype("float32") / 255.0
    img = img[np.newaxis, ...]                     # shape (1,64,64,3)
    raw = model.predict(img, verbose=0)

    # === modelo tem shape (1,1) → saída é P(bg) =============================
    if raw.shape[-1] == 1:
        prob_re = 1.0 - raw[0][0]                 # P(re) = 1 − P(bg)
    # === se por acaso você trocar para Softmax(2) ===========================
    else:                                         # shape (1,2)
        RE_INDEX = 1 if raw.shape[-1] > 1 else 0  # ajuste se necessário
        prob_re = raw[0][RE_INDEX]

    return prob_re, prob_re >= THRESHOLD

# ----------------------------- LOOP CAM -------------------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam (id 0).")

while True:
    ok, frame = cam.read()
    if not ok:
        print("Frame vazio — saindo.")
        break
    frame = cv2.flip(frame, 1)

    x1, y1 = ROI_TL
    x2, y2 = ROI_BR
    cv2.rectangle(frame, ROI_TL, ROI_BR, (0, 255, 0), 2)

    roi = frame[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2]
    prob, is_re = predict_re(roi)

    label = f"re ({prob:.2f})" if is_re else f"bg ({prob:.2f})"
    color = (0, 0, 255) if is_re else (200, 200, 200)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("ROI", roi)
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:      # Esc = sair
        break

cam.release()
cv2.destroyAllWindows()
