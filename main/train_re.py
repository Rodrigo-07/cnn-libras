#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Lucas Lacerda | adaptado por ChatGPT
date  : 05/2019 → 2025‑04‑21

"""

from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from cnn import Convolucao                        # sua arquitetura
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import datetime, time, os, h5py                   # noqa: F401
import math

# -------------------------- hiper‑parâmetros --------------------------------
EPOCHS      = 30
IMG_SIZE    = (64, 64)
CHANNELS    = 3
BATCH_SIZE  = 32
FILE_PREFIX = "cnn_model_LIBRAS_re_vs_bg_"
DATASET_DIR = "../dataset/datasetsRe"               
# ----------------------------------------------------------------------------

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def minutes(start, end):
    return (end - start) / 60.0

print(f"[INFO] [INICIO]: {timestamp()}\n")

# ------------------------- geradores de dados -------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25
)

train_set = train_gen.flow_from_directory(
    f"{DATASET_DIR}/training",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    subset="training"
)

val_set = train_gen.flow_from_directory(
    f"{DATASET_DIR}/training",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
    subset="validation"
)

print(f"[INFO] Classes mapeadas: {train_set.class_indices}")   # {'bg':0,'re':1}
print(f"[INFO] Total imagens treino: {train_set.n}")
print(f"[INFO] Total imagens validação: {val_set.n}")

# --------------------------- construção do modelo ---------------------------
print("[INFO] Inicializando e compilando a CNN…")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# ---------- cria modelo base (saída dummy) ----------------------------------
base = Convolucao.build(*IMG_SIZE, CHANNELS, 1)   # última Dense(1) mas softmax?
# força última camada sigmoid, se necessário
if not isinstance(base.layers[-1], layers.Dense) or base.layers[-1].activation.__name__ != "sigmoid":
    base.pop()                                    # remove última
    base.add(layers.Dense(1, activation="sigmoid"))

model = base
model.compile(
    optimizer=SGD(0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------ treinamento ---------------------------------
start = time.time()
history = model.fit_generator(
    train_set,
    steps_per_epoch=train_set.n // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_set,
    validation_steps=val_set.n // BATCH_SIZE,
    callbacks=[early_stop],
    shuffle=True,
    workers=os.cpu_count(),
    use_multiprocessing=True,
    verbose=2
)

runtime = minutes(start, time.time())

# ------------------------------ salvando ------------------------------------
fname = f"{FILE_PREFIX}{timestamp()}.h5"
model.save(f"../models/{fname}")
print(f"[INFO] Modelo salvo em ../models/{fname}")
print(f"[INFO] Tempo total de treino: {runtime:.1f} min\n")

# ---------------- avaliação + relatório ----------------
# calcula quantos lotes cobrem TODO o conjunto
val_steps = val_set.n // BATCH_SIZE
if val_set.n % BATCH_SIZE != 0:
    val_steps += 1               # garante pegar o resto

loss, acc = model.evaluate_generator(val_set,
                                     steps=val_steps,
                                     verbose=0)
print(f"[INFO] Val Accuracy: {acc*100:.2f}% | Val Loss: {loss:.4f}")

# previsões
y_pred_prob = model.predict_generator(val_set,
                                      steps=val_steps,
                                      verbose=0)

y_pred = (y_pred_prob > 0.5).astype(int).ravel()
y_true = val_set.classes

from sklearn.metrics import confusion_matrix, classification_report
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["bg", "re"]))


# ------------------------------ gráficos ------------------------------------
plt.style.use("ggplot")
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(f"../models/graphics/{FILE_PREFIX}{timestamp()}.png")
print("[INFO] Gráficos salvos em ../models/graphics/\n")

# ----------------------- diagrama da arquitetura ---------------------------
plot_model(model, to_file=f"../models/image/{FILE_PREFIX}{timestamp()}.png",
           show_shapes=True)

print(f"[INFO] [FIM]: {timestamp()}\n")
