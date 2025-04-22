#!/usr/bin/env python3
import os, random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

INPUT_DIR  = Path("./musical_notes_original/Re_cut")
OUTPUT_DIR = Path("./musical_notes_original/Re")
TARGET_SZ: Tuple[int, int] = (64, 64)
AUGS_PER_IMG = 5

# ---------- augmentations ---------------------------------------------------
def random_rotate(img):
    return img.rotate(random.uniform(-20, 20), resample=Image.BICUBIC, expand=False)

def random_flip(img):
    return ImageOps.mirror(img)

def random_zoom(img):
    w, h = img.size
    perc = random.uniform(0.1, 0.2)
    dx, dy = int(w * perc), int(h * perc)
    left   = random.randint(0, dx)
    top    = random.randint(0, dy)
    right  = w - (dx - left)
    bottom = h - (dy - top)
    return img.crop((left, top, right, bottom)).resize(TARGET_SZ, Image.LANCZOS)

def random_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

def add_gaussian_noise(img):
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, 10, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

AUG_FUNCS = [
    random_rotate,
    random_flip,
    random_zoom,
    random_brightness,
    add_gaussian_noise,
]

# --------------------- novo: contador global --------------------------------
file_counter = 1  # ⇢ começa em 1 e só aumenta

def next_name() -> str:             # ⇢ gera "1.jpg", "2.jpg", ...
    global file_counter
    name = f"{file_counter}.jpg"
    file_counter += 1
    return name
# ----------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def process_image(path: Path):
    img = Image.open(path).convert("RGB").resize(TARGET_SZ, Image.LANCZOS)

    # salva original
    img.save(OUTPUT_DIR / next_name())   # ⇢ usa função next_name()

    # 5 variações
    for aug_func in AUG_FUNCS:
        aug_img = aug_func(img)
        aug_img.save(OUTPUT_DIR / next_name())

def main():
    ensure_dir(OUTPUT_DIR)
    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    if not files:
        print(f"[ERRO] Nenhuma imagem encontrada em {INPUT_DIR.resolve()}")
        return

    for idx, file in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processando {file.name} …")
        process_image(file)

    total = len(files) * (AUGS_PER_IMG + 1)
    print(f"✅ Concluído! {total} imagens salvas em {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
