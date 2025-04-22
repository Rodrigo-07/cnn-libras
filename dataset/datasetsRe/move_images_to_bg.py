#!/usr/bin/env python3
"""
move_bg_letters.py
Seleciona N imagens aleatórias de cada letra (exceto 'Re') e copia para
as pastas bg/ de treino e teste.

Autor: ChatGPT
"""

import random
import shutil
from pathlib import Path

SRC_DIR       = Path("../training")
DEST_TRAIN_BG = Path("./training/bg")
DEST_TEST_BG  = Path("./testing/bg")

IMGS_TRAIN = 20   # por letra
IMGS_TEST  = 20   # por letra
RAND_SEED  = 42
# ----------------------------------------------------------------------------

random.seed(RAND_SEED)
DEST_TRAIN_BG.mkdir(parents=True, exist_ok=True)
DEST_TEST_BG.mkdir(parents=True,  exist_ok=True)

def pick_and_copy(letter_dir: Path):
    imgs = sorted([p for p in letter_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    need = IMGS_TRAIN + IMGS_TEST
    if len(imgs) < need:
        raise RuntimeError(f"'{letter_dir.name}' tem só {len(imgs)} imagens; "
                           f"precisa de ≥ {need}.")

    chosen = random.sample(imgs, need)
    train_imgs = chosen[:IMGS_TRAIN]
    test_imgs  = chosen[IMGS_TRAIN:]

    # copia mantendo o nome original; se houver duplicata, adiciona sufixo
    for dst_root, img_list in [(DEST_TRAIN_BG, train_imgs),
                               (DEST_TEST_BG,  test_imgs)]:
        for src_path in img_list:
            dst_path = dst_root / src_path.name
            # evita sobrescrever se nomes repetirem entre letras (raro)
            if dst_path.exists():
                dst_path = dst_root / f"{src_path.stem}_{letter_dir.name}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)

def main():
    letters = [d for d in SRC_DIR.iterdir() if d.is_dir() and d.name.lower() != "re"]

    for letter_dir in letters:
        print(f"Processando {letter_dir.name} …")
        pick_and_copy(letter_dir)

    print("✅ Copiado com sucesso!")
    print(f"Arquivos em {DEST_TRAIN_BG}: {len(list(DEST_TRAIN_BG.iterdir()))}")
    print(f"Arquivos em {DEST_TEST_BG}:  {len(list(DEST_TEST_BG.iterdir()))}")

if __name__ == "__main__":
    main()
