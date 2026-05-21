"""
Toplu arka plan kaldırma scripti.
Kullanım:
  pip install rembg pillow
  python remove_bg.py
"""

from pathlib import Path
from rembg import remove
from PIL import Image
import io

INPUT_DIR = Path("images/original")
OUTPUT_DIR = Path("images/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED]
print(f"{len(files)} fotoğraf bulundu, işleniyor...\n")

for i, path in enumerate(files, 1):
    output_path = OUTPUT_DIR / (path.stem + ".png")
    if output_path.exists():
        print(f"[{i}/{len(files)}] Atlandı (zaten var): {path.name}")
        continue
    print(f"[{i}/{len(files)}] İşleniyor: {path.name}")
    with open(path, "rb") as f:
        result = remove(f.read())
    img = Image.open(io.BytesIO(result)).convert("RGBA")
    img.save(output_path, "PNG")

print("\nTamamlandı! Sonuçlar: images/processed/")
