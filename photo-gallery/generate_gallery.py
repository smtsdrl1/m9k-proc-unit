"""
HTML galeri dosyasını otomatik oluşturur.
images/processed/ klasöründeki PNG'leri tarar ve index.html üretir.

Kullanım:
  python generate_gallery.py

  # WordPress sipariş URL'nizi girin (ör: https://siteniz.com/siparis)
"""

from pathlib import Path

WP_ORDER_URL = "https://SİTENİZ.com/siparis"  # <-- Buraya kendi WP linkinizi yazın
PRICE = "300"

images = sorted(Path("images/processed").glob("*.png"))
print(f"{len(images)} işlenmiş fotoğraf bulundu.")

cards = ""
for img in images:
    cards += f"""
        <div class="card">
            <div class="img-wrap">
                <img src="images/processed/{img.name}" alt="Ürün" loading="lazy">
            </div>
            <div class="card-footer">
                <span class="price">{PRICE} ₺</span>
                <a href="{WP_ORDER_URL}?urun={img.stem}" class="btn" target="_blank">Sipariş Ver</a>
            </div>
        </div>"""

html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ürünler</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
    padding: 32px 16px;
  }}

  h1 {{
    text-align: center;
    margin-bottom: 32px;
    font-size: 2rem;
    color: #222;
    letter-spacing: -0.5px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 24px;
    max-width: 1200px;
    margin: 0 auto;
  }}

  .card {{
    background: #fff;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    transition: transform .2s, box-shadow .2s;
  }}

  .card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,.14);
  }}

  .img-wrap {{
    background: linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    height: 220px;
  }}

  .img-wrap img {{
    max-width: 100%;
    max-height: 180px;
    object-fit: contain;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,.18));
  }}

  .card-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    border-top: 1px solid #f0f0f0;
  }}

  .price {{
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1a1a;
  }}

  .btn {{
    background: #ff4b2b;
    color: #fff;
    text-decoration: none;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: .9rem;
    font-weight: 600;
    transition: background .15s;
  }}

  .btn:hover {{ background: #e03a1e; }}

  @media (max-width: 480px) {{
    .grid {{ grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .img-wrap {{ height: 160px; }}
  }}
</style>
</head>
<body>
<h1>Ürünler</h1>
<div class="grid">{cards}
</div>
</body>
</html>"""

Path("index.html").write_text(html, encoding="utf-8")
print("index.html oluşturuldu!")
