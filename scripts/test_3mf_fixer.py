#!/usr/bin/env python3
"""
test_3mf_fixer.py — fix_3mf_colors.py için sentetik test

Bambu "Image to 3D" çıktısını taklit eden bir insan figürü mesh'i üretir:
- Bere (küre üst)
- Kafatası yüzü (silindir üst, öne bakan)
- Ceket gövdesi (silindir orta)
- Pantolon L/R (iki ayrı silindir alt)
- Ayakkabılar L/R (iki kutu)

Farklı min_island değerleri için adacık istatistiklerini rapor eder.
"""

import io
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np

# fix_3mf_colors'u import edebilmek için scripts/ dizinini ekle
sys.path.insert(0, str(Path(__file__).parent))
from fix_3mf_colors import (
    assign_colors_geometric,
    merge_small_islands,
    compute_face_normals,
    compute_face_centroids,
)

# ─── Sentetik mesh üretici ────────────────────────────────────────────────────

def sphere_mesh(cx, cy, cz, r, lat=12, lon=12):
    """Küre mesh döndürür (bere için)."""
    verts, tris = [], []
    for i in range(lat + 1):
        phi = np.pi * i / lat
        for j in range(lon):
            theta = 2 * np.pi * j / lon
            x = cx + r * np.sin(phi) * np.cos(theta)
            y = cy + r * np.cos(phi)
            z = cz + r * np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])

    base = 0
    for i in range(lat):
        for j in range(lon):
            a = base + i * lon + j
            b = base + i * lon + (j + 1) % lon
            c = base + (i + 1) * lon + j
            d = base + (i + 1) * lon + (j + 1) % lon
            tris += [[a, b, c], [b, d, c]]
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def cylinder_mesh(cx, cy_bot, cy_top, r, segs=20, add_caps=True):
    """Silindir mesh (gövde/pantolon için)."""
    verts, tris = [], []
    # Çevre halkası x2
    for y in [cy_bot, cy_top]:
        for j in range(segs):
            theta = 2 * np.pi * j / segs
            verts.append([cx + r * np.cos(theta), y, r * np.sin(theta)])

    base = 0
    for j in range(segs):
        a = j
        b = (j + 1) % segs
        c = segs + j
        d = segs + (j + 1) % segs
        tris += [[a, b, c], [b, d, c]]

    if add_caps:
        # Üst cap
        top_c = len(verts)
        verts.append([cx, cy_top, 0])
        for j in range(segs):
            tris.append([segs + j, segs + (j+1) % segs, top_c])
        # Alt cap
        bot_c = len(verts)
        verts.append([cx, cy_bot, 0])
        for j in range(segs):
            tris.append([j, bot_c, (j+1) % segs])

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def box_mesh(x0, y0, z0, x1, y1, z1):
    """Kutu mesh (ayakkabı için)."""
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ], dtype=np.float32)
    t = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ], dtype=np.int32)
    return v, t


def subdivide(verts, tris, times=3):
    """
    Mesh'i times kez subdivide et — Bambu Image-to-3D modeller
    genellikle çok yüksek yüz sayısına sahiptir (50K-300K).
    Her subdivide 4x yüz üretir.

    Bug fix: midpoint indeksi için len(verts_list) kullanılmalı,
    len(verts) sabit kalır ve tüm midpoint'ler aynı indekse yazılır.
    """
    for _ in range(times):
        new_tris = []
        edge_mid = {}
        verts_list = list(verts)   # her iterasyon taze başlar

        def mid(a, b, _vl=verts_list, _em=edge_mid):
            key = (min(a, b), max(a, b))
            if key not in _em:
                _em[key] = len(_vl)                        # ← düzeltme
                _vl.append((_vl[a] + _vl[b]) / 2)
            return _em[key]

        for t in tris:
            a, b, c = t
            ab = mid(a, b)
            bc = mid(b, c)
            ca = mid(c, a)
            new_tris += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]

        verts = np.array(verts_list, dtype=np.float32)
        tris  = np.array(new_tris,   dtype=np.int32)
    return verts, tris


def merge_meshes(*mesh_list):
    """Birden fazla (verts, tris) çiftini birleştir."""
    all_v, all_t = [], []
    offset = 0
    for v, t in mesh_list:
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    return np.vstack(all_v), np.vstack(all_t)


def build_humanoid_mesh(subdivisions=3):
    """
    Basit insan figürü mesh'i.
    subdivisions=3 → yaklaşık 40K-80K yüz (gerçekçi boyut).
    """
    meshes = []

    # Bere (en üst)
    v, t = sphere_mesh(0, 9.0, 0, 1.2, lat=10, lon=10)
    meshes.append(subdivide(v, t, subdivisions - 1))

    # Kafatası yüzü (önde biraz öne çıkık silindir)
    v, t = cylinder_mesh(0, 7.5, 9.0, 1.0, segs=16)
    meshes.append(subdivide(v, t, subdivisions - 1))

    # Ceket gövde
    v, t = cylinder_mesh(0, 4.0, 7.5, 1.4, segs=20)
    meshes.append(subdivide(v, t, subdivisions))

    # Pantolon sol
    v, t = cylinder_mesh(-0.55, 1.0, 4.0, 0.6, segs=14)
    meshes.append(subdivide(v, t, subdivisions))

    # Pantolon sağ
    v, t = cylinder_mesh(0.55, 1.0, 4.0, 0.6, segs=14)
    meshes.append(subdivide(v, t, subdivisions))

    # Ayakkabı sol
    v, t = box_mesh(-1.1, 0.0, -0.5, -0.05, 1.0, 0.5)
    meshes.append(subdivide(v, t, subdivisions))

    # Ayakkabı sağ
    v, t = box_mesh(0.05, 0.0, -0.5, 1.1, 1.0, 0.5)
    meshes.append(subdivide(v, t, subdivisions))

    return merge_meshes(*meshes)


# ─── Adacık analizi ───────────────────────────────────────────────────────────

def analyze_islands(tris, color_ids):
    """Renk başına adacık sayısı ve boyutlarını döndürür."""
    n_faces = len(tris)
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(tris):
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_to_faces[e].append(fi)

    adj = defaultdict(set)
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            adj[faces[0]].add(faces[1])
            adj[faces[1]].add(faces[0])

    visited = np.zeros(n_faces, dtype=bool)
    islands = []
    for start in range(n_faces):
        if visited[start]:
            continue
        cid = color_ids[start]
        comp = []
        queue = [start]
        visited[start] = True
        while queue:
            fi = queue.pop()
            comp.append(fi)
            for nb in adj[fi]:
                if not visited[nb] and color_ids[nb] == cid:
                    visited[nb] = True
                    queue.append(nb)
        islands.append((cid, len(comp)))

    return islands


def print_island_report(islands, n_colors, label=""):
    total = sum(s for _, s in islands)
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Toplam yüz   : {total:,}")
    print(f"  Toplam adacık: {len(islands)}")
    for c in range(n_colors):
        c_islands = [s for cid, s in islands if cid == c]
        if not c_islands:
            continue
        print(f"\n  Filament {c+1}:")
        print(f"    Adacık sayısı : {len(c_islands)}")
        print(f"    Toplam yüz    : {sum(c_islands):,}  ({sum(c_islands)/total*100:.1f}%)")
        print(f"    En büyük      : {max(c_islands):,}")
        print(f"    En küçük      : {min(c_islands):,}")
        print(f"    Medyan        : {int(np.median(c_islands)):,}")
        tiny = [s for s in c_islands if s < 500]
        print(f"    < 500 yüz     : {len(tiny)} adacık  ({sum(tiny):,} yüz)")


# ─── Sahte 3MF üretici ────────────────────────────────────────────────────────

def build_3mf_bytes(verts, tris):
    """Minimal geçerli 3MF byte dizisi döndürür (zipfile içinde XML)."""
    ns = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    lines = [f'<?xml version="1.0" encoding="UTF-8"?>',
             f'<model unit="millimeter" xmlns="{ns}">',
             f'  <resources><object id="1" type="model"><mesh>',
             f'    <vertices>']
    for x, y, z in verts:
        lines.append(f'      <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>')
    lines.append('    </vertices><triangles>')
    for v1, v2, v3 in tris:
        lines.append(f'      <triangle v1="{v1}" v2="{v2}" v3="{v3}"/>')
    lines += ['    </triangles></mesh></object></resources>',
              '  <build><item objectid="1"/></build>',
              '</model>']
    xml_bytes = "\n".join(lines).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("3D/3dmodel.model", xml_bytes)
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                    '</Types>')
    return buf.getvalue()


# ─── Ana test ────────────────────────────────────────────────────────────────

def main():
    N_COLORS = 4
    SUBDIVISIONS = 3   # 3 → ~60K yüz; 4 → ~240K (daha yavaş)

    print("=" * 55)
    print("  3MF Color Fixer — Sentetik Test")
    print("=" * 55)
    print(f"\n[1] Humanoid mesh üretiliyor (subdivision={SUBDIVISIONS})...")
    verts, tris = build_humanoid_mesh(subdivisions=SUBDIVISIONS)
    print(f"    Vertex  : {len(verts):,}")
    print(f"    Üçgen   : {len(tris):,}")

    # Mesh'in toplam yüksekliğini göster
    y = verts[:, 1]
    print(f"    Y aralığı: {y.min():.2f} — {y.max():.2f}")

    print(f"\n[2] Renk ataması ({N_COLORS} filament)...")
    color_ids_raw = assign_colors_geometric(verts, tris, n_colors=N_COLORS, debug=True)

    islands_raw = analyze_islands(tris, color_ids_raw)
    print_island_report(islands_raw, N_COLORS, label="Birleştirme YOK (ham geometrik atama)")

    # Farklı threshold değerleri test et
    thresholds = [200, 500, 1000, 2000]
    results = {}
    for thresh in thresholds:
        print(f"\n[3] min_island={thresh} ile birleştirme...")
        merged = merge_small_islands(tris, color_ids_raw, min_faces=thresh)
        islands = analyze_islands(tris, merged)
        results[thresh] = islands
        print_island_report(islands, N_COLORS, label=f"min_island = {thresh}")

    # Özet
    print("\n" + "=" * 55)
    print("  ÖZET — Threshold karşılaştırması")
    print("=" * 55)
    total = len(tris)

    # Otomatik eşik: meshın %1'i (fix_3mf_colors.py varsayılanı)
    auto_thresh = max(50, int(total * 1.0 / 100))
    print(f"\n  Mesh boyutu  : {total:,} yüz")
    print(f"  Auto (%1)    : {auto_thresh} yüz")
    print()
    print(f"  {'Threshold':>10}  {'Toplam Ada':>10}  {'Min ada boyutu':>15}  {'Not'}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*20}")

    raw_min = min(s for _, s in islands_raw)
    print(f"  {'HAM':>10}  {len(islands_raw):>10,}  {raw_min:>15,}  referans")

    scores = {}
    for thresh, islands in results.items():
        n     = len(islands)
        tag   = "auto (%1)" if thresh == auto_thresh else ""
        min_s = min(s for _, s in islands) if islands else 0
        print(f"  {thresh:>10,}  {n:>10,}  {min_s:>15,}  {tag}")
        scores[thresh] = n

    # En az adacık veren threshold'u öner
    best = min(scores, key=lambda k: scores[k])
    best_n = scores[best]

    # 2000'den sonra adacık sayısı artıyorsa (agresif merge bozuyor) geri dön
    ordered = sorted(scores.items(), key=lambda x: x[0])
    for i in range(1, len(ordered)):
        if ordered[i][1] > ordered[i-1][1]:
            best = ordered[i-1][0]
            break

    print(f"\n  Önerilen min_island : {best:,}  ({best/total*100:.2f}% of mesh)")
    print(f"  Otomatik (%1)       : {auto_thresh:,}  ({auto_thresh/total*100:.2f}% of mesh)")
    if abs(best - auto_thresh) / max(best, auto_thresh) < 0.5:
        print(f"  ✓ Auto eşik optimal aralıkta.")
    else:
        print(f"  → --island-pct {best/total*100:.1f} ile manuel override önerilebilir.")

    print()
    return best


if __name__ == "__main__":
    optimal = main()
    sys.exit(0)
