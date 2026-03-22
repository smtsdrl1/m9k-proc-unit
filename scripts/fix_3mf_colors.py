#!/usr/bin/env python3
"""
fix_3mf_colors.py — Bambu Studio 3MF Renk Düzeltici
=====================================================

Bambu "Image to 3D" çıktısındaki parçalı renk bölgelerini temizler.
Mesh yüzlerini geometri + normal vektörlerine göre büyük bölgelere gruplar,
her bölgeye filament rengi atar ve yazıcıya gönderilmeye hazır temiz bir
3MF dosyası üretir.

Kullanım:
    python fix_3mf_colors.py input.3mf -o output.3mf -n 4
    python fix_3mf_colors.py input.3mf  # output: input_fixed.3mf

Seçenekler:
    -o, --output    Çıktı dosyası (varsayılan: <input>_fixed.3mf)
    -n, --colors    Filament sayısı: 2, 3 veya 4 (varsayılan: 4)
    --debug         Her bölgenin yüz sayısını yazdır
"""

import argparse
import base64
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# ─── 3MF namespace ────────────────────────────────────────────────────────────
NS_3MF  = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_SLIC = "http://schemas.slic3r.org/3mf/2017/06"
NS_P    = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

ET.register_namespace("",        NS_3MF)
ET.register_namespace("slic3rpe", NS_SLIC)
ET.register_namespace("p",       NS_P)

TAG = lambda ns, tag: f"{{{ns}}}{tag}"

# ─── Bambu per-face renk kodlaması (2-bit per face, base64) ───────────────────
def encode_bambu_face_colors(color_ids: np.ndarray) -> str:
    """
    Her yüz için 0-3 arası renk id'si alır, Bambu Studio'nun
    painting_color_per_facets formatına dönüştürür.

    Format: her yüz = 3 bit (color 0-7), ardışık bitler, base64 kodlu.
    Bambu kendi içinde 'triangle painting' için bu formatı kullanır.
    """
    n = len(color_ids)
    # 3 bit per face, packed big-endian
    total_bits = n * 3
    total_bytes = (total_bits + 7) // 8
    buf = bytearray(total_bytes)

    for i, cid in enumerate(color_ids):
        bit_pos = i * 3
        byte_idx = bit_pos // 8
        bit_off  = bit_pos % 8
        val = int(cid) & 0x07
        # write 3 bits
        for b in range(3):
            if val & (1 << (2 - b)):
                buf[byte_idx + (bit_off + b) // 8] |= (1 << (7 - (bit_off + b) % 8))

    return base64.b64encode(bytes(buf)).decode("ascii")


def decode_bambu_face_colors(encoded: str, n_faces: int) -> np.ndarray:
    """Encode'un tersine çevirir — debug için."""
    buf = base64.b64decode(encoded)
    result = np.zeros(n_faces, dtype=np.uint8)
    for i in range(n_faces):
        bit_pos = i * 3
        val = 0
        for b in range(3):
            byte_idx = (bit_pos + b) // 8
            bit_off  = 7 - (bit_pos + b) % 8
            if byte_idx < len(buf) and (buf[byte_idx] >> bit_off) & 1:
                val |= (1 << (2 - b))
        result[i] = val
    return result


# ─── Mesh yükleme ─────────────────────────────────────────────────────────────
def load_mesh_from_3mf(zf: zipfile.ZipFile):
    """
    3MF içindeki ilk mesh'i döndürür:
        vertices  : (V, 3) float32
        triangles : (T, 3) int32
        model_xml : ham XML string
        model_path: 3MF içindeki path
    """
    # Relationships ile model dosyasını bul
    model_path = "3D/3dmodel.model"
    for name in zf.namelist():
        if name.endswith(".model"):
            model_path = name
            break

    xml_bytes = zf.read(model_path)
    root = ET.fromstring(xml_bytes)

    # İlk mesh objesini al
    mesh_el = root.find(f".//{TAG(NS_3MF,'mesh')}")
    if mesh_el is None:
        raise ValueError("3MF içinde mesh bulunamadı.")

    verts_el = mesh_el.find(TAG(NS_3MF, "vertices"))
    tris_el  = mesh_el.find(TAG(NS_3MF, "triangles"))

    vertices  = []
    triangles = []

    for v in verts_el.findall(TAG(NS_3MF, "vertex")):
        vertices.append([float(v.get("x")), float(v.get("y")), float(v.get("z"))])

    for t in tris_el.findall(TAG(NS_3MF, "triangle")):
        triangles.append([int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))])

    return (
        np.array(vertices,  dtype=np.float32),
        np.array(triangles, dtype=np.int32),
        xml_bytes,
        model_path,
    )


# ─── Yüz geometrisi ───────────────────────────────────────────────────────────
def compute_face_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
    return (normals / lengths).astype(np.float32)


def compute_face_centroids(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    return ((verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3.0).astype(np.float32)


# ─── Renk ataması ─────────────────────────────────────────────────────────────
def assign_colors_geometric(
    verts: np.ndarray,
    tris:  np.ndarray,
    n_colors: int = 4,
    debug: bool = False,
) -> np.ndarray:
    """
    Geometrik kurallara + K-Means'e dayalı renk ataması.

    Humanoid figür varsayımı (Image-to-3D genellikle ayakta figür üretir):
        Renk 1 (filament 1): Kafatası yüzü / ten rengi bölgesi (front-facing, üst)
        Renk 2 (filament 2): Bere / kask / şapka (en üst bölge)
        Renk 3 (filament 3): Gövde / ceket (orta bölge)
        Renk 4 (filament 4): Pantolon + ayakkabılar (alt bölge)
    """
    normals   = compute_face_normals(verts, tris)
    centroids = compute_face_centroids(verts, tris)

    # Normalize height: 0=en alt, 1=en üst
    y = centroids[:, 1]  # Bambu "Y" ekseni yukarı
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    # Z-normal: öne bakan yüzler (kafatası yüzü)
    nz = normals[:, 2]   # +Z = öne bakan

    if n_colors == 2:
        # Basit: üst yarı vs alt yarı
        color_ids = np.where(y_norm > 0.5, 0, 1).astype(np.uint8)

    elif n_colors == 3:
        color_ids = np.zeros(len(tris), dtype=np.uint8)
        color_ids[y_norm > 0.75] = 0   # baş
        color_ids[(y_norm > 0.35) & (y_norm <= 0.75)] = 1  # gövde
        color_ids[y_norm <= 0.35] = 2  # alt

    else:  # 4 renk — ana kullanım durumu
        # Özellik vektörü: mesh'e özgü, yüklenen dosyaya göre ölçeklenir
        cx_norm = centroids[:, 0] / (abs(centroids[:, 0]).max() + 1e-8)
        features = np.column_stack([
            y_norm * 2.0,    # dikey konum (ağırlıklı)
            nz    * 0.8,     # öne bakış (kafatası yüzü ayrımı)
            normals[:, 0] * 0.4,   # yan bakış
            cx_norm * 0.3,         # sol/sağ simetri
        ])

        # Kural tabanlı ön atama (geniş, net bölgeler)
        mask_hat  = y_norm > 0.80
        mask_face = (y_norm > 0.55) & (y_norm <= 0.80) & (nz > 0.2)
        mask_body = (y_norm > 0.30) & ~(mask_hat | mask_face)
        mask_legs = y_norm <= 0.30

        color_ids = np.zeros(len(tris), dtype=np.uint8)
        color_ids[mask_hat]  = 1
        color_ids[mask_face] = 0
        color_ids[mask_body] = 2
        color_ids[mask_legs] = 3

        # K-Means ile SINIR bölgelerini iyileştir.
        # Sabit eşikler (%80, %55, %30) her modelin şekline tam uymaz;
        # ±%07 bandındaki "geçiş yüzleri" makine öğrenimiyle yeniden atanır.
        BAND = 0.07
        boundary = (
            (np.abs(y_norm - 0.80) < BAND) |
            (np.abs(y_norm - 0.55) < BAND) |
            (np.abs(y_norm - 0.30) < BAND)
        )
        n_boundary = boundary.sum()
        if n_boundary > 20:
            km = KMeans(n_clusters=4, random_state=42, n_init=5)
            km.fit(features[~boundary])            # merkezi sabit yüzlerle eğit
            color_ids[boundary] = km.predict(features[boundary]).astype(np.uint8)
            if debug:
                print(f"  K-Means geçiş bandı: {n_boundary:,} yüz yeniden atandı")

    if debug:
        for c in range(n_colors):
            count = (color_ids == c).sum()
            print(f"  Filament {c+1}: {count:,} yüz ({count/len(tris)*100:.1f}%)")

    return color_ids


# ─── Bağlantılı bileşen birleştirme (küçük parçaları komşuya ata) ─────────────
def merge_small_islands(
    tris: np.ndarray,
    color_ids: np.ndarray,
    min_faces: int = 200,
) -> np.ndarray:
    """
    Aynı renkteki bağlantısız küçük adacıkları (< min_faces yüz) komşu
    rengine atar — bu sayede binlerce küçük parça birleşir.
    """
    n_faces = len(tris)

    # Kenar komşuluğu: her kenar paylaşan iki üçgen komşudur
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

    # Aynı renkteki bağlantılı bileşenleri bul (BFS)
    visited   = np.zeros(n_faces, dtype=bool)
    new_color = color_ids.copy()

    for start in range(n_faces):
        if visited[start]:
            continue
        cid = color_ids[start]
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            fi = queue.pop()
            component.append(fi)
            for nb in adj[fi]:
                if not visited[nb] and color_ids[nb] == cid:
                    visited[nb] = True
                    queue.append(nb)

        if len(component) < min_faces:
            # Komşu renkleri say ve en yaygın olana ata
            neighbor_colors = []
            for fi in component:
                for nb in adj[fi]:
                    if color_ids[nb] != cid:
                        neighbor_colors.append(color_ids[nb])
            if neighbor_colors:
                dominant = max(set(neighbor_colors), key=neighbor_colors.count)
                for fi in component:
                    new_color[fi] = dominant

    return new_color


# ─── Bambu config yazma ───────────────────────────────────────────────────────
def write_bambu_model_settings(
    color_ids: np.ndarray,
    object_id: str = "1",
) -> str:
    """painting_color_per_facets içeren model_settings.config döndürür."""
    encoded = encode_bambu_face_colors(color_ids)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="{object_id}">
    <metadata key="painting_color_per_facets" value="{encoded}"/>
    <metadata key="extruder" value="1"/>
  </object>
</config>
"""
    return xml


# ─── Ana işlev ────────────────────────────────────────────────────────────────
def fix_3mf_colors(
    input_path:   str,
    output_path:  str,
    n_colors:     int   = 4,
    min_island:   int   = 0,    # 0 = otomatik (toplam yüzün %1'i)
    island_pct:   float = 1.0,  # min_island=0 iken kullanılan yüzde
    debug:        bool  = False,
) -> None:
    print(f"[+] Yükleniyor: {input_path}")

    with zipfile.ZipFile(input_path, "r") as zf:
        verts, tris, xml_bytes, model_path = load_mesh_from_3mf(zf)
        all_names = zf.namelist()

    n_faces = len(tris)
    print(f"    Vertex: {len(verts):,}  |  Üçgen: {n_faces:,}")

    # min_island otomatik hesapla: meshın boyutuna oranla
    if min_island == 0:
        min_island = max(50, int(n_faces * island_pct / 100))
    print(f"    min_island = {min_island:,} yüz  ({min_island/n_faces*100:.2f}% of mesh)")

    # Renk ata
    print(f"[+] {n_colors} renk için geometrik analiz yapılıyor...")
    color_ids = assign_colors_geometric(verts, tris, n_colors=n_colors, debug=debug)

    # Küçük adacıkları birleştir
    print(f"[+] Küçük parçalar birleştiriliyor (< {min_island:,} yüz)...")
    color_ids = merge_small_islands(tris, color_ids, min_faces=min_island)

    if debug:
        print("[+] Birleştirme sonrası:")
        for c in range(n_colors):
            count = (color_ids == c).sum()
            print(f"    Filament {c+1}: {count:,} yüz ({count/len(tris)*100:.1f}%)")

    # Bambu settings config üret
    # Object ID'yi XML'den çek
    root = ET.fromstring(xml_bytes)
    obj_el = root.find(f".//{TAG(NS_3MF,'object')}")
    obj_id = obj_el.get("id", "1") if obj_el is not None else "1"
    settings_xml = write_bambu_model_settings(color_ids, object_id=obj_id)

    # Yeni 3MF yaz
    print(f"[+] Yazılıyor: {output_path}")
    with zipfile.ZipFile(input_path, "r") as zin, \
         zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:

        skip = {"Metadata/model_settings.config"}
        for item in zin.namelist():
            if item not in skip:
                zout.writestr(item, zin.read(item))

        # Güncellenmiş settings
        zout.writestr("Metadata/model_settings.config", settings_xml)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Tamamlandı! ({size_kb:.0f} KB) → {output_path}")
    print()
    print("Sonraki adımlar:")
    print("  1. Bambu Studio'yu aç")
    print("  2. Dosyayı sürükleyip bırak")
    print(f"  3. Sağ tıkla → 'Change Filament' ile {n_colors} filament ata")
    print("  4. Renk sınırlarını 'Color Painting' panelinden ince ayar yap")
    print("  5. Slice & Print!")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bambu 3MF dosyasındaki parçalı renkleri düzelt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="Giriş .3mf dosyası")
    parser.add_argument("-o", "--output", help="Çıkış .3mf dosyası")
    parser.add_argument("-n", "--colors", type=int, default=4,
                        choices=[2, 3, 4],
                        help="Filament sayısı (varsayılan: 4)")
    parser.add_argument("--min-island", type=int, default=0,
                        help="Adacık eşiği (yüz sayısı). 0=otomatik: mesh boyutunun %%1'i (varsayılan: 0)")
    parser.add_argument("--island-pct", type=float, default=1.0,
                        help="--min-island=0 iken kullanılan yüzde (varsayılan: 1.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Bölge istatistiklerini yazdır")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Hata: '{args.input}' bulunamadı.", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).parent / f"{stem}_fixed.3mf")

    fix_3mf_colors(
        input_path  = args.input,
        output_path = args.output,
        n_colors    = args.colors,
        min_island  = args.min_island,
        island_pct  = args.island_pct,
        debug       = args.debug,
    )


if __name__ == "__main__":
    main()
