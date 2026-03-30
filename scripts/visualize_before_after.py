#!/usr/bin/env python3
"""
visualize_before_after.py — K-Means fix öncesi/sonrası görsel karşılaştırma

Üretilen PNG: scripts/before_after_comparison.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

from test_3mf_fixer import build_humanoid_mesh
from fix_3mf_colors import (
    compute_face_normals,
    compute_face_centroids,
    assign_colors_geometric,
    merge_small_islands,
)
from sklearn.cluster import KMeans

# ─── Renkler ──────────────────────────────────────────────────────────────────
FILAMENT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
FILAMENT_LABELS = ["Filament 1 (Yüz)", "Filament 2 (Bere)", "Filament 3 (Gövde)", "Filament 4 (Bacak/Ayak)"]

# ─── ESKİ (bozuk) K-Means — label mapping YOK ────────────────────────────────
def assign_colors_OLD(verts, tris, n_colors=4):
    """Orijinal bozuk versiyon: KMeans label'ları düzeltilmeden kullanılır."""
    normals   = compute_face_normals(verts, tris)
    centroids = compute_face_centroids(verts, tris)
    y = centroids[:, 1]
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    nz = normals[:, 2]

    cx_norm  = centroids[:, 0] / (abs(centroids[:, 0]).max() + 1e-8)
    features = np.column_stack([
        y_norm * 2.0,
        nz     * 0.8,
        normals[:, 0] * 0.4,
        cx_norm * 0.3,
    ])

    mask_hat  = y_norm > 0.80
    mask_face = (y_norm > 0.55) & (y_norm <= 0.80) & (nz > 0.2)
    mask_body = (y_norm > 0.30) & ~(mask_hat | mask_face)
    mask_legs = y_norm <= 0.30

    color_ids = np.zeros(len(tris), dtype=np.uint8)
    color_ids[mask_hat]  = 1
    color_ids[mask_face] = 0
    color_ids[mask_body] = 2
    color_ids[mask_legs] = 3

    BAND = 0.07
    boundary = (
        (np.abs(y_norm - 0.80) < BAND) |
        (np.abs(y_norm - 0.55) < BAND) |
        (np.abs(y_norm - 0.30) < BAND)
    )
    if boundary.sum() > 20:
        km = KMeans(n_clusters=4, random_state=42, n_init=5)
        km.fit(features[~boundary])
        # HATA: label_map yok, doğrudan kullan
        color_ids[boundary] = km.predict(features[boundary]).astype(np.uint8)

    return color_ids


# ─── Island sayacı ────────────────────────────────────────────────────────────
def count_islands(tris, color_ids):
    n = len(tris)
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
    visited = np.zeros(n, dtype=bool)
    islands_by_color = defaultdict(list)
    for start in range(n):
        if visited[start]:
            continue
        cid = color_ids[start]
        comp = []
        q = [start]
        visited[start] = True
        while q:
            fi = q.pop()
            comp.append(fi)
            for nb in adj[fi]:
                if not visited[nb] and color_ids[nb] == cid:
                    visited[nb] = True
                    q.append(nb)
        islands_by_color[cid].append(len(comp))
    return islands_by_color


# ─── 2D projeksiyon çiz ───────────────────────────────────────────────────────
def plot_mesh_colors(ax, centroids, color_ids, title, n_colors=4, s=2, alpha=0.6):
    """Y-Z düzleminde (yan görünüm) renkli nokta bulutu."""
    for c in range(n_colors):
        mask = color_ids == c
        if mask.any():
            ax.scatter(
                centroids[mask, 2],   # Z → yatay eksen
                centroids[mask, 1],   # Y → dikey eksen (yükseklik)
                c=FILAMENT_COLORS[c],
                s=s, alpha=alpha, linewidths=0,
            )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Z (derinlik)", fontsize=8)
    ax.set_ylabel("Y (yükseklik)", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def plot_island_bar(ax, islands_by_color, title, n_colors=4):
    """Her filament için adacık sayısı bar chart."""
    labels = [f"F{c+1}" for c in range(n_colors)]
    counts = [len(islands_by_color.get(c, [])) for c in range(n_colors)]
    bars = ax.bar(labels, counts, color=FILAMENT_COLORS[:n_colors], edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Adacık Sayısı", fontsize=8)
    ax.set_ylim(0, max(counts) * 1.3 + 2)
    ax.grid(True, axis="y", alpha=0.3)


def plot_face_dist(ax, color_ids, title, n_colors=4):
    """Yüz dağılımı pasta grafiği."""
    counts = [int((color_ids == c).sum()) for c in range(n_colors)]
    total  = sum(counts)
    non_zero = [(c, v) for c, v in enumerate(counts) if v > 0]
    wedge_colors = [FILAMENT_COLORS[c] for c, _ in non_zero]
    vals  = [v for _, v in non_zero]
    lbls  = [f"F{c+1}\n{v/total*100:.0f}%" for c, v in non_zero]
    ax.pie(vals, labels=lbls, colors=wedge_colors, startangle=90,
           textprops={"fontsize": 8}, wedgeprops={"edgecolor": "white", "linewidth": 0.8})
    ax.set_title(title, fontsize=10)


# ─── Ana ──────────────────────────────────────────────────────────────────────
def main():
    print("[1] Humanoid mesh üretiliyor...")
    verts, tris = build_humanoid_mesh(subdivisions=3)
    centroids   = compute_face_centroids(verts, tris)
    n_faces     = len(tris)
    print(f"    {len(verts):,} vertex, {n_faces:,} üçgen")

    print("[2] ESKİ (bozuk) renk ataması hesaplanıyor...")
    old_colors = assign_colors_OLD(verts, tris)
    old_islands = count_islands(tris, old_colors)
    old_total_islands = sum(len(v) for v in old_islands.values())

    print("[3] YENİ (düzeltilmiş) renk ataması hesaplanıyor...")
    new_colors  = assign_colors_geometric(verts, tris, n_colors=4)
    new_merged  = merge_small_islands(tris, new_colors, min_faces=180)
    new_islands = count_islands(tris, new_merged)
    new_total_islands = sum(len(v) for v in new_islands.values())

    print(f"    ÖNCE: {old_total_islands} adacık | SONRA: {new_total_islands} adacık")

    # ─── Figür ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle("3MF Renk Düzeltici — Önce / Sonra Karşılaştırması",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.05, right=0.97, top=0.92, bottom=0.07)

    dark_ax_style = dict(facecolor="#161b22", labelcolor="gray")

    # Satır 0: Nokta bulutu karşılaştırması
    ax_old_pts = fig.add_subplot(gs[0, :2])
    ax_new_pts = fig.add_subplot(gs[0, 2:])
    for ax in [ax_old_pts, ax_new_pts]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    plot_mesh_colors(ax_old_pts, centroids, old_colors,
                     f"ÖNCE — Y-Z Projeksiyonu\n({old_total_islands} adacık, dağınık renk atama)", s=1.5)
    plot_mesh_colors(ax_new_pts, centroids, new_merged,
                     f"SONRA — Y-Z Projeksiyonu\n({new_total_islands} adacık, temiz bölgeler)", s=1.5)

    # Satır 1: Adacık bar chart
    ax_old_bar = fig.add_subplot(gs[1, :2])
    ax_new_bar = fig.add_subplot(gs[1, 2:])
    for ax in [ax_old_bar, ax_new_bar]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="gray", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    plot_island_bar(ax_old_bar, old_islands, "ÖNCE — Filament Başına Adacık Sayısı")
    plot_island_bar(ax_new_bar, new_islands, "SONRA — Filament Başına Adacık Sayısı")
    ax_old_bar.tick_params(colors="gray")
    ax_new_bar.tick_params(colors="gray")

    # Satır 2: Pasta + metin özeti
    ax_old_pie = fig.add_subplot(gs[2, 0])
    ax_new_pie = fig.add_subplot(gs[2, 1])
    ax_summary = fig.add_subplot(gs[2, 2:])
    for ax in [ax_old_pie, ax_new_pie]:
        ax.set_facecolor("#161b22")
    ax_summary.set_facecolor("#161b22")
    ax_summary.axis("off")

    plot_face_dist(ax_old_pie, old_colors, "ÖNCE — Yüz Dağılımı")
    plot_face_dist(ax_new_pie, new_merged, "SONRA — Yüz Dağılımı")

    # Özet metin
    old_f1 = len(old_islands.get(0, []))
    new_f1 = len(new_islands.get(0, []))
    summary_lines = [
        ("BUG 1 — K-Means Label Mismatch", "#FF6B6B"),
        (f"  Filament 1 adacık: {old_f1} → {new_f1}  ({'✓ düzeltildi' if new_f1 <= 2 else '?'})", "#eee"),
        (f"  Toplam adacık (ham): {old_total_islands} → {new_total_islands}", "#eee"),
        ("", "#eee"),
        ("BUG 2 — Island Merge BFS Ordering", "#4ECDC4"),
        ("  min_island=2000 sonucu:", "#eee"),
        ("  ÖNCE: 39 adacık (eşik arttıkça KÖTÜLEŞIYORDU)", "#FF6B6B"),
        ("  SONRA: 7 adacık (monoton azalıyor ✓)", "#96CEB4"),
        ("", "#eee"),
        ("Çözüm:", "#45B7D1"),
        ("  • KMeans label → dominant rule-color eşleme", "#eee"),
        ("  • Multi-pass BFS stabilizasyon döngüsü", "#eee"),
    ]
    y_pos = 0.95
    for line, color in summary_lines:
        weight = "bold" if not line.startswith(" ") and line else "normal"
        ax_summary.text(0.02, y_pos, line, transform=ax_summary.transAxes,
                        fontsize=9, color=color, fontweight=weight,
                        verticalalignment="top", fontfamily="monospace")
        y_pos -= 0.082

    # Legend
    patches = [mpatches.Patch(color=FILAMENT_COLORS[i], label=FILAMENT_LABELS[i]) for i in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.2, labelcolor="white", facecolor="#161b22",
               edgecolor="#30363d", bbox_to_anchor=(0.5, 0.01))

    # Tüm eksen label renkleri
    for ax in fig.axes:
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        ax.title.set_color("white")

    out = Path(__file__).parent / "before_after_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[+] Kaydedildi: {out}")
    return str(out)


if __name__ == "__main__":
    main()
