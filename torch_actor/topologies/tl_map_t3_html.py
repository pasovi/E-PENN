"""Interactive (rotatable) 3D map of the T^3 transfer-learning outcome: one point per new IC, red =
captured, green = escaped, black ring = capturable but escaped, blue ring = uncapturable but captured;
black squares = the pursuer start and its periodic images; translucent spheres of radius R around them
= the capturable region.  Hover gives the IC, its geodesic distance, V*, T_c and the chased image.
Self-contained HTML (plotly.js embedded).

Usage: python3 torch_actor/topologies/tl_map_t3_html.py [results/T3_tl.json] [figs/T3_tl_map.html]
"""
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))


def main(src=os.path.join(HERE, "results", "T3_tl.json"), out=os.path.join(HERE, "figs", "T3_tl_map.html")):
    r = json.load(open(src)); g = r["game"]; L = r["config"]["L"]; R = g["reach"]
    x0 = np.array(r["x0_new"]); cap = np.array(g["capturable"]); capt = np.array(g["captured"])
    d0 = np.array(g["d0"]); Vs = np.array(g["Vstar"]); Tc = np.array([np.nan if t is None else t for t in g["Tc"]])
    chased = [tuple(int(-c) for c in ch) for ch in g["chart_closest"]]
    hover = [f"IC ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})<br>|Δ0| = {d:.3f}  (R = {R:.2f})<br>V* = {v:.2f}<br>"
             f"{'captured, T_c = %.2f' % t if np.isfinite(t) else 'escapes'}<br>pursuer image chased: {c}"
             for x, d, v, t, c in zip(x0, d0, Vs, Tc, chased)]
    fig = go.Figure()
    for mask, name, color, size, op in ((capt, f"captured ({int(capt.sum())})", "#d62728", 4, 0.9),
                                        (~capt, f"escaped ({int((~capt).sum())})", "#2ca02c", 3, 0.45)):
        fig.add_trace(go.Scatter3d(x=x0[mask, 0], y=x0[mask, 1], z=x0[mask, 2], mode="markers", name=name,
                                   marker=dict(size=size, color=color, opacity=op), text=[h for h, m in zip(hover, mask) if m], hoverinfo="text"))
    for mask, name, color in ((cap & ~capt, "capturable but escaped", "black"), (~cap & capt, "uncapturable but captured", "blue")):
        if mask.any():
            fig.add_trace(go.Scatter3d(x=x0[mask, 0], y=x0[mask, 1], z=x0[mask, 2], mode="markers", name=f"{name} ({int(mask.sum())})",
                                       marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=color, width=3)),
                                       text=[h for h, m in zip(hover, mask) if m], hoverinfo="text"))
    corners = np.array(np.meshgrid([0, L], [0, L], [0, L])).reshape(3, -1).T
    fig.add_trace(go.Scatter3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], mode="markers", name="pursuer start and images",
                               marker=dict(size=6, color="black", symbol="square"), hoverinfo="name"))
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    sx, sy, sz = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    for i, c in enumerate(corners):
        X, Y, Z = c[0] + R * sx, c[1] + R * sy, c[2] + R * sz
        inside = (X >= 0) & (X <= L) & (Y >= 0) & (Y <= L) & (Z >= 0) & (Z <= L)
        X = np.where(inside, X, np.nan); Y = np.where(inside, Y, np.nan); Z = np.where(inside, Z, np.nan)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.12, colorscale=[[0, "#888"], [1, "#888"]], showscale=False,
                                 name="capturable region (geodesic distance ≤ R)", showlegend=(i == 0), hoverinfo="skip"))
    fig.update_layout(title=f"T³ transfer learning: outcome per new IC ({len(x0)} heads, {int(cap.sum())} capturable; "
                            f"{100 * capt[cap].mean():.1f}% of the capturable captured) — drag to rotate, scroll to zoom",
                      scene=dict(xaxis=dict(range=[0, L], title="x"), yaxis=dict(range=[0, L], title="y"), zaxis=dict(range=[0, L], title="z"),
                                 aspectmode="cube"), legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=50, b=0), height=800)
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
