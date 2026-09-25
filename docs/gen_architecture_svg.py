#!/usr/bin/env python3
"""Generate the README architecture figure.

One geometry, two palettes: GitHub picks a file with <picture> +
prefers-color-scheme, and an SVG loaded through <img> cannot see the page's
theme, so the light and dark variants have to exist as separate files.

    python docs/gen_architecture_svg.py
"""
import html
from pathlib import Path

W, H = 1040, 620

SANS = ("ui-sans-serif,-apple-system,'Segoe UI','Noto Sans',Helvetica,Arial,"
        "sans-serif")
MONO = "ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace"

LIGHT = dict(
    panel="#ffffff", panel_edge="#e4eaf3",
    fill="#f9fbfe", fill2="#eff4fb", edge="#d6e0ee",
    title="#0e1d31", sub="#5b6b83",
    ref="#2563eb", mic="#0d9488", mask="#7c3aed", out="#c2410c",
    wire="#9aa9be", node="#ffffff", shadow="0.07",
)
DARK = dict(
    panel="#0f1620", panel_edge="#232e40",
    fill="#182231", fill2="#1e293c", edge="#2c3a52",
    title="#e9eff9", sub="#93a5bf",
    ref="#60a5fa", mic="#2dd4bf", mask="#a78bfa", out="#fb923c",
    wire="#4e5f7a", node="#111a26", shadow="0.45",
)

# ---------------------------------------------------------------- geometry
ROW_A_CY, BOX_H = 88, 76
RAIL_X, MUL_CY = 950, 470
NODE_R = 26
CHIP_CY, CHIP_H = 470, 72
PILL_H, PILL_W = 40, 76
FOUT_CY = 560

# the reference chain: acquire, track, align
B1 = (118, 340, "ref")
B2 = (368, 590, "ref")
B3 = (640, 880, "ref")
# the mask chain: features, recurrence, mask
C1 = (330, 510, "mask")
C2 = (540, 680, "mask")
C3 = (710, 890, "mask")


def esc(s):
    return html.escape(s, quote=True)


def txt(x, y, s, size, fill, anchor="middle", weight="400", mono=False,
        opacity=None):
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{MONO if mono else SANS}"{o}>{esc(s)}</text>')


def rpath(pts, r=13):
    """Orthogonal polyline with rounded corners."""
    d = f"M {pts[0][0]:g} {pts[0][1]:g}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        v = (x1 - x0, y1 - y0)
        l1 = max(abs(v[0]) + abs(v[1]), 1e-9)
        u1 = (v[0] / l1, v[1] / l1)
        w = (x2 - x1, y2 - y1)
        l2 = max(abs(w[0]) + abs(w[1]), 1e-9)
        u2 = (w[0] / l2, w[1] / l2)
        rr = min(r, l1 / 2, l2 / 2)
        d += (f" L {x1 - u1[0] * rr:.1f} {y1 - u1[1] * rr:.1f}"
              f" Q {x1:g} {y1:g} {x1 + u2[0] * rr:.1f} {y1 + u2[1] * rr:.1f}")
    d += f" L {pts[-1][0]:g} {pts[-1][1]:g}"
    return d


def wire(pts, color, marker=None, flow=False, width=1.8, dash=None):
    m = f' marker-end="url(#{marker})"' if marker else ""
    common = (f'd="{rpath(pts)}" fill="none" stroke="{color}" '
              f'stroke-width="{width:g}" stroke-linejoin="round" '
              f'stroke-linecap="round"')
    if flow:
        # A dim continuous wire with a bright dash train travelling along it.
        # The dashes alone are only a quarter duty cycle, which reads as a
        # broken connection however nicely they move.
        return (f'<path {common} opacity="0.34"/>'
                f'<path {common} class="flow"{m}/>')
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path {common}{da}{m}/>'


def card(x0, x1, cy, h, accent, p):
    y = cy - h // 2
    return (
        f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" rx="14" '
        f'fill="url(#card)" stroke="{p["edge"]}" stroke-width="1.4"/>'
        f'<line x1="{x0 + 11}" y1="{y + 17}" x2="{x0 + 11}" y2="{y + h - 17}" '
        f'stroke="{p[accent]}" stroke-width="3" stroke-linecap="round" '
        f'opacity="0.9"/>')


def card_text(x0, x1, cy, title, sub=None, sub2=None, p=None, mono=False):
    cx = (x0 + x1) / 2 + 5
    out = [txt(cx, cy - 8, title, 15.5, p["title"], weight="600",
               mono=mono)]
    if sub:
        out.append(txt(cx, cy + 11, sub, 10.5, p["sub"], mono=True))
    if sub2:
        out.append(txt(cx, cy + 26, sub2, 10.5, p["sub"], mono=True))
    return "".join(out)


def pill(cx, cy, label, accent, p):
    return (
        f'<rect x="{cx - PILL_W / 2}" y="{cy - PILL_H / 2}" width="{PILL_W}" '
        f'height="{PILL_H}" rx="20" fill="url(#pill)" stroke="{p[accent]}" '
        f'stroke-width="1.8"/>'
        + txt(cx, cy + 4.5, label, 13, p[accent], weight="600", mono=True))


def node(cx, cy, glyph, accent, p):
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{NODE_R}" fill="{p["node"]}" '
        f'stroke="{p[accent]}" stroke-width="2.2"/>'
        + txt(cx, cy + 7, glyph, 20, p[accent], weight="600"))


def build(name):
    p = dict(LIGHT if name == "light" else DARK)
    s = []
    add = s.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="ali_jaec architecture">')
    add('<style>'
        '.flow{stroke-dasharray:4 12;animation:flow 1.5s linear infinite}'
        '@keyframes flow{from{stroke-dashoffset:0}to{stroke-dashoffset:-16}}'
        '@media (prefers-reduced-motion:reduce){.flow{animation:none;'
        'stroke-dasharray:none}}'
        '</style>')

    # gradients + arrowheads
    add('<defs>')
    add(f'<linearGradient id="card" x1="0" y1="0" x2="0.4" y2="1">'
        f'<stop offset="0" stop-color="{p["fill"]}"/>'
        f'<stop offset="1" stop-color="{p["fill2"]}"/></linearGradient>')
    add(f'<linearGradient id="pill" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{p["fill"]}"/>'
        f'<stop offset="1" stop-color="{p["fill2"]}"/></linearGradient>')
    for key in ("ref", "mic", "mask", "out", "wire"):
        add(f'<marker id="a-{key}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{p[key]}"/></marker>')
    add(f'<filter id="soft" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="3.2" '
        f'flood-color="#0b1a2e" flood-opacity="{p["shadow"]}"/></filter>')
    add('</defs>')

    add(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="18" '
        f'fill="{p["panel"]}" stroke="{p["panel_edge"]}"/>')

    # ---- row A: the reference chain -- steering only, no signal to the output
    add(f'<g filter="url(#soft)">')
    for x0, x1, acc in (B1, B2, B3):
        add(card(x0, x1, ROW_A_CY, BOX_H, acc, p))
    add('</g>')
    add(card_text(B1[0], B1[1], ROW_A_CY, "Delay estimator",
                  "GCC-PHAT + soft-argmax", "runs once, first 1 s", p))
    add(card_text(B2[0], B2[1], ROW_A_CY, "Per-frame tracker",
                  "±100 samples per 10 ms", "no parameters", p))
    add(card_text(B3[0], B3[1], ROW_A_CY, "Align",
                  "integer shift per frame", "x(n) → x_τ(n)", p))

    add(wire([(96, ROW_A_CY), (B1[0], ROW_A_CY)], p["wire"], "a-wire"))
    add(wire([(B1[1], ROW_A_CY), (B2[0], ROW_A_CY)], p["ref"], "a-ref", True))
    add(wire([(B2[1], ROW_A_CY), (B3[0], ROW_A_CY)], p["ref"], "a-ref", True))
    add(txt((B1[1] + B2[0]) / 2, ROW_A_CY - 9, "τ₀", 11.5, p["sub"], mono=True))
    add(txt((B2[1] + B3[0]) / 2, ROW_A_CY - 9, "τ(n)", 11, p["sub"], mono=True))

    # x_tau drops from the align card into the feature row. The reference
    # STEERS the mask; it never reaches the output itself.
    feat_in_x = (C1[0] + C1[1]) / 2 - 70
    add(wire([(B3[1] - 60, ROW_A_CY + BOX_H // 2),
              (B3[1] - 60, 300), (feat_in_x, 300),
              (feat_in_x, CHIP_CY - CHIP_H // 2)], p["ref"], "a-ref", True))
    add(txt(B3[1] - 52, 292, "x_τ(n)", 11, p["sub"], anchor="start", mono=True))

    # ---- inputs
    add(pill(58, ROW_A_CY, "x(n)", "ref", p))
    add(txt(58, ROW_A_CY + 34, "far-end ref", 9.5, p["sub"], mono=True))
    add(f'<g filter="url(#soft)">')
    add(pill(58, MUL_CY, "d(n)", "mic", p))
    add('</g>')
    add(txt(58, MUL_CY + 34, "near-end mic", 9.5, p["sub"], mono=True))

    # the microphone rail: straight into the multiplier, and tapped for the
    # feature row on the way
    add(wire([(96, MUL_CY), (RAIL_X - NODE_R, MUL_CY)], p["mic"], "a-mic",
             True))
    tap_x = (C1[0] + C1[1]) / 2 + 70
    add(wire([(tap_x, MUL_CY), (tap_x, CHIP_CY + CHIP_H // 2)], p["mic"],
             "a-mic", True))
    add(f'<circle cx="{tap_x}" cy="{MUL_CY}" r="4" fill="{p["mic"]}"/>')

    # ---- row C: whitening -> recurrence -> mask
    add(f'<g filter="url(#soft)">')
    for x0, x1, acc in (C1, C2, C3):
        add(card(x0, x1, CHIP_CY, CHIP_H, acc, p))
    add('</g>')
    add(card_text(C1[0], C1[1], CHIP_CY, "Whitening + bands",
                  "shared per-bin weight", "16 bands of d, x_τ, d·x_τ", p))
    add(card_text(C2[0], C2[1], CHIP_CY, "GRU", "64 → 96", "per frame", p))
    add(card_text(C3[0], C3[1], CHIP_CY, "Spectral mask", "16 bands → 257 bins",
                  "→ 1 where ref quiet", p))
    add(wire([(C1[1], CHIP_CY), (C2[0], CHIP_CY)], p["mask"], "a-mask", True))
    add(wire([(C2[1], CHIP_CY), (C3[0], CHIP_CY)], p["mask"], "a-mask", True))
    add(wire([(C3[1], CHIP_CY), (RAIL_X - NODE_R - 40, CHIP_CY),
              (RAIL_X - NODE_R - 40, MUL_CY + NODE_R + 8)], p["mask"],
             "a-mask", True))
    add(txt((C3[1] + RAIL_X) / 2 - 30, CHIP_CY - 11, "mask", 11, p["sub"],
            mono=True))

    # ---- the multiplier and the output
    add(node(RAIL_X, MUL_CY, "×", "mask", p))
    add(wire([(RAIL_X, MUL_CY + NODE_R), (RAIL_X, FOUT_CY - PILL_H // 2)],
             p["out"], "a-out", True))
    add(f'<g filter="url(#soft)">')
    add(pill(RAIL_X, FOUT_CY, "e(n)", "out", p))
    add('</g>')
    add(txt(RAIL_X, FOUT_CY + 34, "mask · d(n)  (then per-bin weight + OLA)",
            9.5, p["sub"], mono=True))

    add('</svg>')
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    for name in ("light", "dark"):
        f = out / f"architecture-{name}.svg"
        f.write_text(build(name))
        print("%s  %d bytes" % (f, f.stat().st_size))
