"""Builds brief/Project_Brief.pdf (2 pages). Re-run after benchmarking to update the results table:

    python brief/make_brief.py
    python brief/make_brief.py --results brief/results.json   # {"rows": [["Map set", "exit %", ...], ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
INK = colors.HexColor("#1d1d1f")
QUIET = colors.HexColor("#5f6368")
ACCENT = colors.HexColor("#1a73e8")
TINT = colors.HexColor("#e8f0fe")
RULE = colors.HexColor("#c4c7c5")

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=12.4, textColor=INK, alignment=TA_LEFT, spaceAfter=3)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=10.8, textColor=QUIET)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=15.5, leading=19, spaceAfter=1)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=9.6, textColor=QUIET, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT, spaceBefore=5, spaceAfter=2)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.6, leading=10.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT)]
    t.setStyle(TableStyle(style))
    return t


def arch_diagram(width):
    """Inputs -> encoders -> (odometry -> memory map) -> fusion/GRU -> heads."""
    H = 200
    d = Drawing(width, H)
    n, gap = 5, 6
    bw = (width - (n - 1) * gap) / n
    xs = [i * (bw + gap) for i in range(n)]
    cx = [x + bw / 2 for x in xs]

    def box(x, y, w, h, title, sub=None, accent=False):
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=TINT if accent else colors.white,
                   strokeColor=ACCENT if accent else RULE, strokeWidth=1.4 if accent else 0.8))
        ty = y + h / 2 + (3 if sub else -3)
        d.add(String(x + w / 2, ty, title, fontName="Helvetica-Bold", fontSize=7.8, fillColor=INK, textAnchor="middle"))
        if sub:
            for k, line in enumerate(sub.split("\n")):
                d.add(String(x + w / 2, ty - 9.5 - 8.5 * k, line, fontName="Helvetica", fontSize=6.8, fillColor=QUIET, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=QUIET, strokeWidth=0.8))
        if y2 < y1:  # downward
            d.add(Polygon([x2 - 2.6, y2 + 4.5, x2 + 2.6, y2 + 4.5, x2, y2], fillColor=QUIET, strokeColor=QUIET))
        elif x2 > x1:
            d.add(Polygon([x2 - 4.5, y2 - 2.6, x2 - 4.5, y2 + 2.6, x2, y2], fillColor=QUIET, strokeColor=QUIET))
        else:
            d.add(Polygon([x2 + 4.5, y2 - 2.6, x2 + 4.5, y2 + 2.6, x2, y2], fillColor=QUIET, strokeColor=QUIET))

    y1, y2, ym, yf, yh = 176, 136, 88, 46, 4
    ins = ["rays 64 x 2", "entities 32 x 5", "vitals 17", "RGB + depth", "last action 9"]
    encs = [("Ray encoder", "circular 1D conv"), ("Entity encoder", "masked self-attention"),
            ("Vitals MLP", "+ stuck, damage timers"), ("RGB-D CNN", "exit switches, signs"), ("Odometry", "Doom turn rule + velocity")]
    for i in range(n):
        box(xs[i], y1, bw, 20, ins[i])
        box(xs[i], y2, bw, 28, *encs[i])
        arrow(cx[i], y1, cx[i], y2 + 28)
    mx, mw = xs[3], bw * 2 + gap
    box(mx, ym, mw, 36, "Ego memory map (USP)", "8 channels: free, wall, door, locked, item,\nkey, enemy, visited  →  32x32 ahead-up crop", accent=True)
    arrow(cx[4], y2, cx[4], ym + 36)
    arrow(cx[3], y2, cx[3], ym + 36)
    fx, fw = xs[1], bw * 3 + 2 * gap
    box(fx, yf, fw, 28, "Fusion → GRU 512", "ray + entity + vitals + RGB-D + map-CNN features")
    for i in (0,):
        d.add(Line(cx[i], y2, cx[i], yf + 14, strokeColor=QUIET, strokeWidth=0.8))
        arrow(cx[i], yf + 14, fx, yf + 14)
    arrow(cx[1], y2, cx[1], yf + 28)
    arrow(cx[2], y2, cx[2], yf + 28)
    d.add(Line(mx + mw / 2, ym, mx + mw / 2, yf + 14, strokeColor=QUIET, strokeWidth=0.8))
    arrow(mx + mw / 2, yf + 14, fx + fw, yf + 14)
    hw = bw * 1.25
    heads = [("Policy head", "9 Bernoulli buttons"), ("Value head", "PPO critic")]
    hx = [fx + fw / 2 - hw - 8, fx + fw / 2 + 8]
    for (t, s), x in zip(heads, hx):
        box(x, yh, hw, 26, t, s)
        arrow(x + hw / 2, yf, x + hw / 2, yh + 26)
    return d


def build(results_rows=None, out=HERE / "Project_Brief.pdf"):
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=12 * mm, title="MapMind - Project Brief",
                            author="Track 1 team")
    W = A4[0] - 30 * mm
    s = []
    s.append(Paragraph("MapMind: path-integrated spatial memory for zero-shot DOOM", H1))
    s.append(Paragraph("Yugaantar DevForge 2026 · Track 1 “Can It Play DOOM?” · 2-page project brief", SUB))

    s.append(Paragraph("1. Unique selling proposition", H2))
    s.append(Paragraph(
        "The wrapper hides position and heading, so most agents rely on a recurrent state that forgets rooms within seconds. "
        "MapMind <b>builds its own map</b>: it dead-reckons its pose from the observations and its own actions, and writes every "
        "step's 64 raycasts and visible entities into a top-down grid. A locked door seen 600 steps ago is still on the map when the "
        "keycard is picked up. All actions still come from a trained network; the map only encodes what was observed.", BODY))
    s += bullets([
        "<b>Exact heading odometry</b> from Doom's turn rule (1.76°/tic for the first 5 held tics, then 3.52°; no turning in the "
        "first 7 tics after spawn), measured against the engine: 0° error while alive.",
        "<b>Hybrid ego-centric model</b>: rays, entity set, vitals and memory map carry navigation; a small RGB-D branch exists only "
        "because the exit is invisible to rays and entities, yet it is the largest scoring item.",
        "<b>Built for the eval box</b>: 3.4 M parameters, 1.6–3.2 ms per decision on one CPU thread (budget 40 ms), crash-proof <i>act()</i>.",
    ])

    s.append(Paragraph("2. Neural architecture", H2))
    s.append(arch_diagram(W))
    s.append(Spacer(1, 3))
    s.append(table([
        ["Module", "Design", "Why"],
        ["Ray encoder", "Hit type embedded (6→8) + distance + contact flag; 3 circular-padded 1D convs; FC 256",
         "Circular padding keeps “behind” adjacent to left/right; translation-equivariant in angle"],
        ["Entity encoder", "Per-entity MLP on (dist, sin, cos, rel. velocity) + type embedding; null token; 1 self-attention block; "
         "masked mean+max pool → 128", "Any number of enemies (0–32); attention lets threats be weighed against each other; "
         "relative velocity exposes incoming projectiles"],
        ["Map CNN", "3 stride-2 convs over the 8×32×32 ahead-up crop → 256", "Spatial memory in the agent's own frame"],
        ["RGB-D CNN", "4 stride-2 convs on 60×80 RGB + depth → 256", "Exit switches/signs; depth for ledges"],
        ["Core + heads", "Fusion 1024→512, GRU 512, 9 independent Bernoulli logits, value head",
         "Short-term memory (USE timing, combat) on top of the long-term map"],
    ], [26 * mm, 84 * mm, W - 110 * mm]))

    s.append(Paragraph("3. Spatial memory design", H2))
    s += bullets([
        "<b>Heading</b> is integrated per tic from our own TURN buttons (the rule above). <b>Position</b>: displacement = 2.6·v<sub>t</sub> + "
        "1.7·v<sub>t-1</sub> (least-squares fit on Freedoom maps, residual ≈2.5 units/step) with v rotated into the map frame.",
        "<b>Wall pushing</b>: Doom keeps reporting momentum while the player is blocked. If the wall rays are unchanged (after undoing the "
        "known rotation) the step is treated as no motion. Drift after 300 random steps: 9–63 map units, about one 64-unit cell.",
        "<b>Grid</b>: 192×192 cells of 64 units (12,288 units, recentred near edges). Rays mark free space and their hit cell "
        "(wall / door / locked door / item); entities mark enemy (decaying ~10 s), key and item cells; a visit counter marks where the agent "
        "has been. Opened doors are cleared when rays pass through. Keys and items under the agent are cleared when picked up.",
        "<b>Policy view</b>: a 32×32 crop sampled around the agent and rotated ahead-up, so the network learns "
        "“the unexplored corridor is behind-left” independent of map layout. No global coordinates ever reach the policy.",
    ])

    s.append(Paragraph("4. Training and exploration pipeline", H2))
    s.append(Paragraph(
        "<b>Our own recurrent PPO</b> (PyTorch, no SB3): one subprocess per CPU core runs doom_env + feature builder + reward shaping; "
        "a GPU learner samples actions in batch and trains with 128-step rollouts, truncated BPTT over 32-step chunks with stored GRU "
        "states, GAE(λ 0.95, γ 0.995), clipped policy and value losses, return-scale normalisation, entropy 0.03→0.01 and a linear LR decay.", BODY))
    s.append(table([
        ["Reward term (training only)", "Value", "Scorer category"],
        ["Exit reached / remaining health", "+20 / +0.05 per HP", "Exit (largest)"],
        ["Keycard / door opened / locked door opened", "+5 / +1 / +4", "Progression"],
        ["New 128-unit area (ground truth) + <b>intrinsic curiosity</b> 0.03/√visits on the agent's own map", "+0.1 + bonus", "Exploration"],
        ["Kill / damage taken / death", "+1 / −0.02 per HP / −5", "Combat & survival"],
        ["Step / USE press (door bouncing) / firing with no target", "−0.002 / −0.02 / −0.003", "Efficiency, ammo"],
    ], [92 * mm, 38 * mm, W - 130 * mm]))
    s.append(Spacer(1, 3))
    s += bullets([
        "<b>Exploration</b>: the curiosity bonus is computed from the agent's own odometry map, so it rewards genuinely new places from the "
        "agent's point of view and decays on revisits, pushing it out of corners; a <i>stuck</i> input (low speed while moving) and "
        "temperature-0.7 sampling at evaluation stop loops on fixed seeds.",
        "<b>Maps and curriculum</b>: Freedoom 2 MAP01–32, eight ViZDoom scenarios (combat, health, corridor and maze drills) and the 14 "
        "template test WADs: 48 training maps. The sampler starts with 60% small maps, moves to 50% full levels after 8 M steps, and keeps "
        "earlier stages in the mix. Random tick caps (2,100–4,200) and seeds act as domain randomisation.",
        "<b>Held-out benchmark</b>: Freedoom MAP04/09/14/19/24/29 and defend_the_line are never trained on; <i>bench.py</i> plays them "
        "through the real <i>DoomAgent</i> and reports exits, keys, doors, coverage, kills and damage.",
    ])

    s.append(Paragraph("5. Results and compliance", H2))
    rows = results_rows or [
        ["Check", "Result"],
        ["Official validate.py (Linux, Windows)", "PASSED; p50 1.6–3.2 ms, p99 ≤ 7.6 ms per step; init 1–2 s; RAM 0.6 GB"],
        ["Heading / position odometry", "0° error while alive / 9–63 units after 300 random steps"],
        ["Held-out benchmark", "Filled in after the main training run (bench.py)"],
    ]
    s.append(table(rows, [62 * mm, W - 62 * mm]))
    s.append(Spacer(1, 3))
    s.append(Paragraph(
        "Every action is sampled from the trained policy: no rule trees, path planners or aim rules; trigonometry is used only to place "
        "observations on the map. Ground-truth state is used only for training rewards; the agent sees the official observation dict "
        "alone. Code: <i>submission/model</i> (features, policy), <i>submission/agent.py</i>, <i>training/</i>.", SMALL))
    doc.build(s)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="")
    a = ap.parse_args()
    rows = json.loads(Path(a.results).read_text())["rows"] if a.results else None
    print(build(rows))
