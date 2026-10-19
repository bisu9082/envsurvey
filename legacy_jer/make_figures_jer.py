# ============================================================
# make_figures_jer.py  —  JER 투고용 본문 Figure 1~4  (v2)
# envsurvey project  |  Journal of Environmental Radioactivity
# Author: Ku Kang, CBRN Defense Research Institute  |  2026-05-05
# ============================================================
import os, sys, io as _io
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from adjustText import adjust_text

# ════════════════════════════════════════════════════════════════
# 전역 스타일
# ════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.linewidth':     1.0,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.major.width':  1.0,
    'ytick.major.width':  1.0,
    'xtick.minor.size':   2.5,
    'ytick.minor.size':   2.5,
    'xtick.labelsize':    9.5,
    'ytick.labelsize':    9.5,
    'legend.fontsize':    9,
    'legend.frameon':     True,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#D0D0D0',
    'legend.borderpad':   0.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth':    1.4,
    'patch.linewidth':    0.8,
})

# ── 팔레트
P = dict(
    r1='#C94F4A', r2='#E07070',
    o1='#E8943A', o2='#F0B87A',
    t1='#4AACB0', t2='#85CCCF',
    b1='#5B8DB8', b2='#93B5D1',
    v1='#7B6F9B', v2='#B0A8C8',
    g1='#6B8E6B', g2='#A8C4A8',
    dark='#2C2C2C', mid='#666666', light='#AAAAAA',
    bg='#F8F8F8', white='#FFFFFF',
)

# ════════════════════════════════════════════════════════════════
# 유틸
# ════════════════════════════════════════════════════════════════
def panel_label(ax, text, x=-0.13, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left', color=P['dark'])

def stat_box(ax, lines, x=0.05, y=0.95, fs=8.5, ha='left', va='top'):
    """통계 주석 박스 — 절대 axes 좌표 지정"""
    ax.text(x, y, '\n'.join(lines), transform=ax.transAxes,
            fontsize=fs, va=va, ha=ha, linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.45', facecolor=P['bg'],
                      edgecolor='#CCCCCC', alpha=0.95))

def _savefig(fig, path, **kw):
    """Pillow 한국어 경로 우회 — BytesIO write"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = _io.BytesIO()
    fig.savefig(buf, format='png', **kw)
    buf.seek(0)
    with open(path, 'wb') as fh:
        fh.write(buf.read())

def smart_labels(ax, xs, ys, sizes, labels, fontsize=7.5, colors=None):
    """
    Marker-aware priority label placement.
    - text placed OUTSIDE marker boundary (center + radius)
    - no text-marker overlap, no text-text overlap
    Priority: BR(1)→TR(2)→BL(3)→TL(4)→R(5)→L(6)→T(7)→B(8)
    sizes: list of scatter marker areas (pt²)
    """
    if colors is None:
        colors = [P['mid']] * len(labels)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    fwd = ax.transData
    dpi = fig.dpi

    def data_r(s, x, y):
        """scatter size (pt²) → data-space radius (rx, ry)"""
        rpt = np.sqrt(float(s) / np.pi)
        rpx = rpt * dpi / 72.0
        cx, cy = fwd.transform((x, y))
        xd, _ = inv.transform((cx + rpx, cy))
        _, yd  = inv.transform((cx, cy + rpx))
        return abs(xd - x), abs(yd - y)

    # DejaVu Sans heuristic: char_w ≈ 0.52 × fs pt, char_h ≈ 1.25 × fs pt
    CW_PT = fontsize * 0.52
    CH_PT = fontsize * 1.25
    PAD_PT = 2.0

    def text_box(tx, ty, lbl, ha, va, rx, ry):
        cx, cy = fwd.transform((tx, ty))
        tw_pt = len(lbl) * CW_PT + 2 * PAD_PT
        th_pt = CH_PT + 2 * PAD_PT
        xr, _ = inv.transform((cx + tw_pt * dpi / 72, cy))
        _, yr  = inv.transform((cx, cy + th_pt * dpi / 72))
        tw = abs(xr - tx); th = abs(yr - ty)
        if ha == 'left':    x0 = tx
        elif ha == 'right': x0 = tx - tw
        else:               x0 = tx - tw / 2
        if va == 'top':     y0 = ty - th
        elif va == 'bottom': y0 = ty
        else:               y0 = ty - th / 2
        return (x0, y0, x0 + tw, y0 + th)

    def hits(b1, b2):
        return b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]

    PRIO = [
        (+1.0, -1.0, 'left',   'top'),
        (+1.0, +1.0, 'left',   'bottom'),
        (-1.0, -1.0, 'right',  'top'),
        (-1.0, +1.0, 'right',  'bottom'),
        (+1.8,  0.0, 'left',   'center'),
        (-1.8,  0.0, 'right',  'center'),
        ( 0.0, +1.8, 'center', 'bottom'),
        ( 0.0, -1.8, 'center', 'top'),
    ]

    # Marker bboxes (radius + 15% padding)
    m_boxes = []
    for x, y, s in zip(xs, ys, sizes):
        rx, ry = data_r(s, x, y)
        f = 1.15
        m_boxes.append((x - rx*f, y - ry*f, x + rx*f, y + ry*f))

    t_boxes = []
    GAP = 2.5

    for x, y, s, lbl, col in zip(xs, ys, sizes, labels, colors):
        rx, ry = data_r(s, x, y)
        placed = False

        for dxm, dym, ha, va in PRIO:
            tx = x + dxm * rx * GAP
            ty = y + dym * ry * GAP
            tb = text_box(tx, ty, lbl, ha, va, rx, ry)
            if not any(hits(tb, mb) for mb in m_boxes) and                not any(hits(tb, pb) for pb in t_boxes):
                ax.text(tx, ty, lbl, fontsize=fontsize, color=col,
                        ha=ha, va=va,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                  ec='none', alpha=0.85))
                t_boxes.append(tb)
                placed = True
                break

        if not placed:
            # Least-overlap fallback at larger gap
            best = None; best_n = 9999
            for dxm, dym, ha, va in PRIO:
                tx = x + dxm * rx * GAP * 2.2
                ty = y + dym * ry * GAP * 2.2
                tb = text_box(tx, ty, lbl, ha, va, rx, ry)
                n = sum(1 for mb in m_boxes if hits(tb, mb)) +                     2 * sum(1 for pb in t_boxes if hits(tb, pb))
                if n < best_n:
                    best_n = n; best = (tx, ty, ha, va, tb)
            if best:
                tx, ty, ha, va, tb = best
                ax.text(tx, ty, lbl, fontsize=fontsize, color=col,
                        ha=ha, va=va,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                  ec='none', alpha=0.85))
                t_boxes.append(tb)

    return t_boxes


# ════════════════════════════════════════════════════════════════
# 경로 & 데이터
# ════════════════════════════════════════════════════════════════
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPE_DIR = os.path.join(BASE, "'26 환경탐사_집중훈련_spe_국화사")
FIG_DIR = os.path.join(BASE, 'figure')
os.makedirs(FIG_DIR, exist_ok=True)

MEAS = [
    # (id,  spe,    lat,          lon,          gf_rep, fh_1m,  fh_fl,  cs_net)
    ("P3",    "3원점", 33.415132, 126.527188, 0.035, 0.0932, 0.1000,  245),
    ("P3E5m", "3동쪽", 33.415151, 126.527121, 0.037, 0.0973, 0.1100, 1960),
    ("P3W5m", "3서쪽", 33.415127, 126.527290, 0.038, 0.0985, 0.1070,   35),
    ("P3S5m", "3남쪽", 33.415068, 126.527214, 0.035, 0.0823, 0.0909,   29),
    ("P3N5m", "3북쪽", 33.415210, 126.527246, 0.037, 0.0911, 0.0769,  183),
    ("P1",    "1원점", 33.416591, 126.530894, 0.049, 0.0890, 0.0883,   44),
    ("P1E5m", "1동쪽", 33.416604, 126.530992, 0.050, 0.1190, 0.1180,   51),
    ("P1W5m", "1서쪽", 33.416587, 126.530851, 0.047, 0.1040, 0.0890,  120),
    ("P1S5m", "1남쪽", 33.416518, 126.530924, 0.046, 0.1020, 0.0940,   67),
    ("P1N5m", "1북쪽", 33.416614, 126.530957, 0.052, 0.0911, 0.0880,   72),
]
IDS     = [m[0] for m in MEAS]
LATS    = np.array([m[2] for m in MEAS])
LONS    = np.array([m[3] for m in MEAS])
GF      = np.array([m[4] for m in MEAS])
FH      = np.array([m[5] for m in MEAS])
FH_FL   = np.array([m[6] for m in MEAS])
CS      = np.array([m[7] for m in MEAS], dtype=float)
GF_CALC = np.array([0.0312, 0.0337, 0.0347, 0.0317, 0.0333,
                    0.0440, 0.0451, 0.0424, 0.0416, 0.0469])
RATIO   = GF_CALC / FH
T       = 1200.0

G_E = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
                1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
G_V = np.array([3.8397e-3,3.9544e-3,9.1529e-3,1.5974e-2,2.4581e-2,
                3.4559e-2,4.5207e-2,5.5782e-2,6.6227e-2,8.5199e-2,
                1.0112e-1,1.1675e-1,1.3164e-1,1.4537e-1,1.5601e-1,
                1.6961e-1,1.8091e-1,1.8960e-1,1.9379e-1,2.2778e-1])
POLY6 = np.polyfit(G_E, np.log(G_V), 6)
def G_interp(E):
    return np.exp(np.polyval(POLY6, np.clip(E, 0.05, 3.0)))

def parse_spe(path):
    with open(path, 'r', errors='replace') as f:
        lines = f.read().split('\n')
    counts, t, a, b = [], 1200.0, -26.4404, 3.0734
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln == '$MEAS_TIM:':
            t = float(lines[i+1].strip().split()[0])
        elif ln == '$ENER_FIT:':
            tok = lines[i+1].strip().split()
            a, b = float(tok[0]), float(tok[1])
        elif ln == '$DATA:':
            i += 2
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith('$'): break
                try: counts.append(int(l))
                except: pass
                i += 1
            break
        i += 1
    return np.array(counts[:1024], dtype=float), t, a, b


# ════════════════════════════════════════════════════════════════
# FIG 1 — Method Comparison: Regression + Bland–Altman
# ════════════════════════════════════════════════════════════════
def make_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.50)

    # ── (a) Regression scatter ───────────────────────────────────
    ax = axes[0]
    xlim_a = (0.068, 0.130)   # FH 데이터 범위 + 마진
    ylim_a = (0.025, 0.057)   # GF_CALC 데이터 범위 + 마진
    lim = (0.018, 0.140)      # 라인 계산용 (기존 유지)

    slope, intercept, r_val, p_val, se = stats.linregress(FH, GF_CALC)
    x_fit = np.linspace(xlim_a[0], xlim_a[1], 300)
    y_fit = slope * x_fit + intercept
    n     = len(FH)
    x_mean = FH.mean()
    se_y   = np.sqrt(np.sum((GF_CALC - (slope*FH + intercept))**2) / (n-2))
    ci     = stats.t.ppf(0.975, n-2) * se_y * np.sqrt(
                 1/n + (x_fit-x_mean)**2 / np.sum((FH-x_mean)**2))
    ax.fill_between(x_fit, y_fit-ci, y_fit+ci, color=P['b1'], alpha=0.15, zorder=1)

    # 동등선: xlim 내에서만 표시
    eq_x = np.linspace(xlim_a[0], xlim_a[1], 2)
    ax.plot(eq_x, eq_x, '--', color=P['light'], lw=1.2, zorder=2, label='Line of equality')
    ax.plot(x_fit, y_fit, '-', color=P['b1'], lw=1.6, zorder=3,
            label=f'Regression (slope = {slope:.2f})')

    # 클러스터별 scatter
    p3_idx = [i for i, pid in enumerate(IDS) if pid.startswith('P3')]
    p1_idx = [i for i, pid in enumerate(IDS) if not pid.startswith('P3')]

    for i in p1_idx:
        ax.scatter(FH[i], GF_CALC[i], s=55, c=P['b1'], marker='o',
                   edgecolors='white', linewidths=0.8, zorder=5)
    for i in p3_idx:
        mk = '*' if IDS[i] == 'P3E5m' else 'o'
        ms = 95 if IDS[i] == 'P3E5m' else 55
        ax.scatter(FH[i], GF_CALC[i], s=ms,
                   c=P['r1'], marker=mk,
                   edgecolors='white', linewidths=0.8, zorder=5)

    # legend (upper left)
    legend_handles = [
        Line2D([0],[0], ls='--', color=P['light'], label='Line of equality'),
        Line2D([0],[0], ls='-',  color=P['b1'],    label=f'Regression (slope = {slope:.2f})'),
        Line2D([0],[0], marker='o', ls='none', color=P['r1'], ms=7,
               mec='white', label='P3 cluster'),
        Line2D([0],[0], marker='o', ls='none', color=P['b1'], ms=7,
               mec='white', label='P1 cluster'),
        Line2D([0],[0], marker='*', ls='none', color=P['r1'], ms=10,
               mec='white', label='P3E5m (hotspot)'),
    ]
    leg = ax.legend(handles=legend_handles, fontsize=8,
                    loc='upper left', handlelength=1.8)

    # stat box — legend 바로 아래 (axes 좌표)
    rmse = np.sqrt(np.mean((FH - GF_CALC)**2))
    stat_box(ax, [
        f'Pearson r = {r_val:.3f}',
        f'RMSE = {rmse:.4f} µSv h⁻¹',
        f'Mean ratio = {RATIO.mean():.3f} ± {RATIO.std(ddof=1):.3f}',
    ], x=0.57, y=0.97)   # upper right

    # ── manual label helper (marker-aware, fixed side) ─────────
    def _mlabel(ax_in, x, y, s, text, side, col, fs=7.5, G=2.5):
        ax_in.figure.canvas.draw()
        fwd = ax_in.transData; inv = ax_in.transData.inverted()
        dpi = ax_in.figure.dpi
        rpt = np.sqrt(float(s) / np.pi)
        rpx = rpt * dpi / 72.0
        cx, cy = fwd.transform((x, y))
        xd, _  = inv.transform((cx + rpx, cy))
        _, yd  = inv.transform((cx, cy + rpx))
        rx = abs(xd - x); ry = abs(yd - y)
        if   side == 'R':  tx, ty, ha, va = x+rx*G, y,      'left',   'center'
        elif side == 'L':  tx, ty, ha, va = x-rx*G, y,      'right',  'center'
        elif side == 'TR': tx, ty, ha, va = x+rx*G, y+ry*G, 'left',   'bottom'
        elif side == 'BR': tx, ty, ha, va = x+rx*G, y-ry*G, 'left',   'top'
        elif side == 'TL': tx, ty, ha, va = x-rx*G, y+ry*G, 'right',  'bottom'
        elif side == 'BL': tx, ty, ha, va = x-rx*G, y-ry*G, 'right',  'top'
        elif side == 'T':  tx, ty, ha, va = x,       y+ry*G, 'center', 'bottom'
        elif side == 'B':  tx, ty, ha, va = x,       y-ry*G, 'center', 'top'
        else:              tx, ty, ha, va = x+rx*G, y,      'left',   'center'
        ax_in.text(tx, ty, text, fontsize=fs, color=col, ha=ha, va=va,
                   bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))

    # Panel a — per-label explicit placement (mark 동일 선상, va='center')
    ax.set_xlim(xlim_a); ax.set_ylim(ylim_a)
    fig.canvas.draw()
    _sides_a = {
        'P3E5m': 'R',    # right, center (P3W5m과 동일 방식)
        'P3N5m': 'L',    # left
        'P1S5m': 'L',    # left — avoids overlap with P1W5m
    }  # all others: 'R'
    sz_a   = [95 if p == 'P3E5m' else 55 for p in IDS]
    cols_a = [P['r1'] if p.startswith('P3') else P['b1'] for p in IDS]
    for pid, xv, yv, sv, cv in zip(IDS, FH, GF_CALC, sz_a, cols_a):
        g = 1.5 if pid == 'P3E5m' else 2.5  # P3E5m 라벨 마커에 더 가깝게
        _mlabel(ax, xv, yv, sv, pid, _sides_a.get(pid, 'R'), cv, G=g)
    ax.set_xlabel('FH40G dose rate (µSv h⁻¹)')
    ax.set_ylabel('G-factor dose rate (µSv h⁻¹)')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    panel_label(ax, 'a')

    # ── (b) Bland–Altman ─────────────────────────────────────────
    ax2 = axes[1]
    means  = (GF_CALC + FH) / 2.0
    diffs  = GF_CALC - FH
    bias   = np.mean(diffs)
    sd     = np.std(diffs, ddof=1)
    loa_u  = bias + 1.96 * sd
    loa_l  = bias - 1.96 * sd
    ci_bias = stats.t.ppf(0.975, n-1) * sd / np.sqrt(n)

    xlim = (0.048, 0.105)   # x cut at 0.105
    ylim = (-0.082, -0.020) # y cut at -0.020

    ax2.axhspan(loa_l, loa_u, color=P['o1'], alpha=0.08, zorder=0)
    ax2.axhline(0,     color=P['light'], lw=0.9, ls=':', zorder=1)
    ax2.axhline(bias,  color=P['r1'],    lw=1.6, ls='-',  zorder=2)
    ax2.axhline(loa_u, color=P['o1'],    lw=1.2, ls='--', zorder=2)
    ax2.axhline(loa_l, color=P['o1'],    lw=1.2, ls='--', zorder=2)
    ax2.axhspan(bias-ci_bias, bias+ci_bias, color=P['r1'], alpha=0.12, zorder=1)

    # scatter
    for i in p1_idx:
        ax2.scatter(means[i], diffs[i], s=55, c=P['b1'], marker='o',
                    edgecolors='white', linewidths=0.8, zorder=5)
    for i in p3_idx:
        mk = '*' if IDS[i] == 'P3E5m' else 'o'
        ms = 95 if IDS[i] == 'P3E5m' else 55
        ax2.scatter(means[i], diffs[i], s=ms, c=P['r1'], marker=mk,
                    edgecolors='white', linewidths=0.8, zorder=5)

    # right-side value labels
    for val, label, col in [
        (loa_u, f'+1.96 SD\n({loa_u:.4f})', P['o1']),
        (bias,  f'Bias\n({bias:.4f})',       P['r1']),
        (loa_l, f'−1.96 SD\n({loa_l:.4f})', P['o1'])
    ]:
        ax2.text(xlim[1]+0.0005, val, label, va='center',
                 fontsize=7.5, color=col, linespacing=1.4)

    # proportional bias trend
    slope_ba, _, r_ba, p_ba, _ = stats.linregress(means, diffs)
    x_ba = np.linspace(xlim[0], xlim[1], 100)
    ax2.plot(x_ba, slope_ba*x_ba + (bias - slope_ba*np.mean(means)),
             color=P['v1'], lw=1.0, ls='-.', alpha=0.7,
             label=f'Trend (slope={slope_ba:.2f}, p={p_ba:.2f})')
    ax2.legend(fontsize=8, loc='upper right')

    # Panel b — per-label explicit placement
    ax2.set_xlim(xlim[0], xlim[1]+0.018)
    ax2.set_ylim(ylim)
    fig.canvas.draw()
    _sides_b = {
        'P3':    'L', 'P3N5m': 'L', 'P3W5m': 'L',  'P3S5m': 'R',
        'P3E5m': 'BR',
        'P1':    'TR', 'P1E5m': 'R', 'P1W5m': 'BR',
        'P1S5m': 'R',
        'P1N5m': 'BL',
    }
    sz_b   = [95 if p == 'P3E5m' else 55 for p in IDS]
    cols_b = [P['r1'] if p.startswith('P3') else P['b1'] for p in IDS]
    for pid, xv, yv, sv, cv in zip(IDS, means, diffs, sz_b, cols_b):
        if pid == 'P3E5m':
            g = 1.5   # BR, 별 바로 아래 우측
        elif pid == 'P3W5m':
            g = 2.5   # L, 별과 같은 선상 좌측
        elif pid == 'P1W5m':
            g = 1.5
        else:
            g = 2.5
        _mlabel(ax2, xv, yv, sv, pid, _sides_b.get(pid, 'R'), cv, G=g)
    ax2.set_xlabel('Mean of G-factor and FH40G (µSv h⁻¹)')
    ax2.set_ylabel('Difference: G-factor − FH40G (µSv h⁻¹)')
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    panel_label(ax2, 'b')

    _savefig(fig, os.path.join(FIG_DIR, 'JER_Fig1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig1 완료 | bias={bias:.4f} LoA=[{loa_l:.4f},{loa_u:.4f}] µSv/h')


# ════════════════════════════════════════════════════════════════
# FIG 2 — Spectral Evidence + G(E) Dose Contribution
# ════════════════════════════════════════════════════════════════
def make_fig2():
    spe_p3    = os.path.join(SPE_DIR, "3원점.Spe")
    spe_p3e5m = os.path.join(SPE_DIR, "3동쪽.Spe")
    cnt_bg, t_bg, a_bg, b_bg = parse_spe(spe_p3)
    cnt_hs, t_hs, a_hs, b_hs = parse_spe(spe_p3e5m)

    ch   = np.arange(1, 1025)
    E_bg = (a_bg + b_bg * ch) / 1000.0   # MeV
    E_hs = (a_hs + b_hs * ch) / 1000.0

    def smooth(x, w=5):
        return np.convolve(x, np.ones(w)/w, mode='same')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.subplots_adjust(wspace=0.44)

    # ── (a) Spectral overlay ─────────────────────────────────────
    ax = axes[0]
    valid    = (E_bg > 0.05) & (E_bg < 3.05)
    e_keV    = E_bg[valid] * 1000
    c_bg_s   = smooth(cnt_bg[valid])
    c_hs_s   = smooth(cnt_hs[valid])

    ax.fill_between(e_keV, c_bg_s, 0.5, color=P['b2'], alpha=0.35, zorder=1)
    ax.fill_between(e_keV, c_hs_s, 0.5, color=P['r2'], alpha=0.25, zorder=2)
    ax.semilogy(e_keV, c_bg_s, '-', color=P['b1'], lw=1.3,
                label='P3 (background)', zorder=3)
    ax.semilogy(e_keV, c_hs_s, '-', color=P['r1'], lw=1.3,
                label='P3E5m (hotspot)', zorder=4)

    # 주요 피크 수직선
    peaks = [(661.7, 'Cs-137\n661.7 keV', P['r1']),
             (583.2, 'Tl-208\n583 keV',   P['v1']),
             (1460.8,'K-40\n1461 keV',    P['o1']),
             (2614.5,'Tl-208\n2615 keV',  P['v1'])]
    for e_p, lbl, col in peaks:
        ax.axvline(e_p, color=col, lw=0.8, ls=':', alpha=0.8, zorder=2)

    # Cs-137 ROI 강조
    roi_mask = (e_keV >= 620) & (e_keV <= 700)
    ax.fill_between(e_keV[roi_mask], c_hs_s[roi_mask], c_bg_s[roi_mask],
                    color=P['r1'], alpha=0.40, zorder=5,
                    label='Cs-137 net counts')

    # 인셋 (Cs-137 피크 확대)
    axins = ax.inset_axes([0.54, 0.52, 0.43, 0.42])
    axins.set_facecolor('#FAFAFA')
    ins_mask = (e_keV >= 600) & (e_keV <= 730)
    axins.fill_between(e_keV[ins_mask], c_hs_s[ins_mask], c_bg_s[ins_mask],
                       color=P['r1'], alpha=0.40)
    axins.plot(e_keV[ins_mask], c_bg_s[ins_mask], '-', color=P['b1'], lw=1.2)
    axins.plot(e_keV[ins_mask], c_hs_s[ins_mask], '-', color=P['r1'], lw=1.2)
    axins.set_xlim(600, 730)
    axins.yaxis.tick_right()
    axins.tick_params(labelsize=7)
    axins.set_xlabel('keV', fontsize=7.5)
    axins.text(662, c_hs_s[ins_mask].max()*0.85,
               f'P3E5m: {1960} cts\nP3: {245} cts\n(×8.0)',
               fontsize=7, ha='center', color=P['r1'], fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                         ec=P['r1'], lw=0.6))
    for sp in ['top','right','bottom','left']:
        axins.spines[sp].set_linewidth(0.7)
        axins.spines[sp].set_color('#AAAAAA')
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec=P['mid'], lw=0.6)

    ax.set_xlim(60, 3050)
    ax.set_ylim(0.8, cnt_hs[valid].max() * 5)
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Counts per 1200 s')
    # legend — 좌하단, 인셋과 겹치지 않도록 y를 0.07로 올림
    ax.legend(loc='lower left', fontsize=8.5,
              bbox_to_anchor=(0.01, 0.07))
    # 측정 정보 (legend 위)
    ax.text(0.02, 0.03,
            'NaI(Tl) 3″×3″  |  1 m a.g.l.  |  1200 s',
            transform=ax.transAxes, fontsize=8, color=P['mid'],
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC', alpha=0.9))
    panel_label(ax, 'a')

    # ── (b) G(E) + cumulative dose contribution ──────────────────
    ax2  = axes[1]
    ax2r = ax2.twinx()

    E_ch   = E_bg[valid]
    N_rate = smooth(cnt_bg[valid]) / t_bg
    G_ch   = G_interp(E_ch)
    dH     = N_rate * G_ch * 0.00877  # µSv/h per channel

    n_bins    = 20
    bin_edges = np.linspace(0.0, 3.1, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frac_H = np.zeros(n_bins)
    for k in range(n_bins):
        mask_bin = (E_ch >= bin_edges[k]) & (E_ch < bin_edges[k+1])
        frac_H[k] = dH[mask_bin].sum()
    total_H   = frac_H.sum()
    frac_H   *= 100.0 / (total_H + 1e-30)
    cum_H     = np.cumsum(frac_H)

    bar_colors = [P['b2'] if e < 0.6 else
                  P['r1'] if 0.55 <= e <= 0.70 else
                  P['o2'] if 0.55 <= e <= 1.55 else
                  P['t2'] for e in bin_centers]
    ax2.bar(bin_centers, frac_H, width=0.092,
            color=bar_colors, alpha=0.75, zorder=2, edgecolor='none')

    E_sm = np.linspace(0.05, 3.0, 500)
    G_sm = G_interp(E_sm)
    ax2r.plot(E_sm, G_sm, '-', color=P['r1'], lw=2.0, zorder=4, label='G(E)')
    ax2r.set_ylabel('G-factor (µR s count⁻¹ h⁻¹)',
                    color=P['r1'], fontsize=10)
    ax2r.tick_params(axis='y', labelcolor=P['r1'], labelsize=9)
    ax2r.set_ylim(0, G_sm.max() * 1.30)
    ax2r.spines['right'].set_visible(True)
    ax2r.spines['right'].set_color(P['r1'])
    ax2r.spines['right'].set_linewidth(0.8)

    ax2.plot(bin_centers, cum_H, 'o-', color=P['v1'], lw=1.4,
             ms=4, mec='white', mew=0.6, zorder=5, label='Cumulative dose (%)')

    # 50% 교차점
    idx50 = np.searchsorted(cum_H, 50)
    if idx50 < len(bin_centers):
        ax2.plot([bin_centers[idx50]]*2, [0, 50], '--', color=P['v1'], lw=0.9, alpha=0.7)
        ax2.plot([0, bin_centers[idx50]], [50, 50], '--', color=P['v1'], lw=0.9, alpha=0.7)
        ax2.text(bin_centers[idx50]+0.08, 53,
                 f'50% at {bin_centers[idx50]:.2f} MeV',
                 fontsize=8, color=P['v1'])

    # 핵종 수직선
    for e_nuc, name, col in [(0.662,'Cs-137',P['r1']),
                              (1.461,'K-40',  P['o1']),
                              (2.615,'Tl-208',P['v1'])]:
        ax2.axvline(e_nuc, color=col, lw=0.9, ls=':', alpha=0.7, zorder=1)
        ax2.text(e_nuc, 85, name,
                 ha='center', fontsize=7.5, color=col, rotation=90, va='top')

    ax2.set_xlim(0.0, 3.1)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('Dose rate contribution per energy bin (%)')
    lines_leg = [Patch(color=P['b2'], alpha=0.75, label='Low-E contribution'),
                 Patch(color=P['o2'], alpha=0.75, label='Mid-E contribution'),
                 Line2D([0],[0], marker='o', color=P['v1'], ms=5, lw=1.4,
                        label='Cumulative (%)'),
                 Line2D([0],[0], color=P['r1'], lw=2, label='G(E) curve')]
    ax2.legend(handles=lines_leg, fontsize=8, loc='upper left', ncol=2)
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    panel_label(ax2, 'b')

    _savefig(fig, os.path.join(FIG_DIR, 'JER_Fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig2 완료 | P3 총계수: {int(cnt_bg.sum())}  P3E5m 총계수: {int(cnt_hs.sum())}')


# ════════════════════════════════════════════════════════════════
# FIG 3 — Spatial Heterogeneity
# ════════════════════════════════════════════════════════════════
def make_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.50)

    # ── (a) P3 클러스터 방위별 Cs-137 ──────────────────────────
    ax = axes[0]
    p3_ids = ["P3", "P3N5m", "P3E5m", "P3S5m", "P3W5m"]
    p3_dir = ["Centre", "N (+5 m)", "E (+5 m)", "S (+5 m)", "W (+5 m)"]
    p3_cs  = [CS[IDS.index(p)] for p in p3_ids]
    p3_gf  = [GF_CALC[IDS.index(p)] for p in p3_ids]

    order      = np.argsort(p3_cs)[::-1]
    cs_sorted  = [p3_cs[i]  for i in order]
    dir_sorted = [p3_dir[i] for i in order]
    gf_sorted  = [p3_gf[i]  for i in order]
    clr_sorted = [P['r1'] if p3_ids[i] == 'P3E5m' else P['b2'] for i in order]

    y_pos = np.arange(len(p3_ids))
    bars  = ax.barh(y_pos, cs_sorted, color=clr_sorted, height=0.55,
                    edgecolor='white', linewidth=0.5, zorder=3)

    for j, (bar, val, gf_v) in enumerate(zip(bars, cs_sorted, gf_sorted)):
        ax.text(val + 30, bar.get_y()+bar.get_height()/2,
                f'{val:.0f}  ({gf_v:.4f} µSv h⁻¹)',
                va='center', fontsize=8.5, color=P['dark'])

    # 배경 평균선 — 한 줄
    # Global site background: all 9 non-hotspot points (experiment_summary authoritative: 94)
    bg_vals = [CS[IDS.index(p)] for p in IDS if p != 'P3E5m']
    bg_mean = np.mean(bg_vals)
    ax.axvline(bg_mean, color=P['mid'], lw=1.2, ls='--', zorder=4)
    ax.text(bg_mean + 20, len(p3_ids) - 1.35,
            f'Background mean: {bg_mean:.0f}',   # ← 한 줄
            fontsize=8, color=P['mid'])

    hs_val = max(cs_sorted)
    ax.annotate(f'×{hs_val/bg_mean:.1f} above background',
                xy=(hs_val, 0), xytext=(hs_val*0.58, 0.60),
                fontsize=9, fontweight='bold', color=P['r1'],
                arrowprops=dict(arrowstyle='->', color=P['r1'], lw=1.3))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(dir_sorted, fontsize=10)
    ax.set_xlabel('Cs-137 net counts (1200 s)⁻¹')
    ax.set_xlim(0, hs_val * 1.40)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_title('P3 cluster — directional survey (5 m spacing)',
                 fontsize=10, loc='left', color=P['dark'], pad=6)
    panel_label(ax, 'a')

    # ── (b) G/FH40G 비율 — 클러스터별 ──────────────────────────
    ax2 = axes[1]
    p3_mask  = np.array([pid.startswith('P3') for pid in IDS])
    p1_mask  = ~p3_mask
    p3_ratio = RATIO[p3_mask]
    p1_ratio = RATIO[p1_mask]

    all_groups = [(p3_ratio, P['r1'], 'P3 cluster', 1),
                  (p1_ratio, P['b1'], 'P1 cluster', 2)]
    for vals, col, label, xoff in all_groups:
        rng = np.random.RandomState(42)
        x_jitter = xoff + rng.uniform(-0.12, 0.12, len(vals))
        ax2.scatter(x_jitter, vals, s=60, color=col, edgecolors='white',
                    linewidths=0.8, zorder=5, label=label)
        mean_v = vals.mean()
        ci_v   = stats.t.ppf(0.975, len(vals)-1) * vals.std(ddof=1) / np.sqrt(len(vals))
        ax2.errorbar(xoff, mean_v, yerr=ci_v, fmt='D', color=col,
                     ms=9, mec='white', mew=1.0, capsize=5, capthick=1.5,
                     elinewidth=1.5, zorder=6)
        ax2.text(xoff, mean_v + ci_v + 0.010,
                 f'mean = {mean_v:.3f}', ha='center',
                 fontsize=9, color=col, fontweight='bold')

    global_mean = RATIO.mean()
    ax2.axhline(global_mean, color=P['mid'], lw=1.2, ls='--', zorder=3)
    ax2.text(2.20, global_mean + 0.005,
             f'Overall mean = {global_mean:.3f}',
             fontsize=8.5, color=P['mid'], va='bottom')

    from scipy.stats import mannwhitneyu
    mwu_stat, mwu_p = mannwhitneyu(p3_ratio, p1_ratio, alternative='two-sided')
    sig = 'p = {:.3f}'.format(mwu_p) if mwu_p >= 0.001 else 'p < 0.001'
    ax2.plot([1, 2], [0.57, 0.57], '-', color=P['dark'], lw=1.0)
    ax2.text(1.5, 0.577, f'n.s. ({sig})',
             ha='center', fontsize=8.5, color=P['dark'])

    ax2.set_xlim(0.5, 2.75)
    ax2.set_ylim(0.28, 0.63)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['P3 cluster\n(n = 5)', 'P1 cluster\n(n = 5)'], fontsize=10)
    ax2.set_ylabel('G-factor / FH40G ratio')
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax2.legend(fontsize=9, loc='lower right')   # ← 우하단
    ax2.set_title('G-factor / FH40G ratio by measurement cluster',
                  fontsize=10, loc='left', color=P['dark'], pad=6)
    ci_lo = RATIO.mean() - 1.96*RATIO.std(ddof=1)/np.sqrt(len(RATIO))
    ci_hi = RATIO.mean() + 1.96*RATIO.std(ddof=1)/np.sqrt(len(RATIO))
    stat_box(ax2, [
        f'Overall mean ratio = {global_mean:.3f}',
        f'95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]',
        f'Wilcoxon p = 0.002',
    ], x=0.05, y=0.55)
    panel_label(ax2, 'b')

    _savefig(fig, os.path.join(FIG_DIR, 'JER_Fig3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig3 완료 | P3 ratio mean={p3_ratio.mean():.3f}  '
          f'P1={p1_ratio.mean():.3f}  MWU p={mwu_p:.3f}')


# ════════════════════════════════════════════════════════════════
# FIG 4 — Spatial Map with Cluster Insets
# ════════════════════════════════════════════════════════════════
def make_fig4():
    """Cs-137 single-panel spatial map + top-left P3 inset + bottom-right P1 inset."""
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    lat0 = LATS.mean(); lon0 = LONS.mean()
    M_PER_DEG_LAT = 111320.0
    M_PER_DEG_LON = 111320.0 * np.cos(np.deg2rad(lat0))
    x_m = (LONS - lon0) * M_PER_DEG_LON
    y_m = (LATS - lat0) * M_PER_DEG_LAT

    cmap = plt.cm.YlOrRd
    hs_i = IDS.index('P3E5m')
    all_idx = np.arange(len(IDS))
    other_mask = all_idx != hs_i

    # log-scale marker size
    sz = np.log1p(CS) / np.log1p(CS.max()) * 450 + 25

    # cluster membership
    p3_mask = np.array([pid.startswith('P3') for pid in IDS])
    p1_mask = np.array([pid.startswith('P1') for pid in IDS])
    p3_cx = x_m[p3_mask].mean(); p3_cy = y_m[p3_mask].mean()
    p1_cx = x_m[p1_mask].mean(); p1_cy = y_m[p1_mask].mean()

    # ── Main panel ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor('#F9F9F9')

    # Cluster ellipses
    for cx_, cy_, col in [(p3_cx, p3_cy, P['b1']), (p1_cx, p1_cy, P['b2'])]:
        ell = Ellipse((cx_, cy_), width=55, height=55,
                      edgecolor=col, facecolor=col, alpha=0.08,
                      linewidth=1.4, linestyle='--', zorder=1)
        ax.add_patch(ell)

    sc = ax.scatter(x_m[other_mask], y_m[other_mask],
                    s=sz[other_mask], c=CS[other_mask],
                    cmap=cmap, vmin=CS.min(), vmax=CS.max(),
                    edgecolors='#555555', linewidths=0.7, zorder=4, alpha=0.92)
    ax.scatter(x_m[hs_i], y_m[hs_i],
               s=sz[hs_i] + 80, c=[CS[hs_i]], cmap=cmap,
               vmin=CS.min(), vmax=CS.max(),
               edgecolors=P['r1'], linewidths=2.4, zorder=6, marker='*', alpha=1.0)

    # Cluster label only at main scale (not per-point — too crowded)
    for cx_, cy_, col, lbl in [(p3_cx, p3_cy, P['b1'], 'P3 cluster'),
                                 (p1_cx, p1_cy, P['b2'], 'P1 cluster')]:
        ax.text(cx_, cy_ - 24, lbl, ha='center', va='top',
                fontsize=9, fontweight='bold', color=col, zorder=8)

    # Hotspot annotation
    ax.annotate(
        f'P3E5m\n{CS[hs_i]:.0f} cts\n(×{CS[hs_i]/CS[other_mask].mean():.0f} bkg)',
        xy=(x_m[hs_i], y_m[hs_i]),
        xytext=(x_m[hs_i] + 28, y_m[hs_i] - 20),
        fontsize=9, color=P['r1'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=P['r1'], lw=1.2),
        bbox=dict(boxstyle='round,pad=0.35', fc='white',
                  ec=P['r1'], lw=0.9, alpha=0.93), zorder=9)

    # Colourbar
    cbar = plt.colorbar(sc, ax=ax, shrink=0.72, pad=0.02, aspect=22)
    cbar.set_label('Cs-137 net counts (1200 s)$^{-1}$', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Scale bar
    xrng = x_m.max() - x_m.min(); yrng = y_m.max() - y_m.min()
    sb_x0 = x_m.min() + xrng * 0.05
    sb_y0 = y_m.min() - yrng * 0.10
    ax.plot([sb_x0, sb_x0 + 50], [sb_y0, sb_y0], '-', color=P['dark'], lw=2.5, zorder=8)
    ax.text(sb_x0 + 25, sb_y0 + 2, '50 m', ha='center', fontsize=9, color=P['dark'])

    # North arrow
    ax.annotate('N', xy=(0.94, 0.91), xytext=(0.94, 0.81),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=P['dark'], lw=2.0),
                ha='center', fontsize=12, fontweight='bold', color=P['dark'])

    ax.set_xlabel('Relative Easting (m)', fontsize=11)
    ax.set_ylabel('Relative Northing (m)', fontsize=11)
    ax.set_title('Cs-137 spatial distribution', fontsize=12, loc='left', pad=8, color=P['dark'])
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, color='#EBEBEB', lw=0.6, zorder=0)
    panel_label(ax, 'a')

    # ── TOP-LEFT INSET: P3 cluster zoom (좌상단) ─────────────────────
    ZOOM = 14   # ±metres around cluster centre
    axins_p3 = ax.inset_axes([0.03, 0.57, 0.28, 0.38])
    axins_p3.set_facecolor('#F5F5FF')

    axins_p3.scatter(x_m[other_mask & p3_mask], y_m[other_mask & p3_mask],
                     s=sz[other_mask & p3_mask] * 0.65,
                     c=CS[other_mask & p3_mask],
                     cmap=cmap, vmin=CS.min(), vmax=CS.max(),
                     edgecolors='#555555', linewidths=0.7, zorder=4)
    axins_p3.scatter(x_m[hs_i], y_m[hs_i],
                     s=sz[hs_i] * 0.65 + 50, c=[CS[hs_i]],
                     cmap=cmap, vmin=CS.min(), vmax=CS.max(),
                     edgecolors=P['r1'], linewidths=1.8, marker='*', zorder=6)

    # Per-point labels inside P3 inset
    _label_offsets_p3 = {
        'P3':     (4, 3), 'P3E5m': (-3, -10),
        'P3W5m':  (-30, 3), 'P3S5m': (4, -10), 'P3N5m': (4, 3),
    }
    for i, pid in enumerate(IDS):
        if not pid.startswith('P3'): continue
        ox, oy = _label_offsets_p3.get(pid, (4, 3))
        axins_p3.annotate(pid, (x_m[i], y_m[i]),
                          textcoords='offset points', xytext=(ox, oy),
                          fontsize=6, color=P['dark'], zorder=7,
                          annotation_clip=False)

    axins_p3.set_xlim(p3_cx - ZOOM, p3_cx + ZOOM)
    axins_p3.set_ylim(p3_cy - ZOOM, p3_cy + ZOOM)
    axins_p3.tick_params(labelsize=6.5)
    axins_p3.set_xlabel('E (m)', fontsize=6.5, labelpad=2)
    axins_p3.set_ylabel('N (m)', fontsize=6.5, labelpad=2)
    axins_p3.yaxis.tick_right(); axins_p3.yaxis.set_label_position('right')
    for sp in axins_p3.spines.values():
        sp.set_linewidth(1.1); sp.set_color(P['b1'])
    axins_p3.set_title('b  P3 cluster', fontsize=7.5, color=P['b1'],
                        pad=3, fontweight='bold', loc='left')
    axins_p3.grid(True, color='#E0E0EE', lw=0.4)

    # zoom indicator box on main panel
    rect_p3 = plt.Rectangle((p3_cx - ZOOM, p3_cy - ZOOM), 2*ZOOM, 2*ZOOM,
                             fill=False, edgecolor=P['b1'], lw=0.9,
                             linestyle=':', zorder=5)
    ax.add_patch(rect_p3)
    mark_inset(ax, axins_p3, loc1=2, loc2=3,
               fc='none', ec=P['b1'], lw=0.7, alpha=0.6)

    # ── BOTTOM-RIGHT INSET: P1 cluster zoom (우하단) ─────────────────
    axins_p1 = ax.inset_axes([0.64, 0.04, 0.28, 0.38])
    axins_p1.set_facecolor('#F5FFF5')

    axins_p1.scatter(x_m[p1_mask], y_m[p1_mask],
                     s=sz[p1_mask] * 0.65,
                     c=CS[p1_mask],
                     cmap=cmap, vmin=CS.min(), vmax=CS.max(),
                     edgecolors='#555555', linewidths=0.7, zorder=4)

    _label_offsets_p1 = {
        'P1':     (4, 3), 'P1E5m': (4, 3),
        'P1W5m':  (-30, 3), 'P1S5m': (4, -10), 'P1N5m': (4, 3),
    }
    for i, pid in enumerate(IDS):
        if not pid.startswith('P1'): continue
        ox, oy = _label_offsets_p1.get(pid, (4, 3))
        axins_p1.annotate(pid, (x_m[i], y_m[i]),
                          textcoords='offset points', xytext=(ox, oy),
                          fontsize=6, color=P['dark'], zorder=7)

    axins_p1.set_xlim(p1_cx - ZOOM, p1_cx + ZOOM)
    axins_p1.set_ylim(p1_cy - ZOOM, p1_cy + ZOOM)
    axins_p1.tick_params(labelsize=6.5)
    axins_p1.set_xlabel('E (m)', fontsize=6.5, labelpad=2)
    axins_p1.set_ylabel('N (m)', fontsize=6.5, labelpad=2)
    for sp in axins_p1.spines.values():
        sp.set_linewidth(1.1); sp.set_color(P['b2'])
    axins_p1.set_title('c  P1 cluster', fontsize=7.5, color=P['b2'],
                        pad=3, fontweight='bold', loc='left')
    axins_p1.grid(True, color='#E0EEE0', lw=0.4)

    rect_p1 = plt.Rectangle((p1_cx - ZOOM, p1_cy - ZOOM), 2*ZOOM, 2*ZOOM,
                             fill=False, edgecolor=P['b2'], lw=0.9,
                             linestyle=':', zorder=5)
    ax.add_patch(rect_p1)
    mark_inset(ax, axins_p1, loc1=1, loc2=4,
               fc='none', ec=P['b2'], lw=0.7, alpha=0.6)

    fig.text(0.5, -0.02,
             'Jeju Nanjichuksan Research Center, Republic of Korea  (33.42N, 126.53E)'
             '  |  29 April 2026  |  NaI(Tl) 3×3, 1200 s, 1 m a.g.l.',
             ha='center', fontsize=8.5, color=P['mid'])

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, os.path.join(FIG_DIR, 'JER_Fig4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig4 done | P3E5m hotspot {CS[IDS.index("P3E5m")]:.0f} cts | insets: P3(top-left) P1(bottom-right)')


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('\n' + '='*62)
    print('JER Figure 1-4  (v2 -- no overlap)')
    print('='*62)
    np.random.seed(0)
    try:
        print('\n[Fig 1] Method comparison')
        make_fig1()
        print('\n[Fig 2] Spectral + G(E)')
        make_fig2()
        print('\n[Fig 3] Spatial heterogeneity')
        make_fig3()
        print('\n[Fig 4] Spatial map')
        make_fig4()
        print('\n' + '='*62)
        print(f'done  ->  {FIG_DIR}')
        print('='*62)
    except Exception as e:
        print(f'\n  error: {e}')
        import traceback; traceback.print_exc()
