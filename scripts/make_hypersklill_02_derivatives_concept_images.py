#!/usr/bin/env python3
"""Concept sheets for hypersklill/02_derivaties (derivatives).

One page per notebook that summarizes its central idea. Every number is
computed here from the same example functions, points, step sizes and
learning rates the notebook uses, with asserts against the notebook's values.
Images are written to hypersklill/02_derivaties/img/.

02_derivatives_of_trigonometric_functions_updated.ipynb is a near-duplicate of
02_derivatives_of_trigonometric_functions.ipynb (only the sine derivation
differs), so only the latter gets the sheet.

Run from anywhere:  python scripts/make_hypersklill_02_derivatives_concept_images.py
"""
from functools import partial
from math import e, exp, log, pi, sqrt

import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle

from viz_style import (REPO, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "hypersklill" / "02_derivaties"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#bfe8d6"


# ================================================================ helpers
def fwd_diff(func, x, h):
    """The notebooks' forward difference quotient (f(x+h) - f(x)) / h."""
    return (func(x + h) - func(x)) / h


def tangent(ax, x0, y0, slope, half, color=ORANGE, lw=2.0, ls="-", label=None, dot=True, zorder=3):
    xs = np.array([x0 - half, x0 + half])
    ax.plot(xs, y0 + slope * (xs - x0), color=color, linewidth=lw, linestyle=ls, label=label,
            zorder=zorder)
    if dot:
        ax.plot(x0, y0, "o", color=color, markersize=6, zorder=zorder + 1)


def axes_lines(ax, x=True, y=True):
    if y:
        ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    if x:
        ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def stacked(fig, cell, ratios=(1.25, 1), hspace=0.08):
    """Two axes stacked in one grid cell, sharing x (function on top, derivative below)."""
    sub = cell.subgridspec(2, 1, height_ratios=ratios, hspace=hspace)
    top = fig.add_subplot(sub[0])
    bottom = fig.add_subplot(sub[1], sharex=top)
    top.tick_params(labelbottom=False)
    return top, bottom


def sign_shading(ax, x, d, label_pos="f′ > 0: f increasing", label_neg="f′ < 0: f decreasing"):
    ax.fill_between(x, d, 0, where=d > 0, color=AQUA, alpha=0.25, linewidth=0, label=label_pos)
    ax.fill_between(x, d, 0, where=d < 0, color=ORANGE, alpha=0.25, linewidth=0, label=label_neg)


# ================================================================ notebook 00
def sheet_00_introduction():
    def f(x):
        return x ** 2

    x0 = 1
    hs = [1.5, 0.7, 0.25]
    slopes = [(f(x0 + h) - f(x0)) / h for h in hs]
    assert np.allclose(slopes, [3.5, 2.7, 2.25])
    approx = fwd_diff(lambda t: t ** 2, 3, 1e-5)
    assert abs(approx - 6.000009999951316) < 1e-12

    fig = new_figure("The derivative is the secant slope in the limit Δx → 0",
                     "Average slope Δf/Δx = (f(x+Δx) − f(x))/Δx;   f′(x) = lim of that quotient as Δx → 0.   "
                     "For f(x) = x² at x = 1 it is 2.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-1, 4, 400)
    ax.plot(x, f(x), color=INK, linewidth=2.2, label="f(x) = x²")
    for h, m, color in zip(hs, slopes, [BLUE_LIGHT, BLUE, AQUA]):
        ax.plot(x, f(x0) + m * (x - x0), color=color, linewidth=1.6,
                label=f"secant h = {h:g}: slope {m:g}")
        ax.plot([x0 + h], [f(x0 + h)], "o", color=color, markersize=6)
    ax.plot(x, f(x0) + 2 * (x - x0), color=ORANGE, linewidth=2.2, linestyle="--",
            label="tangent: slope 2")
    ax.plot(x0, f(x0), "o", color=ORANGE, markersize=8, zorder=5)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-3, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Secant lines swing into the tangent")

    ax = fig.add_subplot(gs[0, 1])
    h_cont = np.linspace(0.005, 1.6, 300)
    ax.plot(h_cont, 2 + h_cont, color=BLUE, linewidth=2, label="secant slope = 2 + h")
    ax.plot(hs, slopes, "o", color=BLUE, markersize=8)
    for h, m in zip(hs, slopes):
        ax.annotate(f"{m:g}", (h, m), textcoords="offset points", xytext=(6, -14), fontsize=10)
    ax.axhline(2, color=ORANGE, linewidth=2, linestyle="--", label="limit f′(1) = 2")
    ax.plot(0, 2, "o", color=ORANGE, markersize=9, clip_on=False, zorder=5)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(1.5, 4)
    ax.set_xlabel("interval width h = Δx")
    ax.set_ylabel("slope of the secant")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Shrink Δx: the slope settles at 2")

    panel(fig.add_subplot(gs[0, 2]), "The limit, done by hand", [
        r"$f(x) = x^2$:",
        r"$\frac{(x+\Delta x)^2 - x^2}{\Delta x} = \frac{2x\Delta x + (\Delta x)^2}{\Delta x} = 2x + \Delta x$",
        r"$\Delta x \to 0:\quad f'(x) = 2x$",
        "",
        r"$f(x) = 5x + 8$:",
        r"$\Delta f = 5\Delta x \;\Rightarrow\; \Delta f/\Delta x = 5 = f'(x)$",
        "A line has the same slope everywhere.",
        "",
        "Notation:  f′(x) = df/dx = slope of the",
        "tangent = instantaneous rate of change.",
    ], spacing=0.093)

    ax = fig.add_subplot(gs[1, 0])
    steps = np.logspace(-15, 0, 300)
    errors = np.abs(fwd_diff(lambda t: t ** 2, 3.0, steps) - 6)
    ax.loglog(steps, errors, color=BLUE, linewidth=1.6, label="|(f(3+h) − f(3))/h − 6|")
    ax.loglog(steps, steps, color=MUTED, linestyle=":", linewidth=1.4, label="truncation error h")
    ax.plot(1e-5, abs(approx - 6), "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate(f"notebook: h = 10⁻⁵\n→ {approx:.6f}", (1e-5, abs(approx - 6)), xytext=(1e-3, 1e-10),
                ha="center", fontsize=10, color=ORANGE, weight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_ylim(1e-12, 1e4)
    ax.set_xlabel("step h (log scale)")
    ax.set_ylabel("error vs exact f′(3) = 6")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Numerically: small h, but not too small")

    top, bottom = stacked(fig, gs[1, 1])
    x = np.linspace(-2.6, 2.6, 400)
    y, dy = x ** 3 - 3 * x, 3 * x ** 2 - 3
    top.plot(x, y, color=INK, linewidth=2, label="f(x) = x³ − 3x")
    top.plot([-1, 1], [2, -2], "o", color=ORANGE, markersize=8, label="f′ = 0: (−1, 2), (1, −2)")
    for c in (-1, 1):
        tangent(top, c, c ** 3 - 3 * c, 0, 0.6, dot=False)
    axes_lines(top, x=False)
    top.set_ylim(-6, 6)
    top.legend(loc="upper left", fontsize=9)
    top.grid(True)
    top.set_title("The sign of f′ says up or down")
    bottom.plot(x, dy, color=BLUE, linewidth=2, label="f′(x) = 3x² − 3")
    sign_shading(bottom, x, dy)
    axes_lines(bottom, x=False)
    bottom.set_ylim(-4.5, 18)
    bottom.set_xlabel("x")
    bottom.legend(loc="upper center", fontsize=8.5, ncol=1)
    bottom.grid(True)

    panel(fig.add_subplot(gs[1, 2]), "Reading the sign", [
        "f′(x) > 0  →  f is increasing",
        "f′(x) < 0  →  f is decreasing",
        "f′(x) = 0  →  max, min, or a flat point",
        "",
        "x³ − 3x: rising, then falling on (−1, 1),",
        "then rising again.",
        "",
        "Caution: f′ = 0 is not always an extremum.",
        r"$f(x) = x^3$:  $f'(0) = 0$, yet f keeps rising.",
        "",
        "Quick check: f(x) = 3x² + 2x  →  f′ = 6x + 2.",
    ], spacing=0.085)
    save(fig, name="00_derivative_definition.png")


# ================================================================ notebook 01
def sheet_01_polynomials():
    xs = sp.symbols("x")
    assert sp.diff(3 * xs ** 3 + 5 * xs ** 2 + 8 * xs + 1, xs) == 9 * xs ** 2 + 10 * xs + 8
    quotient = (xs ** 5 + 2) / (xs + 4)
    q_prime = sp.simplify(sp.diff(quotient, xs) - (4 * xs ** 5 + 20 * xs ** 4 - 2) / (xs + 4) ** 2)
    assert q_prime == 0
    assert sp.diff((3 * xs ** 4) * (5 * xs ** 2), xs) == 90 * xs ** 5

    def poly(t):
        return 3 * t ** 3 + 5 * t ** 2 + 8 * t + 1

    numeric = fwd_diff(poly, 2.0, 1e-6)
    assert abs(numeric - 64.00002300921415) < 1e-8
    terms = [9 * 2 ** 2, 10 * 2, 8, 0]
    assert sum(terms) == 64

    fig = new_figure("Polynomials: differentiate term by term with the power rule",
                     "(xⁿ)′ = n xⁿ⁻¹,   (c f)′ = c f′,   (f ± g)′ = f′ ± g′;   "
                     "products and quotients need their own rules.")
    gs = grid(fig, top=0.84, wspace=0.3)

    top, bottom = stacked(fig, gs[0, 0])
    x = np.linspace(-1, 4, 400)
    y, dy = x ** 3 - 3 * x ** 2 + 2, 3 * x ** 2 - 6 * x
    top.plot(x, y, color=INK, linewidth=2, label="f(x) = x³ − 3x² + 2")
    top.plot([0, 2], [2, -2], "o", color=ORANGE, markersize=8)
    for c in (0, 2):
        tangent(top, c, c ** 3 - 3 * c ** 2 + 2, 0, 0.6, dot=False)
    top.set_ylim(-3.5, 6)
    top.legend(loc="upper left", fontsize=9.5)
    top.grid(True)
    top.set_title("A polynomial and its derivative")
    bottom.plot(x, dy, color=BLUE, linewidth=2, label="f′(x) = 3x² − 6x")
    sign_shading(bottom, x, dy)
    axes_lines(bottom, x=False)
    bottom.set_ylim(-4, 26)
    bottom.set_xlabel("x")
    bottom.legend(loc="upper left", fontsize=8.5)
    bottom.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    names = ["(3x³)′ = 9x²", "(5x²)′ = 10x", "(8x)′ = 8", "(1)′ = 0"]
    left = 0
    for name, value, color in zip(names, terms, [BLUE, AQUA, ORANGE, MUTED]):
        ax.barh(0, value, left=left, color=color, height=0.5, zorder=2)
        if value:
            ax.text(left + value / 2, 0, f"{value}", ha="center", va="center", color="white",
                    fontsize=12, weight="bold")
            ax.text(left + value / 2, 0.42, name, ha="center", va="bottom", fontsize=10)
        left += value
    ax.text(64, -0.45, f"total f′(2) = {sum(terms)}", ha="right", fontsize=11, weight="bold")
    ax.text(64, -0.75, f"forward difference, h = 10⁻⁶: {numeric:.8f}", ha="right", fontsize=10,
            color=INK2)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.95, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("contribution to the slope at x = 2")
    ax.grid(axis="x")
    ax.set_title("f = 3x³ + 5x² + 8x + 1: slopes add up")

    panel(fig.add_subplot(gs[0, 2]), "The rules", [
        r"$(c)' = 0 \qquad (x)' = 1$",
        r"$(x^n)' = n\,x^{n-1}$",
        r"$(c\,f)' = c\,f' \qquad (f \pm g)' = f' \pm g'$",
        r"$(f g)' = f' g + f g'$",
        r"$(f/g)' = (f' g - f g')\,/\,g^2,\quad g \neq 0$",
        "",
        "Every rule follows from the limit definition;",
        "e.g. (x + Δx)² − x² = 2xΔx + (Δx)².",
        "",
        r"$(4x^5 - 2x^3 + 7x - 9)' = 20x^4 - 6x^2 + 7$",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    x1, dx = 1.0, 0.1
    f1, g1 = 3 * x1 ** 4, 5 * x1 ** 2
    f2, g2 = 3 * (x1 + dx) ** 4, 5 * (x1 + dx) ** 2
    ax.add_patch(Rectangle((0, 0), f1, g1, facecolor=BLUE_LIGHT, edgecolor=INK2))
    ax.add_patch(Rectangle((f1, 0), f2 - f1, g1, facecolor=AQUA, alpha=0.6, edgecolor=INK2))
    ax.add_patch(Rectangle((0, g1), f1, g2 - g1, facecolor=ORANGE, alpha=0.6, edgecolor=INK2))
    ax.add_patch(Rectangle((f1, g1), f2 - f1, g2 - g1, facecolor=MUTED, alpha=0.5, edgecolor=INK2))
    ax.text(f1 / 2, g1 / 2, "f · g", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(f1 + (f2 - f1) / 2, g1 / 2, "Δf · g", ha="center", va="center", fontsize=11,
            rotation=90)
    ax.text(f1 / 2, g1 + (g2 - g1) / 2, "f · Δg", ha="center", va="center", fontsize=11)
    ax.set_xlim(0, f2 * 1.05)
    ax.set_ylim(0, g2 * 1.3)
    ax.set_xlabel("f = 3x⁴: 3 at x = 1, 4.39 at x = 1.1")
    ax.set_ylabel("g = 5x²: 5 → 6.05")
    ax.set_title("Product rule: the area grows on two sides")
    ax.text(f2 * 1.03, g2 * 1.28, "÷ Δx and let Δx → 0 (the corner ΔfΔg vanishes):\n"
            "(fg)′ = f′g + fg′ = 12·5 + 3·10 = 90 at x = 1",
            ha="right", va="top", fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    xq = np.linspace(-1.5, 1.5, 400)
    qf = (xq ** 5 + 2) / (xq + 4)
    right = -2 / 16
    wrong = 2 / 16
    assert abs(fwd_diff(lambda t: (t ** 5 + 2) / (t + 4), 0.0, 1e-7) - right) < 1e-5
    ax.plot(xq, qf, color=INK, linewidth=2, label="f(x) = (x⁵ + 2)/(x + 4)")
    tangent(ax, 0, 0.5, right, 1.2, color=ORANGE, label=f"numerator −2: slope {right:+.3f}")
    tangent(ax, 0, 0.5, wrong, 1.2, color=MUTED, ls="--", dot=False,
            label=f"numerator +2: slope {wrong:+.3f}")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Quotient rule: the sign of −2 matters")

    panel(fig.add_subplot(gs[1, 2]), "Worked quotient", [
        r"$f(x) = \frac{x^5 + 2}{x + 4}$",
        r"$f'(x) = \frac{5x^4(x+4) - (x^5+2)\cdot 1}{(x+4)^2}$",
        r"$\qquad = \frac{4x^5 + 20x^4 - 2}{(x+4)^2}$",
        "",
        "f′(0) = −2/16 = −0.125, as the finite",
        "difference confirms; +2 flips the sign.",
        "(fg)′ ≠ f′g′: (3x⁴·5x²)′ = 90x⁵, not 120x⁴.",
    ], spacing=0.118)
    save(fig, name="01_power_rule.png")


# ================================================================ notebook 02
def sheet_02_trigonometric():
    xs = sp.symbols("x")
    assert sp.diff(sp.sin(xs), xs) == sp.cos(xs)
    assert sp.simplify(sp.diff(sp.tan(xs), xs) - 1 / sp.cos(xs) ** 2) == 0
    assert sp.simplify(sp.diff(sp.cot(xs), xs) + 1 / sp.sin(xs) ** 2) == 0
    approx_sin = fwd_diff(np.sin, 1.0, 1e-6)
    approx_cos = fwd_diff(np.cos, 1.0, 1e-6)
    assert abs(approx_sin - 0.5403018851213304) < 1e-12
    assert abs(approx_cos + 0.8414712548976411) < 1e-12

    fig = new_figure("Trig derivatives: the slope of sin is cos, the slope of cos is −sin",
                     "(sin x)′ = cos x,   (cos x)′ = −sin x,   (tan x)′ = 1/cos² x,   (cot x)′ = −1/sin² x")
    gs = grid(fig, top=0.84, wspace=0.3)
    x = np.linspace(-2 * pi, 2 * pi, 800)
    ticks = [-2 * pi, -pi, 0, pi, 2 * pi]
    tick_labels = ["−2π", "−π", "0", "π", "2π"]
    marks = np.array([-1.5 * pi, -pi, -0.5 * pi, 0, 0.5 * pi, pi, 1.5 * pi])

    for cell, func, deriv, name, dname in [
            (gs[0, 0], np.sin, np.cos, "sin x", "cos x"),
            (gs[1, 0], np.cos, lambda t: -np.sin(t), "cos x", "−sin x")]:
        top, bottom = stacked(fig, cell)
        top.plot(x, func(x), color=INK, linewidth=1.6, label=name)
        for m in marks:
            tangent(top, m, func(m), deriv(m), 0.85, color=ORANGE, lw=2.4, zorder=4)
        top.set_ylim(-1.6, 2.2)
        top.set_yticks([-1, 0, 1])
        top.legend(loc="upper right", fontsize=9.5)
        top.grid(True)
        top.set_title(f"Tangent slopes of {name} trace {dname}")
        bottom.plot(x, deriv(x), color=BLUE, linewidth=2, label=f"slope = {dname}")
        bottom.plot(marks, deriv(marks), "o", color=ORANGE, markersize=6)
        axes_lines(bottom, x=False)
        bottom.set_ylim(-1.5, 2.2)
        bottom.set_yticks([-1, 0, 1])
        bottom.set_xticks(ticks, tick_labels)
        bottom.set_xlabel("x")
        bottom.legend(loc="upper right", fontsize=9.5)
        bottom.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    h = np.linspace(-1.5, 1.5, 601)
    h = h[h != 0]
    ax.plot(h, np.sin(h) / h, color=BLUE, linewidth=2.2, label="sin h / h  → 1")
    ax.plot(h, (np.cos(h) - 1) / h, color=AQUA, linewidth=2.2, label="(cos h − 1) / h  → 0")
    ax.plot([0, 0], [1, 0], "o", color=ORANGE, markersize=9, zorder=5)
    axes_lines(ax)
    ax.set_ylim(-0.9, 1.3)
    ax.set_xlabel("h")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)
    ax.set_title("The two limits behind (sin x)′ = cos x")

    panel(fig.add_subplot(gs[0, 2]), "Deriving (sin x)′ from the limit", [
        r"$\frac{\sin(x+h) - \sin x}{h} = \sin x\,\frac{\cos h - 1}{h} + \cos x\,\frac{\sin h}{h}$",
        "",
        r"$h \to 0:\quad \sin x \cdot 0 + \cos x \cdot 1 = \cos x$",
        "",
        "Cosine works the same way: (cos x)′ = −sin x.",
        "",
        "Differentiate four times and you are back:",
        "sin → cos → −sin → −cos → sin.",
    ], spacing=0.1)

    ax = fig.add_subplot(gs[1, 1])
    xt = np.linspace(-pi / 2 + 0.05, pi / 2 - 0.05, 400)
    ax.plot(xt, np.tan(xt), color=INK, linewidth=2, label="tan x")
    ax.plot(xt, 1 / np.cos(xt) ** 2, color=BLUE, linewidth=2, label="(tan x)′ = 1/cos² x")
    ax.axhline(1, color=ORANGE, linestyle=":", linewidth=1.5)
    ax.text(-1.45, 0.3, "slope ≥ 1 everywhere", color=ORANGE, fontsize=10, weight="bold")
    for edge in (-pi / 2, pi / 2):
        ax.axvline(edge, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(pi / 2 - 0.05, -5.3, "cos x = 0:\nundefined", ha="right", fontsize=9.5, color=INK2)
    axes_lines(ax)
    ax.set_xlim(-pi / 2 - 0.15, pi / 2 + 0.15)
    ax.set_ylim(-6, 8)
    ax.set_xticks([-pi / 2, 0, pi / 2], ["−π/2", "0", "π/2"])
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("tan x = sin x / cos x, by the quotient rule")

    panel(fig.add_subplot(gs[1, 2]), "Checks from the notebook", [
        r"$(\tan x)' = \frac{\cos^2 x + \sin^2 x}{\cos^2 x} = \frac{1}{\cos^2 x}$",
        "",
        "Forward difference at x = 1, h = 10⁻⁶:",
        f"`(sin)′(1) ≈ {approx_sin:.10f}`",
        f"`cos 1     = {np.cos(1.0):.10f}`",
        f"`(cos)′(1) ≈ {approx_cos:.10f}`",
        f"`−sin 1    = {-np.sin(1.0):.10f}`",
        "",
        r"$(\tan x + \cos x)' = 1/\cos^2 x - \sin x$",
        r"$(x / \tan x)' = 1/\tan x - x/\sin^2 x$",
    ], spacing=0.085)
    save(fig, name="02_trig_derivatives.png")


# ================================================================ notebook 03
def sheet_03_chain_rule():
    xs = sp.symbols("x")
    F = (xs ** 4 + 2) ** sp.Rational(1, 3)
    assert sp.simplify(sp.diff(F, xs) - 4 * xs ** 3 / (3 * (xs ** 4 + 2) ** sp.Rational(2, 3))) == 0
    assert sp.diff(sp.cos(xs ** 3), xs) == -3 * xs ** 2 * sp.sin(xs ** 3)
    assert sp.simplify(sp.diff(sp.cos(xs) ** 3, xs) + 3 * sp.cos(xs) ** 2 * sp.sin(xs)) == 0
    layered = sp.cos(sp.sin(sp.cot(xs)))
    layered_d = sp.sin(sp.sin(sp.cot(xs))) * sp.cos(sp.cot(xs)) / sp.sin(xs) ** 2
    assert abs(float((sp.diff(layered, xs) - layered_d).subs(xs, 0.9))) < 1e-12

    x0 = 1.0
    u0 = x0 ** 4 + 2
    inner_slope = 4 * x0 ** 3
    outer_slope = (1 / 3) * u0 ** (-2 / 3)
    chain = inner_slope * outer_slope
    numeric = fwd_diff(lambda t: (t ** 4 + 2) ** (1 / 3), x0, 1e-7)
    assert abs(numeric - chain) < 1e-5

    fig = new_figure("Chain rule: outer slope (at the inside) times inner slope",
                     "F(x) = f(g(x))   ⇒   F′(x) = f′(g(x)) · g′(x).   "
                     "Example F(x) = ∛(x⁴ + 2) at x = 1.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 1.5, 300)
    ax.plot(x, x ** 4 + 2, color=INK, linewidth=2, label="inner  g(x) = x⁴ + 2")
    tangent(ax, x0, u0, inner_slope, 0.3, label=f"g′(1) = {inner_slope:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u = g(x)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Step 1: inner slope at x = 1")

    ax = fig.add_subplot(gs[0, 1])
    u = np.linspace(0.2, 7, 300)
    ax.plot(u, u ** (1 / 3), color=INK, linewidth=2, label="outer  f(u) = ∛u")
    tangent(ax, u0, u0 ** (1 / 3), outer_slope, 1.8, color=AQUA,
            label=r"$f'(3) = \frac{1}{3}\,3^{-2/3}$" + f" = {outer_slope:.4f}")
    ax.set_xlabel("u")
    ax.set_ylabel("f(u)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Step 2: outer slope at u = g(1) = 3")

    ax = fig.add_subplot(gs[0, 2])
    x = np.linspace(0, 1.6, 300)
    Fx = (x ** 4 + 2) ** (1 / 3)
    ax.plot(x, Fx, color=INK, linewidth=2, label="F(x) = ∛(x⁴ + 2)")
    tangent(ax, x0, u0 ** (1 / 3), chain, 0.4, color=BLUE,
            label=f"F′(1) = {inner_slope:g} × {outer_slope:.4f} = {chain:.4f}")
    ax.text(0.03, 0.62, f"finite difference: {numeric:.4f}", transform=ax.transAxes, fontsize=10,
            color=INK2)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Step 3: multiply the rates")

    top, bottom = stacked(fig, gs[1, 0], ratios=(1, 1))
    x = np.linspace(-2, 2, 600)
    top.plot(x, np.cos(x ** 3), color=BLUE, linewidth=2, label="cos(x³)")
    top.plot(x, np.cos(x) ** 3, color=ORANGE, linewidth=2, label="cos³ x = (cos x)³")
    axes_lines(top)
    top.set_ylim(-1.3, 2.2)
    top.legend(loc="upper left", fontsize=9, ncol=2)
    top.grid(True)
    top.set_title("Order matters: cos(x³) vs cos³ x")
    bottom.plot(x, -3 * x ** 2 * np.sin(x ** 3), color=BLUE, linewidth=2, label="−3x² sin(x³)")
    bottom.plot(x, -3 * np.cos(x) ** 2 * np.sin(x), color=ORANGE, linewidth=2,
                label="−3 cos² x sin x")
    axes_lines(bottom)
    bottom.set_ylim(-12, 19)
    bottom.set_xlabel("x")
    bottom.legend(loc="upper left", fontsize=9, ncol=2)
    bottom.grid(True)

    ax = fig.add_subplot(gs[1, 1])
    xl = np.linspace(0.35, pi - 0.35, 500)
    cot = 1 / np.tan(xl)
    Fl = np.cos(np.sin(cot))
    dFl = np.sin(np.sin(cot)) * np.cos(cot) / np.sin(xl) ** 2
    probe = np.linspace(0.45, pi - 0.45, 9)
    num = fwd_diff(lambda t: np.cos(np.sin(1 / np.tan(t))), probe, 1e-6)
    ax.plot(xl, Fl, color=INK, linewidth=2, label="F = cos(sin(cot x))")
    ax.plot(xl, dFl, color=BLUE, linewidth=2, label="chain-rule F′")
    ax.plot(probe, num, "o", color=ORANGE, markersize=7, label="finite differences")
    axes_lines(ax, x=False)
    ax.set_xlabel("x")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Three layers: apply the rule three times")

    panel(fig.add_subplot(gs[1, 2]), "Outside → inside", [
        "1. Differentiate the outer function,",
        "    keeping the inside unchanged.",
        "2. Multiply by the derivative of the inside.",
        "3. Repeat for every layer.",
        "",
        r"$((3x^2+1)^5)' = 5(3x^2+1)^4 \cdot 6x$",
        r"$(\cos(x^3))' = -\sin(x^3)\cdot 3x^2$",
        r"$(\cos^3 x)' = 3\cos^2 x \cdot(-\sin x)$",
        r"$(\cos\sin\cot x)' = \frac{\sin(\sin\cot x)\,\cos(\cot x)}{\sin^2 x}$",
    ], spacing=0.095)
    save(fig, name="03_chain_rule.png")


# ================================================================ notebook 04
def sheet_04_implicit():
    xs = sp.symbols("x")
    y = sp.Function("y")(xs)
    d1 = sp.solve(sp.diff(xs ** 2 + y ** 2 - 25, xs), sp.diff(y, xs))[0]
    assert sp.simplify(d1 + xs / y) == 0
    d2 = sp.solve(sp.diff(2 * xs ** 3 + xs ** 3 * y + y ** 3 - 8, xs), sp.diff(y, xs))[0]
    assert sp.simplify(d2 - (-6 * xs ** 2 - 3 * xs ** 2 * y) / (xs ** 3 + 3 * y ** 2)) == 0
    assert -3 / 4 == -0.75

    fig = new_figure("Implicit differentiation: differentiate both sides, y is a function of x",
                     "d/dx (y²) = 2y·y′ by the chain rule.   Circle x² + y² = 25:  2x + 2y·y′ = 0  ⇒  y′ = −x/y")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * pi, 400)
    ax.plot(5 * np.cos(theta), 5 * np.sin(theta), color=INK, linewidth=2)
    placements = {(3, 4): ((8, 8), "left"), (-4, 3): ((10, -4), "left"),
                  (0, -5): ((0, 18), "center"), (4.8, -1.4): ((-10, -4), "right")}
    for (px, label_y), (offset, ha) in placements.items():
        py = label_y if (px, label_y) != (4.8, -1.4) else -sqrt(25 - 4.8 ** 2)
        m = -px / py + 0.0
        color = ORANGE if (px, py) == (3, 4) else BLUE
        tangent(ax, px, py, m, 1.4 / sqrt(1 + m * m), color=color)
        text = f"({px:g}, {label_y:g})  y′ = {m:.2f}".replace("-", "−")
        ax.annotate(text, (px, py), textcoords="offset points", xytext=offset, ha=ha,
                    va="center", fontsize=9.5, color=color)
    axes_lines(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("Tangents with slope −x/y on x² + y² = 25")

    ax = fig.add_subplot(gs[0, 1])
    xv = np.linspace(-4.95, 4.95, 500)
    upper = np.sqrt(25 - xv ** 2)
    explicit = -xv / np.sqrt(25 - xv ** 2)
    probe = np.array([-4.5, -3, -1.5, 0, 1.5, 3, 4.5])
    ax.plot(xv, explicit, color=BLUE, linewidth=2, label="explicit: d/dx √(25 − x²)")
    ax.plot(probe, -probe / np.sqrt(25 - probe ** 2), "o", color=ORANGE, markersize=8,
            label="implicit: −x/y on the upper half")
    ax.plot(3, -0.75, "o", color=ORANGE, markersize=12, markerfacecolor="none", markeredgewidth=2)
    ax.annotate("(3, 4): −3/4", (3, -0.75), textcoords="offset points", xytext=(-80, -30),
                fontsize=10, color=ORANGE, weight="bold")
    axes_lines(ax)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("dy/dx")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Same answer, without solving for y")
    assert np.allclose(-xv / upper, explicit)

    panel(fig.add_subplot(gs[0, 2]), "The recipe", [
        "1. Differentiate every term with respect to x.",
        "2. Treat y as y(x): each y-term gets a y′.",
        "3. Product rule where x and y multiply.",
        "4. Collect the y′ terms and solve for y′.",
        "",
        r"$\frac{d}{dx}(y^3) = 3y^2 y' \qquad \frac{d}{dx}(\sin y) = \cos y\; y'$",
        "",
        "The slope depends on the point (x, y),",
        "not on x alone: the circle has two points",
        "for every |x| < 5, with opposite slopes.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    gx, gy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3.5, 500))
    ax.contour(gx, gy, 2 * gx ** 3 + gx ** 3 * gy + gy ** 3 - 8, levels=[0], colors=INK,
               linewidths=2)
    # point on the curve at x = 1: y^3 + y - 6 = 0
    roots = np.roots([1, 0, 1, -6])
    y1 = float(roots[np.isclose(roots.imag, 0)].real[0])
    slope1 = (-6 - 3 * y1) / (1 + 3 * y1 ** 2)
    # check against the slope of the solved branch y(x) near x = 1

    def branch(xv_):
        r = np.roots([1, 0, xv_ ** 3, 2 * xv_ ** 3 - 8])
        return float(r[np.isclose(r.imag, 0)].real.max())
    assert abs(fwd_diff(branch, 1.0, 1e-6) - slope1) < 1e-4
    tangent(ax, 1.0, y1, slope1, 0.9)
    tangent(ax, 0.0, 2.0, 0.0, 0.9, color=BLUE)
    ax.annotate(f"(1, {y1:.3f}):  y′ = {slope1:.3f}", (1, y1), textcoords="offset points",
                xytext=(10, 6), fontsize=10, color=ORANGE, weight="bold")
    ax.annotate("(0, 2):  y′ = 0", (0, 2), textcoords="offset points", xytext=(-14, -22), ha="right",
                fontsize=10, color=BLUE, weight="bold")
    axes_lines(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("2x³ + x³y + y³ = 8: no need to solve for y")

    ax = fig.add_subplot(gs[1, 1])
    xh = np.linspace(0.25, 6, 400)
    for k, color in [(0, BLUE), (1, AQUA)]:
        c = pi / 2 + k * pi
        ax.plot(xh, c / xh, color=color, linewidth=2, label="xy = 3π/2" if k else "xy = π/2")
    xp = 1.5
    yp = (pi / 2) / xp
    tangent(ax, xp, yp, -yp / xp, 0.9, label=f"y′ = −y/x = {-yp / xp:.3f}")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("cos(xy) = 0: hyperbolas, y′ = −y/x")

    panel(fig.add_subplot(gs[1, 2]), "Second example, step by step", [
        r"$2x^3 + x^3 y + y^3 = 8$",
        r"$6x^2 + 3x^2 y + x^3 y' + 3y^2 y' = 0$",
        r"$y' = \frac{-6x^2 - 3x^2 y}{x^3 + 3y^2}$",
        "",
        "",
        f"At x = 1 the curve has y = {y1:.4f}; the formula",
        f"gives y′ = {slope1:.4f}, matching the slope of",
        "the numerically solved branch.",
        "",
        r"$\cos(xy) = 0:\ -\sin(xy)(xy' + y) = 0$;",
        "sin(xy) = ±1 there, so y′ = −y/x.",
    ], spacing=0.088)
    save(fig, name="04_implicit_differentiation.png")


# ================================================================ notebook 05
def sheet_05_logarithms():
    xs = sp.symbols("x", positive=True)
    assert sp.diff(sp.log(xs), xs) == 1 / xs
    assert sp.simplify(sp.diff(sp.log(sp.sin(5 * xs)), xs) - 5 / sp.tan(5 * xs)) == 0
    assert sp.simplify(sp.diff(sp.log(xs ** 3, 20), xs) - 3 / (xs * sp.log(20))) == 0
    assert sp.simplify(sp.diff(xs ** xs, xs) - xs ** xs * (sp.log(xs) + 1)) == 0
    x_min = exp(-1)
    f_min = x_min ** x_min

    fig = new_figure("ln x has slope 1/x, and logarithms turn hard derivatives into easy ones",
                     "(ln x)′ = 1/x,   (logₐ x)′ = 1/(x ln a),   (ln g(x))′ = g′(x)/g(x),   "
                     "(xˣ)′ = xˣ (ln x + 1)")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.1, 6, 600)
    ax.plot(x, np.log(x), color=INK, linewidth=2, label="ln x")
    ax.plot(x, 1 / x, color=BLUE, linewidth=2, label="(ln x)′ = 1/x")
    for p in (0.5, 1, 2, 4):
        tangent(ax, p, log(p), 1 / p, 0.45 if p < 1 else 0.8)
        ax.plot(p, 1 / p, "o", color=ORANGE, markersize=6, zorder=4)
        ax.plot([p, p], [log(p), 1 / p], color=ORANGE, linewidth=0.8, linestyle=":")
    ax.text(1.3, 3.5, "tangent slopes at x = ½, 1, 2, 4:\n2, 1, ½, ¼ — the height of 1/x",
            fontsize=10, color=ORANGE, va="top", weight="bold")
    axes_lines(ax)
    ax.set_ylim(-2.5, 4)
    ax.set_xlabel("x")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)
    ax.set_title("ln x climbs ever more slowly")

    ax = fig.add_subplot(gs[0, 1])
    for a, color, name in [(2, ORANGE, "2"), (e, BLUE, "e"), (10, AQUA, "10")]:
        ax.plot(x, 1 / (x * log(a)), color=color, linewidth=2,
                label=f"a = {name}: 1/(x ln a), at x = 1: {1 / log(a):.3f}")
    ax.set_ylim(0, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("(logₐ x)′")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("logₐ x = ln x / ln a: same shape, scaled")

    panel(fig.add_subplot(gs[0, 2]), "Chain rule with ln", [
        r"$\frac{d}{dx}\ln(\sin 5x) = \frac{5\cos 5x}{\sin 5x} = 5\cot 5x$",
        "",
        r"$\frac{d}{dx}\log_{20}(x^3) = \frac{3x^2}{x^3 \ln 20} = \frac{3}{x \ln 20}$",
        "",
        r"$\frac{d}{dx}\ln(x^2+1) = \frac{2x}{x^2+1}$",
        "",
        "Domains: ln x and logₐ x need x > 0;",
        "the base needs a > 0, a ≠ 1.",
    ], spacing=0.105)

    ax = fig.add_subplot(gs[1, 0])
    xv = np.linspace(0.2, 3, 500)
    yv = xv ** xv
    ax.plot(xv, yv, color=INK, linewidth=2, label="xˣ")
    ax.plot(xv, yv * (np.log(xv) + 1), color=BLUE, linewidth=2, label="xˣ (ln x + 1)")
    tangent(ax, x_min, f_min, 0, 0.35)
    ax.annotate(f"x = 1/e ≈ {x_min:.3f}: slope 0,\nminimum xˣ ≈ {f_min:.3f}", (x_min, f_min),
                xytext=(0.1, 5.2), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    axes_lines(ax, x=False)
    ax.set_ylim(-1.5, 10)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("xˣ: variable in base and exponent")

    ax = fig.add_subplot(gs[1, 1])
    probe = np.linspace(0.25, 3, 12)
    log_slope = fwd_diff(lambda t: t ** t, probe, 1e-7) / probe ** probe
    ax.plot(xv, np.log(xv) + 1, color=BLUE, linewidth=2, label="ln x + 1")
    ax.plot(probe, log_slope, "o", color=ORANGE, markersize=8,
            label="f′/f from finite differences")
    axes_lines(ax, x=False)
    ax.set_xlabel("x")
    ax.set_ylabel("f′(x) / f(x)  =  (ln f)′")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("ln f = x ln x makes the slope easy")
    assert np.allclose(log_slope, np.log(probe) + 1, atol=1e-5)

    panel(fig.add_subplot(gs[1, 2]), "Logarithmic differentiation", [
        r"$f = x^x \;\Rightarrow\; \ln f = x \ln x$",
        r"$\frac{f'}{f} = \ln x + x\cdot\frac{1}{x} = \ln x + 1$",
        r"$f' = x^x(\ln x + 1)$",
        "",
        "Also for (x² + 1)ˣ:",
        r"$f' = (x^2+1)^x\left[\ln(x^2+1) + \frac{2x^2}{x^2+1}\right]$",
        "",
        "Logs turn products into sums and powers",
        "into coefficients.",
    ], spacing=0.098)
    save(fig, name="05_log_derivatives.png")


# ================================================================ notebook 06
def sheet_06_exponentials():
    xs = sp.symbols("x", real=True)
    assert sp.simplify(sp.diff(8 ** xs, xs) - 8 ** xs * sp.log(8)) == 0
    assert sp.simplify(sp.diff(3 ** (xs ** 2), xs) - 2 * xs * 3 ** (xs ** 2) * sp.log(3)) == 0
    assert sp.simplify(sp.diff(sp.tan(3 ** xs), xs) - 3 ** xs * sp.log(3) / sp.cos(3 ** xs) ** 2) == 0

    fig = new_figure("Exponentials: the slope of aˣ is aˣ itself times ln a",
                     "(aˣ)′ = aˣ ln a,   (eˣ)′ = eˣ,   " + r"$(a^{g(x)})' = a^{g(x)} \ln a \cdot g'(x)$"
                     + ",   since " + r"$a^x = e^{x \ln a}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-3, 3, 500)
    ax.plot(x, 2 ** x, color=INK, linewidth=2, label="2ˣ")
    ax.plot(x, 2 ** x * np.log(2), color=BLUE, linewidth=2, label=f"(2ˣ)′ = 2ˣ ln 2 = {log(2):.3f}·2ˣ")
    ax.plot(x, np.exp(x), color=ORANGE, linewidth=2, linestyle="--", label="eˣ = (eˣ)′, one curve")
    ax.set_ylim(0, 9)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Same shape, scaled by ln a")

    ax = fig.add_subplot(gs[0, 1])
    bases = np.linspace(1.2, 10.5, 300)
    ax.plot(bases, np.log(bases), color=BLUE, linewidth=2, label="ln a")
    probe = np.array([2, e, 8, 10])
    est = (probe ** 1e-6 - 1) / 1e-6
    assert np.allclose(est, np.log(probe), atol=1e-5)
    ax.plot(probe, est, "o", color=ORANGE, markersize=8, label="(a^h − 1)/h, h = 10⁻⁶")
    for a, v, name, offset in zip(probe, est, ["2", "e", "8", "10"],
                                  [(8, -14), (-10, 8), (-10, 6), (0, -24)]):
        ax.annotate(f"a = {name}: {v:.3f}", (a, v), textcoords="offset points", xytext=offset,
                    ha="left" if offset[0] > 0 else "right" if offset[0] < 0 else "center",
                    fontsize=9.5)
    ax.axhline(1, color=MUTED, linestyle=":", linewidth=1.3)
    ax.set_xlabel("base a")
    ax.set_ylabel("slope of aˣ at x = 0")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True)
    ax.set_title("Slope at 0 is ln a, and equals 1 only for e")

    panel(fig.add_subplot(gs[0, 2]), "Why ln a appears", [
        r"$a^x = e^{x \ln a}$",
        r"$(a^x)' = e^{x \ln a}\cdot \ln a = a^x \ln a$",
        "",
        r"$\frac{a^{x+h} - a^x}{h} = a^x \cdot \frac{a^h - 1}{h}$",
        "",
        "The slope is the current value times a",
        "constant: rate proportional to amount,",
        "the signature of growth and decay.",
        "",
        "For a = e that constant is 1.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(-3, 2.2, 400)
    ax.plot(x, np.exp(x), color=INK, linewidth=2, label="eˣ")
    for p in (-1, 0, 1):
        xs_t = np.array([p - 1, p + 0.35])
        ax.plot(xs_t, exp(p) + exp(p) * (xs_t - p), color=ORANGE, linewidth=1.8)
        ax.plot(p, exp(p), "o", color=ORANGE, markersize=6, zorder=4)
        ax.plot(p - 1, 0, "o", color="white", markeredgecolor=ORANGE, markeredgewidth=1.8,
                markersize=7, zorder=4)
        ax.annotate("", xy=(p - 1, -0.35), xytext=(p, -0.35),
                    arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1))
    ax.text(-2.9, 3.6, "Tangent at x = p has slope eᵖ = its height,\nso it meets the x-axis at p − 1:\n"
            "always exactly 1 to the left.", fontsize=10, color=ORANGE, va="top")
    axes_lines(ax, x=False)
    ax.set_xlim(-3, 2)
    ax.set_ylim(-0.6, 4.5)
    ax.set_xlabel("x")
    ax.legend(loc="center left", fontsize=10)
    ax.grid(True)
    ax.set_title("eˣ: slope = height at every point")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-1.5, 1.5, 400)
    ax.plot(x, 3 ** (x ** 2), color=INK, linewidth=2, label=r"$h(x) = 3^{x^2}$")
    ax.plot(x, 2 * x * 3 ** (x ** 2) * np.log(3), color=BLUE, linewidth=2,
            label=r"$h'(x) = 3^{x^2} \ln 3 \cdot 2x$")
    axes_lines(ax)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Chain rule with the exponent g(x) = x²")

    panel(fig.add_subplot(gs[1, 2]), "Worked examples", [
        r"$(8^x)' = 8^x \ln 8$",
        r"$(3^{x^2})' = 3^{x^2}\ln 3 \cdot 2x$",
        r"$(\tan 3^x)' = \frac{3^x \ln 3}{\cos^2(3^x)}$",
        r"$(10^{2x})' = 10^{2x}\ln 10 \cdot 2$",
        r"$(x\,2^x)' = 2^x + x\,2^x \ln 2$",
        r"$(e^{x^2})' = e^{x^2}\cdot 2x$",
        "",
        "Exponentials model interest, populations,",
        "epidemics and radioactive decay.",
    ], spacing=0.095)
    save(fig, name="06_exp_derivatives.png")


# ================================================================ notebook 07
def sheet_07_inverse():
    ys = sp.symbols("y", positive=True)
    assert sp.diff(sp.sqrt(ys), ys) == 1 / (2 * sp.sqrt(ys))
    a = 1.5
    b = a ** 2
    slope_f, slope_inverse = 2 * a, 1 / (2 * a)
    assert (b, slope_f) == (2.25, 3.0) and abs(slope_inverse - 1 / 3) < 1e-15

    fig = new_figure("Inverse functions: reflect the graph, take the reciprocal slope",
                     "(f⁻¹)′(f(a)) = 1 / f′(a).   Reflection in y = x swaps Δx and Δy, "
                     "so the slope Δy/Δx becomes Δx/Δy.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 4, 500)
    ax.plot(x, x ** 2, color=BLUE, linewidth=2, label="f(x) = x²,  x ≥ 0")
    ax.plot(x, np.sqrt(x), color=AQUA, linewidth=2, label="f⁻¹(x) = √x")
    ax.plot([0, 4], [0, 4], color=MUTED, linestyle="--", linewidth=1.3, label="y = x")
    for px in (0.5, 1, 1.5):
        py = px ** 2
        ax.plot([px, py], [py, px], ":", color=ORANGE, linewidth=1.3)
        ax.plot(px, py, "o", color=BLUE, markersize=7)
        ax.plot(py, px, "o", color=AQUA, markersize=7)
    ax.annotate("(1/2, 1/4) ↔ (1/4, 1/2)", (0.5, 0.25), xytext=(1.25, 0.3), fontsize=9.5,
                va="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.annotate("(3/2, 9/4) ↔ (9/4, 3/2)", (2.25, 1.5), textcoords="offset points", xytext=(10, -18),
                fontsize=9.5)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Points (a, b) and (b, a) mirror in y = x")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, x ** 2, color=BLUE, linewidth=2)
    ax.plot(x, np.sqrt(x), color=AQUA, linewidth=2)
    ax.plot([0, 4], [0, 4], color=MUTED, linestyle="--", linewidth=1.3)
    tangent(ax, a, b, slope_f, 0.55, color=BLUE, label=f"tangent to f at (1.5, 2.25): slope {slope_f:g}")
    tangent(ax, b, a, slope_inverse, 1.4, color=AQUA,
            label="tangent to f⁻¹ at (2.25, 1.5): slope 1/3")
    # slope triangles: run 0.3 / rise 0.9 on f, run 0.9 / rise 0.3 on the inverse
    ax.plot([a, a + 0.3, a + 0.3], [b, b, b + 0.9], color=ORANGE, linewidth=1.6)
    ax.text(a + 0.36, b - 0.05, "Δx = 0.3", fontsize=9, color=ORANGE, va="top")
    ax.text(a + 0.36, b + 0.45, "Δy = 0.9", fontsize=9, color=ORANGE, va="center")
    ax.plot([b, b, b + 0.9], [a, a + 0.3, a + 0.3], color=ORANGE, linewidth=1.6)
    ax.text(b + 0.45, a + 0.36, "Δx = 0.9", fontsize=9, color=ORANGE, ha="center")
    ax.text(b + 0.05, a - 0.1, "Δy = 0.3", fontsize=9, color=ORANGE, va="top")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)
    ax.set_title("Mirrored tangents: slope 3 becomes 1/3")

    panel(fig.add_subplot(gs[0, 2]), "Why it is true", [
        r"$f^{-1}(f(x)) = x$",
        "Differentiate both sides (chain rule):",
        r"$(f^{-1})'(f(x)) \cdot f'(x) = 1$",
        r"$(f^{-1})'(f(x)) = \frac{1}{f'(x)}$",
        "",
        r"With $f = x^2$:  $(f^{-1})'(a^2) = \frac{1}{2a}$,",
        r"so  $\frac{d}{dy}\sqrt{y} = \frac{1}{2\sqrt{y}}$.",
        "",
        "Needs: f invertible near a, differentiable,",
        "and f′(a) ≠ 0.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    yv = np.linspace(0.05, 4.5, 400)
    ax.plot(yv, 1 / (2 * np.sqrt(yv)), color=AQUA, linewidth=2, label="(√y)′ = 1/(2√y)")
    pts = [(0.5, 0.25), (1.5, 2.25), (2, 4)]
    for a_pt, y_pt in pts:
        v = 1 / (2 * a_pt)
        ax.plot(y_pt, v, "o", color=ORANGE, markersize=8, zorder=4)
        ax.annotate(f"y = {y_pt:g}\n1/f′({a_pt:g}) = 1/{2 * a_pt:g}", (y_pt, v),
                    textcoords="offset points", xytext=(10, 6) if y_pt < 1 else (0, 12),
                    ha="left" if y_pt < 1 else "center", fontsize=10)
    ax.set_xlim(0, 4.8)
    ax.set_ylim(0, 2.3)
    ax.set_xlabel("y")
    ax.set_ylabel("slope of the inverse")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Reciprocal slopes, point by point")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-2, 3, 400)
    xl = np.linspace(0.05, 3.5, 400)
    ax.plot(x, np.exp(x), color=BLUE, linewidth=2, label="eˣ")
    ax.plot(xl, np.log(xl), color=AQUA, linewidth=2, label="ln x")
    ax.plot([-2, 3.5], [-2, 3.5], color=MUTED, linestyle="--", linewidth=1.3)
    tangent(ax, 1, e, e, 0.4, color=BLUE, label=f"slope e ≈ {e:.3f} at (1, e)")
    tangent(ax, e, 1, 1 / e, 1.0, color=AQUA, label=f"slope 1/e ≈ {1 / e:.3f} at (e, 1)")
    ax.set_xlim(-2, 3.5)
    ax.set_ylim(-2, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("eˣ and ln x: (ln y)′ = 1/eᵃ = 1/y")

    panel(fig.add_subplot(gs[1, 2]), "Checks", [
        "f = x²:   (f⁻¹)′(4) = 1/f′(2) = 1/4",
        "f = x³:   (f⁻¹)′(8) = 1/f′(2) = 1/12",
        "f = 2x + 5:   (f⁻¹)′ = 1/2",
        "f = eˣ:   (ln y)′ = 1/y",
        "",
        "Where f′(a) = 0 the reciprocal is undefined:",
        "a horizontal tangent of f at (0, 0) is a",
        "vertical tangent of √x at (0, 0).",
    ], spacing=0.09)
    save(fig, name="07_inverse_derivative.png")


# ================================================================ notebook 08
def sheet_08_non_differentiable():
    h_values = [1, 0.1, 0.01, -0.01, -0.1, -1]
    quotients = [(abs(h) - abs(0)) / h for h in h_values]
    assert quotients == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]

    fig = new_figure("Differentiable means one slope: corners, cusps, vertical tangents and jumps fail",
                     "f′(a) exists only when both one-sided limits of (f(a+h) − f(a))/h agree.   "
                     "For |x| at 0:  |h|/h = −1 from the left, +1 from the right.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-2, 2, 400)
    ax.plot(x, np.abs(x), color=INK, linewidth=2.2, label="f(x) = |x|")
    for h, color in [(1.5, AQUA), (0.6, AQUA), (-1.5, ORANGE), (-0.6, ORANGE)]:
        ax.plot([0, h], [0, abs(h)], "o-", color=color, linewidth=1.6, markersize=6)
    ax.text(1.1, 0.4, "right secants:\nslope +1", color=AQUA, fontsize=10.5, weight="bold")
    ax.text(-1.9, 0.4, "left secants:\nslope −1", color=ORANGE, fontsize=10.5, weight="bold")
    axes_lines(ax)
    ax.set_ylim(-0.3, 2.1)
    ax.set_xlabel("x")
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True)
    ax.set_title("At the corner, secants never agree")

    ax = fig.add_subplot(gs[0, 1])
    hc = np.linspace(-1, 1, 401)
    hc = hc[hc != 0]
    ax.plot(hc[hc > 0], np.ones((hc > 0).sum()), color=AQUA, linewidth=2.5, label="|h|/h,  h > 0")
    ax.plot(hc[hc < 0], -np.ones((hc < 0).sum()), color=ORANGE, linewidth=2.5, label="|h|/h,  h < 0")
    ax.plot(h_values, quotients, "o", color=INK, markersize=7, label="notebook's h values")
    ax.plot(hc, hc, color=BLUE, linewidth=1.6, linestyle="--", label="x² at 0: quotient h → 0")
    ax.plot([0, 0], [1, -1], "o", color="white", markeredgecolor=INK, markersize=9, zorder=5)
    axes_lines(ax)
    ax.set_ylim(-1.6, 1.8)
    ax.set_xlabel("h")
    ax.set_ylabel("difference quotient at 0")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True)
    ax.set_title("Two limits (−1 and +1) instead of one")

    ax = fig.add_subplot(gs[0, 2])
    x = np.linspace(-4, 4, 400)
    ax.plot(x, np.abs(x), color=INK, linewidth=2.5, label="|x|")
    for m, color in zip([-1, -0.5, 0, 0.5, 1], [ORANGE, ORANGE_LIGHT, MUTED, AQUA_LIGHT, AQUA]):
        ax.plot(x, m * x, color=color, linewidth=1.5, label=f"slope {m:g}")
    ax.plot(0, 0, "o", color=INK, markersize=7)
    axes_lines(ax)
    ax.set_ylim(-2, 6.2)
    ax.set_xlabel("x")
    ax.legend(loc="upper center", fontsize=9, ncol=3)
    ax.grid(True)
    ax.set_title("Lines through the corner: none is the tangent")

    sub = gs[1, 0].subgridspec(2, 2, hspace=0.55, wspace=0.3)
    x = np.linspace(-3, 3, 500)
    cases = [("corner: |x|", np.abs(x)), (r"cusp: $|x|^{2/3}$", np.abs(x) ** (2 / 3)),
             ("vertical tangent: ∛x", np.cbrt(x)), ("jump: 1 then 2", None)]
    for k, (name, values) in enumerate(cases):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        if values is None:
            ax.plot(x[x < 0], np.ones((x < 0).sum()), color=BLUE, linewidth=2)
            ax.plot(x[x >= 0], 2 * np.ones((x >= 0).sum()), color=BLUE, linewidth=2)
            ax.plot(0, 1, "o", color="white", markeredgecolor=BLUE, markersize=6, zorder=4)
            ax.plot(0, 2, "o", color=BLUE, markersize=6, zorder=4)
            ax.set_ylim(0, 2.6)
        else:
            ax.plot(x, values, color=BLUE, linewidth=2)
            ax.plot(0, 0, "o", color=ORANGE, markersize=6, zorder=4)
        axes_lines(ax)
        ax.set_title(name, fontsize=10.5)
        ax.tick_params(labelsize=8)
        ax.grid(True)

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-0.2, 2.2, 400)
    piece = np.where(x < 1, x ** 2, x)
    ax.plot(x[x < 1], piece[x < 1], color=BLUE, linewidth=2.2, label="x²  for x < 1")
    ax.plot(x[x >= 1], piece[x >= 1], color=INK, linewidth=2.2, label="x  for x ≥ 1")
    tangent(ax, 1, 1, 2, 0.5, color=ORANGE, ls="--", label="left slope 2x = 2")
    tangent(ax, 1, 1, 1, 0.5, color=AQUA, ls="--", label="right slope 1", dot=False)
    ax.set_ylim(-0.2, 2.4)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Continuous at 1, yet 2 ≠ 1: no derivative")

    panel(fig.add_subplot(gs[1, 2]), "Continuity vs differentiability", [
        "differentiable  ⇒  continuous",
        "continuous  ⇏  differentiable  (|x| at 0)",
        "",
        "One-sided derivatives of |x| at 0:",
        "left −1,  right +1  →  f′(0) does not exist.",
        "",
        "Zoom in on a smooth point: it looks like a",
        "line. Zoom in on a corner: still two lines.",
        "",
        "|x − 2| fails at x = 2,  |x − 3| at x = 3.",
    ], spacing=0.088)
    save(fig, name="08_non_differentiable.png")


# ================================================================ notebook 09
def sheet_09_higher_order():
    xs, ts = sp.symbols("x t")
    f = 6 * xs ** 3 + 2 * xs + 8
    assert sp.diff(f, xs) == 18 * xs ** 2 + 2 and sp.diff(f, xs, 2) == 36 * xs
    s = ts ** 4 + 5 * ts - 7
    assert sp.diff(s, ts, 2).subs(ts, 8) == 768
    chain = [xs ** 5]
    for _ in range(6):
        chain.append(sp.diff(chain[-1], xs))
    assert chain[1:] == [5 * xs ** 4, 20 * xs ** 3, 60 * xs ** 2, 120 * xs, 120, 0]
    at_one = [int(c.subs(xs, 1)) for c in chain]

    fig = new_figure("Higher derivatives: differentiate the derivative again",
                     "f″ = (f′)′ = d²y/dx²,  f⁽ⁿ⁾ = n-th derivative.   Position → velocity → acceleration;  "
                     "the sign of f″ gives concavity.")
    gs = grid(fig, top=0.84, wspace=0.3)

    sub = gs[0, 0].subgridspec(3, 1, hspace=0.12)
    t = np.linspace(0, 5, 400)
    rows = [("s(t) = t⁴ + 5t − 7", t ** 4 + 5 * t - 7, INK, "m"),
            ("v = s′(t) = 4t³ + 5", 4 * t ** 3 + 5, BLUE, "m/s"),
            ("a = s″(t) = 12t²", 12 * t ** 2, ORANGE, "m/s²")]
    first = None
    for k, (name, values, color, unit) in enumerate(rows):
        ax = fig.add_subplot(sub[k], sharex=first)
        first = first or ax
        ax.plot(t, values, color=color, linewidth=2, label=name)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(True)
        if k == 0:
            ax.set_title("Each curve is the slope of the one above")
        if k < 2:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("time t (s)")

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(-2, 2, 400)
    ax.plot(x, x ** 3, color=INK, linewidth=2, label="f(x) = x³")
    ax.plot(x, 6 * x, color=BLUE, linewidth=2, linestyle="--", label="f″(x) = 6x")
    ax.fill_between(x, -9, 9, where=x < 0, color=ORANGE, alpha=0.12, linewidth=0,
                    label="f″ < 0: concave down")
    ax.fill_between(x, -9, 9, where=x >= 0, color=AQUA, alpha=0.12, linewidth=0,
                    label="f″ > 0: concave up")
    ax.plot(0, 0, "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate("inflection point", (0, 0), textcoords="offset points", xytext=(10, -26), fontsize=10,
                color=ORANGE, weight="bold")
    ax.set_ylim(-9, 9)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("The sign of f″ is the bend")

    panel(fig.add_subplot(gs[0, 2]), "Notation", [
        "`order   prime     Leibniz`",
        "`1st     f'(x)     dy/dx`",
        "`2nd     f''(x)    d²y/dx²`",
        "`3rd     f'''(x)   d³y/dx³`",
        "`n-th    f^(n)(x)  d^n y/dx^n`",
        "",
        r"$\frac{d}{dx}\left(\frac{dy}{dx}\right) = \frac{d^2y}{dx^2}$",
        "",
        "",
        "From the 4th on, write f⁽⁴⁾ instead of primes.",
        "sp.diff(f, x, n) takes the order as argument.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    labels = ["x⁵", "5x⁴", "20x³", "60x²", "120x", "120", "0"]
    positions = np.arange(7)
    ax.bar(positions, at_one, width=0.62, color=[BLUE] * 5 + [ORANGE, MUTED], zorder=2)
    for p, v in zip(positions, at_one):
        ax.text(p, v + 3, str(v), ha="center", fontsize=10)
    ax.set_xticks(positions, [(f"$f^{{({n})}}$" if n else "$f$") + f"\n{lab}" for n, lab in zip(positions, labels)],
                  fontsize=9.5)
    ax.set_ylim(0, 140)
    ax.set_ylabel("value at x = 1")
    ax.grid(axis="y")
    ax.set_title("x⁵: five steps to a constant, then 0")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-1.5, 1.5, 400)
    ax.plot(x, 6 * x ** 3 + 2 * x + 8, color=INK, linewidth=2, label="f = 6x³ + 2x + 8  (degree 3)")
    ax.plot(x, 18 * x ** 2 + 2, color=BLUE, linewidth=2, label="f′ = 18x² + 2  (degree 2)")
    ax.plot(x, 36 * x, color=ORANGE, linewidth=2, label="f″ = 36x  (degree 1)")
    axes_lines(ax)
    ax.set_ylim(-45, 50)
    ax.set_xlabel("x")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Each derivative lowers the degree by one")

    panel(fig.add_subplot(gs[1, 2]), "Acceleration after 8 seconds", [
        r"$s(t) = t^4 + 5t - 7$",
        r"$v(t) = s'(t) = 4t^3 + 5$",
        r"$a(t) = s''(t) = 12t^2$",
        r"$a(8) = 12 \cdot 64 = 768\ \mathrm{m/s^2}$",
        "",
        "Second derivative = rate of change of the",
        "rate of change.",
        "",
        "Check: (x⁶)⁽⁶⁾ = 720 = 6!, the first constant.",
    ], spacing=0.092)
    save(fig, name="09_higher_derivatives.png")


# ================================================================ notebook 10
def sheet_10_local_extrema():
    xs = sp.symbols("x")
    f = sp.Rational(1, 3) * xs ** 3 - xs
    fp, fpp = sp.diff(f, xs), sp.diff(f, xs, 2)
    assert sp.solve(fp, xs) == [-1, 1]
    assert [f.subs(xs, c) for c in (-1, 1)] == [sp.Rational(2, 3), -sp.Rational(2, 3)]
    assert [fpp.subs(xs, c) for c in (-1, 1)] == [-2, 2]
    g = xs ** 4 - xs ** 2
    crit = sp.solve(sp.diff(g, xs), xs)
    assert np.allclose(sorted(float(c) for c in crit), [-1 / sqrt(2), 0.0, 1 / sqrt(2)])
    lo, hi = 0, 1.5
    candidates = sorted({round(float(c), 12) for c in crit if lo <= float(c) <= hi} | {lo, hi})
    values = [c ** 4 - c ** 2 for c in candidates]
    assert np.allclose(values, [0, -0.25, 2.8125])
    gmax = candidates[int(np.argmax(values))]
    gmin = candidates[int(np.argmin(values))]
    assert gmax == 1.5 and abs(gmin - 1 / sqrt(2)) < 1e-9

    fig = new_figure("Extrema: f′ = 0 finds candidates, f″ classifies, endpoints decide the global winner",
                     "Critical point: f′(x₀) = 0.   f″(x₀) > 0 → minimum,  f″(x₀) < 0 → maximum,  "
                     "f″(x₀) = 0 → test fails.")
    gs = grid(fig, top=0.84, wspace=0.3)

    top, bottom = stacked(fig, gs[0, 0])
    x = np.linspace(-2, 2, 400)
    top.plot(x, x ** 3 / 3 - x, color=INK, linewidth=2, label="f(x) = x³/3 − x")
    tangent(top, -1, 2 / 3, 0, 0.5, color=ORANGE)
    tangent(top, 1, -2 / 3, 0, 0.5, color=AQUA)
    top.annotate("max (−1, 2/3)", (-1, 2 / 3), textcoords="offset points", xytext=(-30, 10),
                 fontsize=10, color=ORANGE, weight="bold")
    top.annotate("min (1, −2/3)", (1, -2 / 3), textcoords="offset points", xytext=(-40, -22),
                 fontsize=10, color=AQUA, weight="bold")
    axes_lines(top)
    top.set_ylim(-1.5, 2.0)
    top.legend(loc="upper right", fontsize=9.5)
    top.grid(True)
    top.set_title("Flat tangents where f′ = 0")
    bottom.plot(x, x ** 2 - 1, color=BLUE, linewidth=2, label="f′(x) = x² − 1")
    sign_shading(bottom, x, x ** 2 - 1)
    bottom.plot([-1, 1], [0, 0], "o", color=INK, markersize=7)
    axes_lines(bottom, x=False)
    bottom.set_ylim(-1.5, 4.5)
    bottom.set_xlabel("x")
    bottom.legend(loc="upper center", fontsize=8.5)
    bottom.grid(True)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, 2 * x, color=BLUE, linewidth=2, label="f″(x) = 2x")
    ax.fill_between(x, 2 * x, 0, where=x < 0, color=ORANGE, alpha=0.2, linewidth=0)
    ax.fill_between(x, 2 * x, 0, where=x > 0, color=AQUA, alpha=0.2, linewidth=0)
    ax.plot(-1, -2, "o", color=ORANGE, markersize=10, zorder=5)
    ax.plot(1, 2, "o", color=AQUA, markersize=10, zorder=5)
    ax.annotate("f″(−1) = −2 < 0\n→ maximum (cap)", (-1, -2), textcoords="offset points",
                xytext=(12, -36), fontsize=10, color=ORANGE, weight="bold")
    ax.annotate("f″(1) = 2 > 0\n→ minimum (cup)", (1, 2), textcoords="offset points",
                xytext=(-120, 14), fontsize=10, color=AQUA, weight="bold")
    axes_lines(ax)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Second-derivative test at x = ±1")

    panel(fig.add_subplot(gs[0, 2]), "The two tests", [
        "First derivative: f′ > 0 increasing,",
        "f′ < 0 decreasing, f′ = 0 critical point.",
        "",
        "`f''(x0) > 0   local minimum`",
        "`f''(x0) < 0   local maximum`",
        "`f''(x0) = 0   test fails`",
        "",
        "x³ has f′(0) = f″(0) = 0 but no extremum:",
        "it rises on both sides (inflection point).",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(-1.3, 1.3, 400)
    r = 1 / sqrt(2)
    ax.plot(x, x ** 4 - x ** 2, color=INK, linewidth=2, label="f(x) = x⁴ − x²")
    ax.plot([-r, r], [-0.25, -0.25], "o", color=AQUA, markersize=9, label="local minima (±1/√2, −1/4)")
    ax.plot(0, 0, "o", color=ORANGE, markersize=9, label="local maximum (0, 0)")
    ax.axhline(0, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.annotate("f climbs past 0 out here:\nthe local max is not global", (1.2, 1.2 ** 4 - 1.2 ** 2),
                xytext=(0.95, 0.42), ha="right", fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(-0.45, 1.3)
    ax.set_xlabel("x")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True)
    ax.set_title("Local is not global")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-0.3, 1.6, 400)
    inside = (x >= lo) & (x <= hi)
    ax.plot(x, x ** 4 - x ** 2, color=MUTED, linewidth=1.5)
    ax.plot(x[inside], (x ** 4 - x ** 2)[inside], color=INK, linewidth=2.4,
            label="f = x⁴ − x² on [0, 1.5]")
    ax.axvspan(lo, hi, color=BLUE_LIGHT, alpha=0.25, linewidth=0)
    for c, v in zip(candidates, values):
        color = ORANGE if c == gmax else AQUA if c == gmin else BLUE
        ax.plot(c, v, "o", color=color, markersize=10, zorder=5)
        what = "global max" if c == gmax else "global min" if c == gmin else "endpoint + critical"
        offset = (-118, -8) if c == gmax else (12, -10) if c == gmin else (8, 8)
        ax.annotate(f"f({c:.3g}) = {v:.5g}\n{what}", (c, v), textcoords="offset points",
                    xytext=offset, va="top" if c == gmin else "baseline", fontsize=9.5, color=color,
                    weight="bold")
    ax.set_ylim(-0.95, 3.3)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Closed interval: compare candidates")

    panel(fig.add_subplot(gs[1, 2]), "The procedure on [a, b]", [
        "1. Compute f′(x).",
        "2. Solve f′(x) = 0; drop x outside [a, b].",
        "3. Classify with f″.",
        "4. Evaluate f(a) and f(b).",
        "5. Largest value = global max,",
        "    smallest = global min.",
        "",
        "Pen against a wall, 20 m of fence:",
        r"$A(x) = x(20 - 2x),\ A'(x) = 20 - 4x = 0$",
        "→ x = 5 m, A(5) = 50 m² (A″ = −4 < 0).",
    ], spacing=0.085)
    save(fig, name="10_extrema.png")


# ================================================================ notebook 11
def sheet_11_interpretation():
    def f(x):
        return x ** 2

    a = 1.0
    secant_x = [3.0, 2.5, 2.0, 1.5]
    secant_slopes = [(f(q) - f(a)) / (q - a) for q in secant_x]
    assert secant_slopes == [4.0, 3.5, 3.0, 2.5]
    avg_x = [3.0, 2.0, 1.5, 1.1, 1.01]
    avg_v = [(5 * q ** 2 - 5 * a ** 2) / (q - a) for q in avg_x]
    assert [f"{v:.3f}" for v in avg_v] == ["20.000", "15.000", "12.500", "10.500", "10.050"]

    def loss(w):
        return (w - 3) ** 2

    def loss_prime(w):
        return 2 * (w - 3)

    w, eta, history = 0.0, 0.1, [0.0]
    for _ in range(15):
        w = w - eta * loss_prime(w)
        history.append(w)
    assert f"{history[-1]:.4f}" == "2.8944" and f"{loss(history[-1]):.4f}" == "0.0111"
    pop = [100 * exp(0.5 * h) for h in (0, 2, 4)]
    rate = [50 * exp(0.5 * h) for h in (0, 2, 4)]
    assert [f"{p:.1f}" for p in pop] == ["100.0", "271.8", "738.9"]
    assert [f"{r:.1f}" for r in rate] == ["50.0", "135.9", "369.5"]

    fig = new_figure("f′(a): the tangent's slope, the velocity, the instantaneous rate of change",
                     "f′(a) = lim (f(x) − f(a))/(x − a) as x → a;   tangent line  y = f(a) + f′(a)(x − a).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-0.5, 3, 400)
    ax.plot(xs, f(xs), color=INK, linewidth=2.2, label="f(x) = x²")
    shades = [BLUE_LIGHT, "#86b6ef", BLUE, "#1c5cab"]
    for q, m, color in zip(secant_x, secant_slopes, shades):
        ax.plot(xs, f(a) + m * (xs - a), color=color, linewidth=1.4, linestyle="--",
                label=f"secant to x = {q:g}: slope {m:g}")
        ax.plot(q, f(q), "o", color=color, markersize=6)
    ax.plot(xs, 2 * xs - 1, color=ORANGE, linewidth=2.4, label="tangent y = 2x − 1")
    ax.plot(a, f(a), "o", color=ORANGE, markersize=9, zorder=5)
    ax.set_ylim(-1, 9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)
    ax.set_title("Secants → tangent at P = (1, 1)")

    top, bottom = stacked(fig, gs[0, 1])
    x = np.linspace(-2.2, 2.2, 400)
    top.plot(x, x ** 3 / 3 - x, color=INK, linewidth=2, label="f(x) = x³/3 − x")
    for p in (-1.5, -1, 0, 1, 1.5):
        color = MUTED if abs(p) == 1 else (AQUA if p ** 2 - 1 > 0 else ORANGE)
        tangent(top, p, p ** 3 / 3 - p, p ** 2 - 1, 0.45, color=color, lw=1.8)
    axes_lines(top)
    top.set_ylim(-1.7, 1.7)
    top.legend(loc="upper left", fontsize=9)
    top.grid(True)
    top.set_title("The slope tells the shape")
    bottom.plot(x, x ** 2 - 1, color=BLUE, linewidth=2, label="f′(x) = x² − 1")
    sign_shading(bottom, x, x ** 2 - 1)
    bottom.plot([-1, 1], [0, 0], "o", color=INK, markersize=7)
    axes_lines(bottom, x=False)
    bottom.set_ylim(-1.5, 4.2)
    bottom.set_xlabel("x")
    bottom.legend(loc="upper center", fontsize=8.5)
    bottom.grid(True)

    ax = fig.add_subplot(gs[0, 2])
    xc = np.linspace(1.005, 3.1, 300)
    ax.plot(xc, 5 * (xc + 1), color=BLUE, linewidth=2, label="average velocity over [1, x] = 5(x + 1)")
    ax.plot(avg_x, avg_v, "o", color=BLUE, markersize=8)
    for q, v in zip(avg_x[:4], avg_v[:4]):
        ax.annotate(f"{v:g} m/s", (q, v), textcoords="offset points", xytext=(6, -16), fontsize=9.5)
    ax.axhline(10, color=ORANGE, linewidth=2, linestyle="--", label="instantaneous v(1) = s′(1) = 10 m/s")
    ax.plot(1, 10, "o", color=ORANGE, markersize=9, zorder=5)
    ax.set_xlim(0.9, 3.4)
    ax.set_ylim(8, 22)
    ax.set_xlabel("end of the interval x (s)")
    ax.set_ylabel("m/s")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Falling body s(t) = 5t²: velocity")

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 6, 300)
    ax.plot(t, 100 * np.exp(0.5 * t), color=BLUE, linewidth=2,
            label=r"population $P(t) = 100\,e^{0.5t}$")
    ax.plot(t, 50 * np.exp(0.5 * t), color=ORANGE, linewidth=2,
            label=r"growth rate $P'(t) = 50\,e^{0.5t}$")
    for h, p, r in zip((0, 2, 4), pop, rate):
        ax.plot([h, h], [r, p], color=MUTED, linewidth=1)
        ax.plot(h, p, "o", color=BLUE, markersize=6)
        ax.plot(h, r, "o", color=ORANGE, markersize=6)
    table = ["t (h)   P(t)    P'(t)"] + [f"{h:>4}  {p:>6.1f}  {r:>6.1f}" for h, p, r in zip((0, 2, 4), pop, rate)]
    ax.text(0.04, 0.78, "\n".join(table), transform=ax.transAxes, family="monospace", fontsize=9.5,
            va="top", color=INK2)
    ax.set_xlabel("time t (hours)")
    ax.set_ylabel("individuals  /  individuals per hour")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Bacteria: the rate is always ½ of P")

    ax = fig.add_subplot(gs[1, 1])
    ws = np.linspace(-1, 7, 200)
    hist = np.array(history)
    ax.plot(ws, loss(ws), color=INK, linewidth=2, label="L(w) = (w − 3)²")
    ax.plot(hist, loss(hist), "o-", color=ORANGE, markersize=6, linewidth=1.2,
            label="w ← w − 0.1·L′(w), from w = 0")
    ax.plot(3, 0, "o", color=AQUA, markersize=10, zorder=5, label="minimum w = 3")
    tangent(ax, 0, 9, loss_prime(0), 0.6, color=BLUE, label="L′(0) = −6: steep, big step", dot=False)
    ax.text(3.4, 7.8, f"after 15 steps:\nw = {hist[-1]:.4f}\nL = {loss(hist[-1]):.4f}", fontsize=9.5,
            color=ORANGE, weight="bold")
    ax.set_ylim(-1, 17)
    ax.set_xlabel("parameter w")
    ax.set_ylabel("loss L(w)")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True)
    ax.set_title("Gradient descent: steps shrink as L′ → 0")

    panel(fig.add_subplot(gs[1, 2]), "One idea, many fields", [
        "Average rate over [a, x]: secant slope.",
        "Instantaneous rate at a: f′(a).",
        "",
        "`physics      position   -> velocity`",
        "`biology      population -> growth rate`",
        "`economics    cost       -> marginal cost`",
        "`electricity  charge     -> current`",
        "`ML           loss       -> descent step`",
        "",
        "At a max or min of f: f′(a) = 0.",
        "Tangent to x² − 4x at 1:  y = −2x − 1.",
    ], spacing=0.083)
    save(fig, name="11_derivative_meaning.png")


if __name__ == "__main__":
    sheet_00_introduction()
    sheet_01_polynomials()
    sheet_02_trigonometric()
    sheet_03_chain_rule()
    sheet_04_implicit()
    sheet_05_logarithms()
    sheet_06_exponentials()
    sheet_07_inverse()
    sheet_08_non_differentiable()
    sheet_09_higher_order()
    sheet_10_local_extrema()
    sheet_11_interpretation()
