#!/usr/bin/env python3
"""Concept sheets for my_learning (basic derivatives, differentiation rules, squared and log loss).

One page per notebook that summarizes its central idea. Every number is computed here from
the same functions, data and check points the notebook uses, with asserts against the values
the notebook prints (or states in its text). Images are written to my_learning/img/, next to the
hand-made chain_rule.png, product_rule.png, sum_rule.png and sum_rule_derivatives.png (untouched).

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_my_learning_concept_images.py
"""
from functools import partial

import numpy as np
import sympy as sp

from viz_style import (REPO, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "my_learning"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"


# ================================================================ helpers
def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)


def tangent(ax, f, slope, x0, half, color, lw=1.8):
    xs = np.array([x0 - half, x0 + half])
    ax.plot(xs, f(x0) + slope * (xs - x0), color=color, linewidth=lw, linestyle="--")
    ax.plot(x0, f(x0), "o", color=color, markersize=7, zorder=5)


def sign_shading(ax, xs, ys):
    """Shade where a derivative curve is negative (blue) or positive (orange)."""
    ax.fill_between(xs, ys, 0, where=ys < 0, color=BLUE_LIGHT, alpha=0.8, linewidth=0)
    ax.fill_between(xs, ys, 0, where=ys > 0, color=ORANGE_LIGHT, alpha=0.8, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=1)


def central_slope(f, x0, h=1e-5):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def f_and_fprime(ax, xs, f, fp, f_label, fp_label, title, x0=None, ylim=None, loc="upper left"):
    ax.plot(xs, f(xs), color=BLUE, linewidth=2.2, label=f_label)
    ax.plot(xs, fp(xs), color=ORANGE, linewidth=2.2, label=fp_label)
    if x0 is not None:
        tangent(ax, f, fp(x0), x0, 0.35 * (xs[-1] - xs[0]) / 4, INK)
        ax.plot(x0, fp(x0), "o", color=ORANGE, markersize=7, zorder=5)
        ax.plot([x0, x0], [f(x0), fp(x0)], color=MUTED, linewidth=1, linestyle=":")
    axes_cross(ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc=loc, fontsize=9.5)
    ax.set_title(title)


# ================================================================ basic_derivatives
def sheet_basic_derivatives():
    x, a, n, c = sp.symbols("x a n c")
    # the notebook's SymPy cell and its printed output
    printed = {
        "Constant": (sp.diff(c, x), "0"),
        "Linear": (sp.diff(a * x + c, x), "a"),
        "Power": (sp.diff(x ** n, x), "n*x**n/x"),
        "x^5": (sp.diff(x ** 5, x), "5*x**4"),
        "1/x": (sp.diff(1 / x, x), "-1/x**2"),
        "sin(x)": (sp.diff(sp.sin(x), x), "cos(x)"),
        "cos(x)": (sp.diff(sp.cos(x), x), "-sin(x)"),
        "tan(x)": (sp.diff(sp.tan(x), x), "tan(x)**2 + 1"),
        "e^x": (sp.diff(sp.exp(x), x), "exp(x)"),
        "a^x": (sp.diff(a ** x, x), "a**x*log(a)"),
        "ln(x)": (sp.diff(sp.log(x), x), "1/x"),
        "log_a(x)": (sp.diff(sp.log(x, a), x), "1/(x*log(a))"),
        "asin(x)": (sp.diff(sp.asin(x), x), "1/sqrt(1 - x**2)"),
        "acos(x)": (sp.diff(sp.acos(x), x), "-1/sqrt(1 - x**2)"),
        "atan(x)": (sp.diff(sp.atan(x), x), "1/(x**2 + 1)"),
    }
    for name, (expr, text) in printed.items():
        assert str(expr) == text, (name, expr)
    # the odd-looking SymPy forms are the table's formulas
    assert sp.simplify(sp.diff(x ** n, x) - n * x ** (n - 1)) == 0
    assert sp.simplify(sp.diff(sp.tan(x), x) - 1 / sp.cos(x) ** 2) == 0
    assert sp.diff(x ** sp.Rational(1, 2), x) == x ** sp.Rational(-1, 2) / 2

    # numerical check: central differences against each table formula at x0 = 0.5
    x0 = 0.5
    table = [("x³", lambda t: t ** 3, lambda t: 3 * t ** 2),
             ("1/x", lambda t: 1 / t, lambda t: -1 / t ** 2),
             ("sin x", np.sin, np.cos),
             ("cos x", np.cos, lambda t: -np.sin(t)),
             ("tan x", np.tan, lambda t: 1 / np.cos(t) ** 2),
             ("eˣ", np.exp, np.exp),
             ("2ˣ", lambda t: 2.0 ** t, lambda t: 2.0 ** t * np.log(2)),
             ("ln x", np.log, lambda t: 1 / t),
             ("arcsin x", np.arcsin, lambda t: 1 / np.sqrt(1 - t ** 2)),
             ("arctan x", np.arctan, lambda t: 1 / (1 + t ** 2))]
    checks = [(name, central_slope(f, x0), fp(x0)) for name, f, fp in table]
    for name, numeric, formula in checks:
        assert close(numeric, formula, 1e-8), name

    fig = new_figure("A short table of basic derivatives is all you differentiate with",
                     "Power rule (xⁿ)′ = n xⁿ⁻¹ covers constants, lines, polynomials, 1/x and √x;  "
                     "(sin x)′ = cos x,  (eˣ)′ = eˣ,  (ln x)′ = 1/x,  (arctan x)′ = 1/(1 + x²).")
    gs = grid(fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1.6, 1.6, 300)
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.2, label="x²")
    ax.plot(xs, 2 * xs, color=BLUE, linewidth=1.6, linestyle="--", label="(x²)′ = 2x")
    ax.plot(xs, xs ** 3, color=ORANGE, linewidth=2.2, label="x³")
    ax.plot(xs, 3 * xs ** 2, color=ORANGE, linewidth=1.6, linestyle="--", label="(x³)′ = 3x²")
    tangent(ax, lambda t: t ** 3, 3.0, 1.0, 0.45, INK, lw=1.5)
    ax.annotate("slope 3·1² = 3", (1.0, 1.0), xytext=(0.15, -1.9), fontsize=10.5, weight="bold",
                arrowprops=dict(arrowstyle="-", color=MUTED))
    axes_cross(ax)
    ax.set_ylim(-2.5, 6)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.set_title("Power rule: bring n down, lower the power")

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    ax.plot(xs, np.sin(xs), color=BLUE, linewidth=2.2, label="sin x")
    ax.plot(xs, np.cos(xs), color=ORANGE, linewidth=2.2, label="(sin x)′ = cos x")
    ax.plot(xs, -np.sin(xs), color=AQUA, linewidth=1.6, linestyle="--", label="(cos x)′ = −sin x")
    for xt in [-np.pi, 0.0, np.pi / 2]:
        tangent(ax, np.sin, np.cos(xt), xt, 0.9, INK, lw=1.5)
    axes_cross(ax)
    ax.set_ylim(-1.6, 2.1)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, ncol=3)
    ax.set_title("Trig: each slope is the next curve")

    panel(fig.add_subplot(gs[0, 2]), "The table (notebook §8)", [
        "c → 0,     ax + b → a",
        "xⁿ → n xⁿ⁻¹,     1/x → −1/x²",
        "sin x → cos x,     cos x → −sin x",
        "tan x → sec² x",
        "eˣ → eˣ,     aˣ → aˣ ln a",
        "ln x → 1/x,     logₐ x → 1/(x ln a)",
        "arcsin x → 1/√(1 − x²)",
        "arccos x → −1/√(1 − x²)",
        "arctan x → 1/(1 + x²)",
    ], spacing=0.092)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2, 2, 300)
    xl = np.linspace(0.08, 2, 300)
    ax.plot(xs, np.exp(xs), color=BLUE, linewidth=2.4, label="eˣ = (eˣ)′")
    ax.plot(xs, 2.0 ** xs, color=AQUA, linewidth=2.0, label="2ˣ")
    ax.plot(xs, 2.0 ** xs * np.log(2), color=AQUA, linewidth=1.6, linestyle="--",
            label=f"(2ˣ)′ = 2ˣ ln 2  (ln 2 = {np.log(2):.4f})")
    ax.plot(xl, np.log(xl), color=ORANGE, linewidth=2.2, label="ln x")
    ax.plot(xl, 1 / xl, color=ORANGE, linewidth=1.6, linestyle="--", label="(ln x)′ = 1/x")
    tangent(ax, np.exp, np.exp(1.0), 1.0, 0.5, INK, lw=1.5)
    axes_cross(ax)
    ax.set_ylim(-2.5, 7.5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("Exponentials and logarithms")

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-0.995, 0.995, 400)
    xw = np.linspace(-3, 3, 400)
    ax.plot(xs, np.arcsin(xs), color=BLUE, linewidth=2.2, label="arcsin x")
    ax.plot(xs, 1 / np.sqrt(1 - xs ** 2), color=BLUE, linewidth=1.6, linestyle="--",
            label="1/√(1 − x²)  (blows up at ±1)")
    ax.plot(xw, np.arctan(xw), color=ORANGE, linewidth=2.2, label="arctan x")
    ax.plot(xw, 1 / (1 + xw ** 2), color=ORANGE, linewidth=1.6, linestyle="--", label="1/(1 + x²)")
    axes_cross(ax)
    ax.set_ylim(-2, 4.5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Inverse trig: slopes are algebraic")

    panel(fig.add_subplot(gs[1, 2]), f"Checked at x = {x0}", [
        "central difference (h = 1e−5) vs formula:",
        *[f"`{name:<9}{numeric:>9.5f}  {formula:>9.5f}`" for name, numeric, formula in checks[:7]],
        "",
        "SymPy prints  n·xⁿ/x  and  tan²x + 1:",
        "the same as n xⁿ⁻¹ and sec² x.",
    ], spacing=0.074, fontsize=12)
    save(fig, name="basic_derivatives.png")


# ================================================================ differentiation_rules
def sheet_differentiation_rules():
    x = sp.symbols("x")
    # worked examples of the notebook (text and SymPy cells)
    assert sp.diff(x ** 2 + x ** 3, x) == 2 * x + 3 * x ** 2
    assert sp.diff(sp.sin(x) + sp.cos(x), x) == sp.cos(x) - sp.sin(x)
    assert sp.diff((3 * x ** 2 + 5) ** 4, x) == 24 * x * (3 * x ** 2 + 5) ** 3
    assert sp.diff(sp.sin(5 * x ** 2), x) == 10 * x * sp.cos(5 * x ** 2)
    assert sp.diff(x ** 2 * sp.sin(x), x) == x ** 2 * sp.cos(x) + 2 * x * sp.sin(x)
    assert sp.simplify(sp.diff(x * sp.exp(x), x) - sp.exp(x) * (1 + x)) == 0
    assert sp.simplify(sp.diff(x ** 2 / (x + 1), x) - (x ** 2 + 2 * x) / (x + 1) ** 2) == 0
    assert sp.simplify(sp.diff(sp.sin(x) / x, x) - (x * sp.cos(x) - sp.sin(x)) / x ** 2) == 0

    # numerical checks printed by the notebook
    chain_x0, chain_val = 1.2, 10 * 1.2 * np.cos(5 * 1.2 ** 2)
    assert close(chain_val, 7.300215774387063, 1e-12)
    prod_x0 = 1.5
    prod_terms = [2 * prod_x0 * np.sin(prod_x0), prod_x0 ** 2 * np.cos(prod_x0)]
    assert close(sum(prod_terms), 3.151643663564495, 1e-12)
    quot_x0 = 2.5
    quot_terms = [quot_x0 * np.cos(quot_x0) / quot_x0 ** 2, -np.sin(quot_x0) / quot_x0 ** 2]
    assert close(sum(quot_terms), -0.4162129892754065, 1e-12)
    comb_x0 = 1.3
    u = comb_x0 ** 2 + 1
    comb_terms = [6 * comb_x0 * u ** 2 * np.sin(comb_x0) / (comb_x0 + 2),
                  u ** 3 * np.cos(comb_x0) / (comb_x0 + 2),
                  -u ** 3 * np.sin(comb_x0) / (comb_x0 + 2) ** 2]
    assert close(sum(comb_terms), 16.33578032711926, 1e-11)
    xr = sp.symbols("x", real=True)
    comb = (xr ** 2 + 1) ** 3 * sp.sin(xr) / (xr + 2)
    assert close(sp.diff(comb, xr).subs(xr, comb_x0), 16.33578032711926, 1e-11)

    fig = new_figure("Four rules break any derivative into basic pieces",
                     "sum (f + g)′ = f′ + g′    product (fg)′ = f′g + fg′    "
                     "chain f(g(x))′ = f′(g(x))·g′(x)    quotient (f/g)′ = (f′g − fg′)/g²")
    gs = grid(fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1.6, 1.3, 300)
    ax.plot(xs, 2 * xs, color=BLUE, linewidth=2.0, label="(x²)′ = 2x")
    ax.plot(xs, 3 * xs ** 2, color=ORANGE, linewidth=2.0, label="(x³)′ = 3x²")
    ax.plot(xs, 2 * xs + 3 * xs ** 2, color=INK, linewidth=2.6, label="(x² + x³)′ = 2x + 3x²")
    xm = 1.0
    ax.plot([xm, xm], [0, 2 * xm], color=BLUE, linewidth=5, zorder=4)
    ax.plot([xm, xm], [2 * xm, 2 * xm + 3 * xm ** 2], color=ORANGE, linewidth=5, zorder=4)
    ax.text(0.15, 5.6, "at x = 1:\n2 + 3 = 5", ha="left", fontsize=10.5, weight="bold")
    axes_cross(ax)
    ax.set_ylim(-3.5, 9)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Sum rule: slopes stack")

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-1.5, 1.5, 600)
    f_and_fprime(ax, xs, lambda t: np.sin(5 * t ** 2), lambda t: 10 * t * np.cos(5 * t ** 2),
                 "y = sin(5x²)", "y′ = cos(5x²)·10x", "Chain rule: outer′ × inner′",
                 ylim=(-17, 22), loc="upper left")
    ax.plot(chain_x0, chain_val, "o", color=ORANGE, markersize=8, zorder=5)
    ax.annotate(f"y′(1.2) = {chain_val:.4f}", (chain_x0, chain_val), xytext=(-0.35, -14),
                fontsize=10.5, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))

    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(-4, 4, 400)
    f_and_fprime(ax, xs, lambda t: t ** 2 * np.sin(t), lambda t: 2 * t * np.sin(t) + t ** 2 * np.cos(t),
                 "y = x² sin x", "y′ = 2x sin x + x² cos x", "Product rule: f′g + fg′",
                 ylim=(-16, 19), loc="upper left")
    ax.plot(prod_x0, sum(prod_terms), "o", color=ORANGE, markersize=8, zorder=5)
    ax.annotate(f"y′(1.5) = {sum(prod_terms):.4f}", (prod_x0, sum(prod_terms)), xytext=(0.6, -12),
                fontsize=10.5, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.5, 10, 400)
    f_and_fprime(ax, xs, lambda t: np.sin(t) / t, lambda t: (t * np.cos(t) - np.sin(t)) / t ** 2,
                 "y = sin x / x", "y′ = (x cos x − sin x)/x²", "Quotient rule: (f′g − fg′)/g²",
                 ylim=(-0.75, 1.25), loc="upper right")
    ax.plot(quot_x0, sum(quot_terms), "o", color=ORANGE, markersize=8, zorder=5)
    ax.annotate(f"y′(2.5) = {sum(quot_terms):.4f}", (quot_x0, sum(quot_terms)), xytext=(3.6, -0.62),
                fontsize=10.5, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))

    ax = fig.add_subplot(gs[1, 1])
    rows = [("product\nx = 1.5", prod_terms, ["f′g", "fg′"], sum(prod_terms)),
            ("quotient\nx = 2.5", quot_terms, ["f′/g", "−fg′/g²"], sum(quot_terms)),
            ("combined\nx = 1.3", comb_terms, ["sin part", "cos part", "−fg′/g²"], sum(comb_terms))]
    colors = [BLUE, ORANGE, AQUA]
    for r, (_, terms, names, total) in enumerate(rows):
        scale = max(abs(v) for v in terms) + abs(total)
        start = 0.0
        for k, (v, nm) in enumerate(zip(terms, names)):
            ax.barh(r, v / scale, left=start / scale, height=0.5 - 0.14 * k, color=colors[k], zorder=2)
            start += v
        ax.plot(total / scale, r, "D", color=INK, markersize=8, zorder=4)
        labels = "  +  ".join(f"{nm} ({v:.3f})".replace("-", "−") for nm, v in zip(names, terms))
        ax.text(-0.98, r - 0.36, labels + f"  =  {total:.4f}".replace("-", "−"), ha="left",
                va="center", fontsize=9.3, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_ylim(len(rows) - 0.35, -0.65)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_xlabel("terms laid end to end (each row scaled);  ◆ = value the notebook prints")
    ax.set_title("The pieces add up to the printed checks")

    panel(fig.add_subplot(gs[1, 2]), "Which rule, and combining them", [
        "f + g → sum      f·g → product",
        "f(g(x)) → chain  f/g → quotient",
        "",
        r"$y = (3x^2+5)^4$:  outer $u^4$, inner $3x^2+5$",
        r"$y' = 4(3x^2+5)^3\cdot 6x = 24x(3x^2+5)^3$",
        "",
        "y = (x² + 1)³ sin x / (x + 2): a quotient",
        "of a product with a chain inside:",
        r"$f' = 6x(x^2+1)^2\sin x + (x^2+1)^3\cos x$",
        f"y′(1.3) = {sum(comb_terms):.4f}  (direct = rules)",
    ], spacing=0.085, fontsize=12)
    save(fig, name="differentiation_rules.png")


# ================================================================ optimization_of_log_loss
def sheet_log_loss():
    y = np.array([0, 0, 1, 1, 1])                     # the notebook's data
    mean_y = np.mean(y)
    assert mean_y == 0.6

    def total_log_loss(p, observations):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.sum(observations * np.log(p) + (1 - observations) * np.log(1 - p))

    def dJ(p):
        return -np.sum(y[:, None] / p - (1 - y[:, None]) / (1 - p), axis=0)

    p_range = np.linspace(0.01, 0.99, 100)           # the notebook's grid
    losses = np.array([total_log_loss(p, y) for p in p_range])
    j_min = total_log_loss(mean_y, y)
    assert losses.min() >= j_min and abs(p_range[np.argmin(losses)] - mean_y) < 0.01
    assert close(dJ(np.array([mean_y]))[0], 0.0, 1e-12)
    # SymPy: the notebook's printed derivative equals -sum (y_i - p) / (p (1 - p))
    p, yi = sp.symbols("p y_i")
    term = -(yi * sp.log(p) + (1 - yi) * sp.log(1 - p))
    assert sp.simplify(sp.diff(term, p) - (-1 + yi / p) / (p - 1)) == 0
    assert sp.simplify(sp.diff(term, p) + (yi - p) / (p * (1 - p))) == 0
    assert sp.simplify(sp.diff(term, p, 2) - (yi / p ** 2 + (1 - yi) / (1 - p) ** 2)) == 0
    k, n = int(y.sum()), len(y)
    compare = [(pv, total_log_loss(pv, y)) for pv in (0.5, mean_y, 0.9)]
    wrong = -np.log(1 - 0.99)

    fig = new_figure("Log loss is minimized by the share of 1s:  p = mean(y) = 0.6",
                     "J(p) = −Σ [yᵢ ln p + (1 − yᵢ) ln(1 − p)],   J′(p) = −Σ (yᵢ − p) / (p(1 − p)) = 0"
                     "   ⇒   p = (1/n) Σ yᵢ   (data y = [0, 0, 1, 1, 1])")
    gs = grid(fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(p_range, losses, color=BLUE, linewidth=2.4, label="total log loss J(p)")
    ax.axvline(mean_y, color=ORANGE, linestyle="--", linewidth=1.6, label=f"optimal p = {mean_y}")
    ax.plot(mean_y, j_min, "o", color=AQUA, markersize=9, zorder=5)
    ax.annotate(f"J(0.6) = {j_min:.3f}", (mean_y, j_min), xytext=(0.62, 8.5), fontsize=10.5,
                weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(0, 16)
    ax.set_xlabel("probability parameter p")
    ax.set_ylabel("loss J(p)")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("One convex valley, bottom at the mean")

    ax = fig.add_subplot(gs[0, 1])
    pp = np.linspace(0.01, 0.99, 300)
    ax.plot(pp, -k * np.log(pp), color=BLUE, linewidth=2.2, label=f"{k} ones:  −{k} ln p")
    ax.plot(pp, -(n - k) * np.log(1 - pp), color=ORANGE, linewidth=2.2,
            label=f"{n - k} zeros:  −{n - k} ln(1 − p)")
    ax.plot(pp, -k * np.log(pp) - (n - k) * np.log(1 - pp), color=INK, linewidth=2.6,
            label="sum = J(p)")
    ax.axvline(mean_y, color=MUTED, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 16)
    ax.set_xlabel("p")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("Each class pulls p its own way")

    ax = fig.add_subplot(gs[0, 2])
    pd = np.linspace(0.05, 0.95, 400)
    d = dJ(pd)
    sign_shading(ax, pd, d)
    ax.plot(pd, d, color=INK2, linewidth=2.2, label="J′(p) = −Σ (yᵢ − p)/(p(1 − p))")
    ax.plot(mean_y, 0, "o", color=AQUA, markersize=9, zorder=5)
    ax.text(0.42, -35, "J′ < 0: raise p", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(0.62, 35, "J′ > 0: lower p", ha="left", fontsize=10.5, color=ORANGE, weight="bold")
    ax.annotate("J′(0.6) = 0", (mean_y, 0), xytext=(0.62, -30), fontsize=10.5, color=AQUA,
                weight="bold", arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.set_ylim(-60, 60)
    ax.set_xlabel("p")
    ax.set_ylabel("slope J′(p)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("The derivative crosses zero once")

    ax = fig.add_subplot(gs[1, 0])
    like = pp ** k * (1 - pp) ** (n - k)
    ax.plot(pp, like, color=BLUE, linewidth=2.4, label=r"likelihood $p^3(1-p)^2 = e^{-J(p)}$")
    ax.plot(mean_y, mean_y ** k * (1 - mean_y) ** (n - k), "o", color=AQUA, markersize=9, zorder=5)
    assert close(np.exp(-j_min), mean_y ** k * (1 - mean_y) ** (n - k), 1e-12)
    ax.annotate(f"peak {np.exp(-j_min):.5f}", (mean_y, np.exp(-j_min)), xytext=(0.68, 0.038),
                fontsize=10.5, weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(0, 0.042)
    ax.set_xlabel("p")
    ax.set_ylabel("probability of the data")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Min log loss = max likelihood (MLE)")

    panel(fig.add_subplot(gs[1, 1]), "The derivation", [
        r"$J'(p) = -\sum\,[\,y_i/p - (1-y_i)/(1-p)\,]$",
        r"$\quad = -\sum\,(y_i - p)\,/\,(p(1-p))$",
        r"Set to 0:  $\sum (y_i - p) = 0$",
        r"$\Rightarrow\ n\,p = \sum y_i\ \Rightarrow\ p = \bar y$",
        "",
        r"$J''(p) = \sum\,[\,y_i/p^2 + (1-y_i)/(1-p)^2\,] > 0$",
        "so the stationary point is the minimum.",
        "",
        f"Here: p = {k}/{n} = {mean_y}.",
    ], spacing=0.092, fontsize=12.5)

    panel(fig.add_subplot(gs[1, 2]), "Penalties and the baseline", [
        *[f"`J({pv:<3}) = {jv:6.3f}" + ("   ← minimum`" if pv == mean_y else "`")
          for pv, jv in compare],
        "",
        "Confident but wrong costs a lot:",
        f"predict p = 0.99 for a 0  →  −ln 0.01 = {wrong:.2f}",
        "",
        "A model that always predicts the share",
        "of positives is the best constant model",
        "under log loss, just as the mean is",
        "under squared loss.",
    ], spacing=0.085)
    save(fig, name="optimization_of_log_loss.png")


# ================================================================ optimization_of_squared_loss
def sheet_squared_loss():
    np.random.seed(42)                                # as in the notebook (unused by its data)
    y = np.array([10, 12, 15, 20, 25])
    mean_y = np.mean(y)
    assert mean_y == 16.4

    def total_squared_loss(w, observations):
        return np.sum((observations - w) ** 2)

    w_range = np.linspace(min(y) - 5, max(y) + 5, 100)
    losses = np.array([total_squared_loss(w, y) for w in w_range])
    j_min = total_squared_loss(mean_y, y)
    assert close(j_min, 149.2, 1e-9) and losses.min() >= j_min
    residuals = y - mean_y
    assert close(residuals.sum(), 0.0, 1e-12)
    median = float(np.median(y))
    j_median = total_squared_loss(median, y)
    assert close(j_median, 159.0, 1e-9)
    n = len(y)
    # J(w) = n (w - mean)^2 + J(mean)
    assert np.allclose(losses, n * (w_range - mean_y) ** 2 + j_min)
    w, yi = sp.symbols("w y_i")
    assert sp.expand(sp.diff((yi - w) ** 2, w)) == 2 * w - 2 * yi   # the notebook's SymPy output
    contrib = residuals ** 2
    share = contrib[-1] / j_min

    fig = new_figure("Squared loss is minimized by the mean:  w = mean(y) = 16.4",
                     "J(w) = Σ (yᵢ − w)²,   J′(w) = −2 Σ (yᵢ − w) = 0   ⇒   w = (1/n) Σ yᵢ"
                     "   (data y = [10, 12, 15, 20, 25])")
    gs = grid(fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(w_range, losses, color=BLUE, linewidth=2.4, label="total squared loss J(w)")
    ax.axvline(mean_y, color=ORANGE, linestyle="--", linewidth=1.6, label=f"mean w = {mean_y}")
    ax.plot(y, np.zeros_like(y), "x", color=INK, markersize=9, mew=2, label="data points yᵢ", clip_on=False)
    ax.plot(mean_y, j_min, "o", color=AQUA, markersize=9, zorder=5)
    ax.annotate(f"J(16.4) = {j_min:.1f}", (mean_y, j_min), xytext=(17.5, 500), fontsize=10.5,
                weight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(0, 1500)
    ax.set_xlabel("parameter w")
    ax.set_ylabel("loss J(w)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("A parabola with its bottom at the mean")

    ax = fig.add_subplot(gs[0, 1])
    d = -2 * (y[:, None] - w_range).sum(axis=0)
    sign_shading(ax, w_range, d)
    ax.plot(w_range, d, color=INK2, linewidth=2.2, label="J′(w) = −2 Σ (yᵢ − w) = 10(w − 16.4)")
    ax.plot(mean_y, 0, "o", color=AQUA, markersize=9, zorder=5)
    ax.text(9.5, -25, "J′ < 0: raise w", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(24.5, 25, "J′ > 0: lower w", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.annotate("J′(16.4) = 0", (mean_y, 0), xytext=(17.5, -110), fontsize=10.5, color=AQUA,
                weight="bold", arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.set_ylim(-170, 170)
    ax.set_xlabel("w")
    ax.set_ylabel("slope J′(w)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("The derivative is a straight line")

    panel(fig.add_subplot(gs[0, 2]), "The derivation", [
        r"$J'(w) = \sum 2(y_i - w)\cdot(-1) = -2\sum (y_i - w)$",
        r"Set to 0:  $\sum y_i - n\,w = 0$",
        r"$\Rightarrow\ w = \frac{1}{n}\sum y_i = \bar y$",
        "",
        "Completing the square:",
        r"$J(w) = n\,(w - \bar y)^2 + J(\bar y)$",
        f"     = 5 (w − 16.4)² + {j_min:.1f}",
        "",
        "Any other w adds n·(distance to the mean)².",
    ], spacing=0.092, fontsize=12.5)

    ax = fig.add_subplot(gs[1, 0])
    from matplotlib.patches import Rectangle
    for yv, r in zip(y, residuals):
        color = ORANGE if r > 0 else BLUE
        ax.add_patch(Rectangle((min(yv, mean_y), 0), abs(r), abs(r), facecolor=color, alpha=0.3,
                               edgecolor=color, linewidth=1.5))
        ax.plot(yv, 0, "x", color=INK, markersize=9, mew=2, clip_on=False, zorder=4)
    ax.axvline(mean_y, color=INK, linewidth=1.4)
    ax.text(mean_y + 0.2, 9.0, "w = 16.4", fontsize=10.5, weight="bold")
    ax.text(9.2, 10.4, "residuals " + ", ".join(f"{r:+.1f}" for r in residuals).replace("-", "−")
            + "  sum to 0", fontsize=10, color=INK2)
    ax.set_xlim(9, 26)
    ax.set_ylim(0, 11.5)
    ax.set_aspect("equal")
    ax.set_xlabel("value")
    ax.grid(True)
    ax.set_title("Squared loss = total area of the squares")

    ax = fig.add_subplot(gs[1, 1])
    labels = [str(v) for v in y]
    positions = np.arange(n)
    ax.bar(positions, contrib, width=0.62, color=[ORANGE if v == 25 else BLUE for v in y], zorder=2)
    for p_, v in zip(positions, contrib):
        ax.text(p_, v + 1.5, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 90)
    ax.set_xlabel("data point yᵢ")
    ax.set_ylabel("(yᵢ − 16.4)²")
    ax.grid(axis="y", zorder=0)
    ax.text(-0.3, 80, f"y = 25 alone: {share:.0%} of {j_min:.1f}", fontsize=10.5, weight="bold",
            color=ORANGE)
    ax.set_title("Squaring makes far points dominate")

    panel(fig.add_subplot(gs[1, 2]), "Mean vs median", [
        f"`J(mean   = {mean_y}) = {j_min:.1f}`",
        f"`J(median = {median:g})   = {j_median:.1f}`",
        f"`difference = 5·1.4² = {j_median - j_min:.1f}`",
        "",
        "The mean is the best fit under squared",
        "loss; it is pulled toward outliers",
        "(the median minimizes Σ|yᵢ − w| instead).",
        "",
        "Same story as log loss: the best constant",
        "prediction is the average of the data.",
    ], spacing=0.085)
    save(fig, name="optimization_of_squared_loss.png")


if __name__ == "__main__":
    sheet_basic_derivatives()
    sheet_differentiation_rules()
    sheet_log_loss()
    sheet_squared_loss()
