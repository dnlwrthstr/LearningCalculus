#!/usr/bin/env python3
"""Concept sheets for hypersklill/01_limits and hypersklill/04_integrals.

One page per notebook that summarizes its central idea. Every number is
computed here from the same functions, intervals and sample points the
notebook uses, with asserts against the values the notebook prints.
Images are written to hypersklill/01_limits/img/ and hypersklill/04_integrals/img/.

Run from anywhere:  python scripts/make_hypersklill_limits_integrals_concept_images.py
"""
from fractions import Fraction
from functools import partial
from math import e, log, pi, sqrt

import numpy as np
import sympy as sp
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon, Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel, trapezoid)
from viz_style import save as _save

LIMITS = REPO / "hypersklill" / "01_limits"
INTEGRALS = REPO / "hypersklill" / "04_integrals"
save_limits = partial(_save, folder=LIMITS / "img")
save_integrals = partial(_save, folder=INTEGRALS / "img")

ORANGE_LIGHT = "#f8c9b3"     # a light tint of ORANGE for secondary fills
X = sp.symbols("x", real=True)


# ================================================================ helpers
def style(ax, title, xlabel="x", ylabel="y", axes_lines=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if axes_lines:
        ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)


def hole(ax, x, y, color=ORANGE, size=9):
    """An open circle: the point is missing from the graph."""
    ax.plot(x, y, "o", markersize=size, markerfacecolor=SURFACE, markeredgecolor=color,
            markeredgewidth=2, zorder=5)


def dot(ax, x, y, color=ORANGE, size=8):
    ax.plot(x, y, "o", markersize=size, color=color, zorder=5)


def slope_check(ax, xs, F_vals, label="slope of F (numerical)", color=INK):
    """Dashed numerical derivative of F, to lay over the integrand f."""
    ax.plot(xs, np.gradient(F_vals, xs), "--", color=color, linewidth=1.6, label=label, zorder=4)


def numeric(expr):
    return sp.lambdify(X, expr, "numpy")


def split_at(xs, ys, cut, gap):
    """Insert NaNs near a pole so matplotlib does not draw a vertical connector."""
    ys = np.array(ys, float)
    ys[np.abs(xs - cut) < gap] = np.nan
    return ys


# ================================================================ limits 00
def sheet_00_sequence_limits():
    n25 = np.arange(1, 26)
    a = 1 / (2 ** n25)
    b = (n25 - 1) / n25
    assert f"{a[9]:.10f}" == "0.0009765625" and f"{b[9]:.10f}" == "0.9000000000"
    n20 = np.arange(1, 21)
    c = (-1) ** n20

    # the notebook's search for 1/2^n (exact in floating point: powers of two)
    def min_N_for_eps_power_of_two(eps, max_search=100):
        for N in range(max_search):
            if all((1 / (2 ** n)) < eps for n in range(N + 1, max_search + 1)):
                return N
        return None

    eps_pow = [0.1, 0.01, 0.001, 1e-5]
    N_pow = [min_N_for_eps_power_of_two(eps) for eps in eps_pow]
    assert N_pow == [3, 6, 9, 16]

    # for b_n = (n-1)/n the condition is 1/n < eps; solved in exact fractions
    def min_N_for_eps_b(eps):
        eps = Fraction(str(eps))
        N = 0
        while not Fraction(1, N + 1) < eps:
            N += 1
        return N

    eps_b = [0.1, 0.05, 0.01, 0.001]
    N_b = [min_N_for_eps_b(eps) for eps in eps_b]
    assert N_b == [10, 20, 100, 1000]

    def seq_piecewise(k):
        return 1 / k if k % 2 == 0 else 1 / (2 * k)

    def seq_divisible_by_10(k):
        return 1 / 10 if k % 10 == 0 else 1 / k

    fig = new_figure("A sequence converges when its tail fits inside every ε-band",
                     "lim aₙ = A  ⇔  for every ε > 0 there is an N with |aₙ − A| < ε for all n > N.  "
                     "The terms need not reach A, only stay close to it.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(n20, c, "o", color=MUTED, markersize=5, label="cₙ = (−1)ⁿ  diverges")
    ax.plot(n25, b, "o", color=AQUA, markersize=5, label="bₙ = (n−1)/n → 1")
    ax.plot(n25, a, "o", color=BLUE, markersize=5, label="aₙ = 1/2ⁿ → 0")
    ax.axhline(1, color=AQUA, linestyle=":", linewidth=1.2)
    style(ax, "Two settle down, one never does", "n", "term")
    ax.set_ylim(-1.35, 1.75)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    n45 = np.arange(1, 46)
    b45 = (n45 - 1) / n45
    eps, N = 0.1, N_b[0]
    ax.axhspan(1 - eps, 1 + eps, color=BLUE_LIGHT, alpha=0.5, linewidth=0, label="band 1 ± ε, ε = 0.1")
    inside = n45 > N
    ax.plot(n45[~inside], b45[~inside], "o", color=ORANGE, markersize=5, label=f"n ≤ N = {N}")
    ax.plot(n45[inside], b45[inside], "o", color=BLUE, markersize=5, label="n > N: all inside")
    ax.axvline(N + 0.5, color=INK2, linestyle="--", linewidth=1.2)
    ax.axhline(1, color=INK2, linewidth=1)
    style(ax, "bₙ = (n−1)/n: the tail stays in the band", "n", "bₙ", axes_lines=False)
    ax.set_ylim(-0.05, 1.3)
    ax.legend(loc="lower right", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "Reading the ε–N definition", [
        "1. Someone picks a precision ε > 0.",
        "2. You answer with an index N.",
        "3. Every term after N lies in (A − ε, A + ε).",
        "",
        "bₙ = (n−1)/n:  |bₙ − 1| = 1/n < ε  ⇔  n > 1/ε,",
        "so N = 1/ε works (ε = 0.1 → N = 10).",
        "",
        "Uniqueness: if A ≠ A′, take ε = |A − A′|/2.",
        "The two bands are disjoint, so one tail",
        "cannot sit in both. A limit, if any, is unique.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    vals = np.array([seq_piecewise(k) for k in n45])
    assert f"{vals[2]:.6f}" == "0.166667" and f"{vals[11]:.6f}" == "0.083333"
    even = n45 % 2 == 0
    xs = np.linspace(1, 45, 300)
    ax.plot(xs, 1 / xs, color=INK2, linewidth=1.2, linestyle="--", label="envelope 1/n")
    ax.plot(n45[even], vals[even], "o", color=BLUE, markersize=5, label="even n: 1/n")
    ax.plot(n45[~even], vals[~even], "o", color=AQUA, markersize=5, label="odd n: 1/(2n)")
    style(ax, "Zig-zag, but squeezed to 0", "n", "aₙ")
    ax.set_ylim(-0.03, 0.6)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    spikes = np.array([seq_divisible_by_10(k) for k in n45])
    tens = n45 % 10 == 0
    ax.axhline(0.1, color=ORANGE, linestyle=":", linewidth=1.2)
    ax.plot(n45[~tens], spikes[~tens], "o", color=BLUE, markersize=5, label="1/n → 0")
    ax.plot(n45[tens], spikes[tens], "o", color=ORANGE, markersize=8, label="every 10th term = 1/10")
    style(ax, "Spikes forever: no limit", "n", "bₙ")
    ax.set_ylim(-0.03, 0.6)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(eps_pow, N_pow, "o-", color=BLUE, linewidth=2, label="aₙ = 1/2ⁿ")
    ax.plot(eps_b, N_b, "o-", color=AQUA, linewidth=2, label="bₙ = (n−1)/n")
    for e_, n_ in zip(eps_pow, N_pow):
        ax.annotate(f"N = {n_}", (e_, n_), textcoords="offset points", xytext=(6, -14),
                    fontsize=9.5, color=BLUE)
    for e_, n_ in zip(eps_b, N_b):
        first = e_ == eps_b[0]
        ax.annotate(f"N = {n_}", (e_, n_), textcoords="offset points",
                    xytext=(8, -12) if first else (-8, 2), ha="left" if first else "right",
                    fontsize=9.5, color=AQUA)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlim(0.6, 3e-6)
    ax.set_ylim(1.5, 4000)
    style(ax, "Smaller ε needs a later N", "ε (log scale, shrinking →)", "N (log scale)",
          axes_lines=False)
    ax.legend(loc="upper left", fontsize=9.5)
    save_limits(fig, name="00_sequence_limits.png")


# ================================================================ limits 01
def sheet_01_function_limits():
    def g(x):
        return (x ** 2 + x - 2) / (x - 1)

    xs_left = [0.9, 0.99, 0.999, 0.9999]
    xs_right = [1.1, 1.01, 1.001, 1.0001]
    assert f"{g(0.9):.4f}" == "2.9000" and f"{g(1.0001):.4f}" == "3.0001"
    expr = (X ** 2 + X - 2) / (X - 1)
    assert sp.limit(expr, X, 1) == 3 and sp.simplify(expr) == X + 2

    fig = new_figure("A limit is where f(x) is headed, not where it is",
                     "lim (x→a) f(x) = A: for every ε > 0 there is δ > 0 with 0 < |x − a| < δ ⇒ |f(x) − A| < ε.  "
                     "g(x) = (x² + x − 2)/(x − 1) is undefined at 1 yet tends to 3.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 4, 600)
    ys = split_at(xs, g(np.where(np.abs(xs - 1) < 1e-9, 0.5, xs)), 1, 1e-3)
    ax.plot(xs, ys, color=BLUE, linewidth=2.2, label="g(x) = x + 2 for x ≠ 1")
    hole(ax, 1, 3)
    ax.annotate("hole at (1, 3):\ng(1) = 0/0 undefined", (1, 3), textcoords="offset points",
                xytext=(18, -46), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "A hole, not a wall", "x", "g(x)")
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    eps = 0.3
    delta = eps
    xs = np.linspace(0.4, 1.6, 300)
    ax.axhspan(3 - eps, 3 + eps, color=BLUE_LIGHT, alpha=0.45, linewidth=0, label="3 ± ε  (ε = 0.3)")
    ax.axvspan(1 - delta, 1 + delta, color=ORANGE_LIGHT, alpha=0.5, linewidth=0, label="1 ± δ  (δ = ε)")
    ax.plot(xs, xs + 2, color=BLUE, linewidth=2.2)
    hole(ax, 1, 3)
    ax.plot([1 - delta, 1 + delta], [3 - eps, 3 + eps], "s", color=INK, markersize=5)
    style(ax, "ε–δ: inside the orange strip, g stays in the band", "x", "g(x)", axes_lines=False)
    ax.set_xlim(0.4, 1.6)
    ax.set_ylim(2.35, 3.75)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "Proof: choose δ = ε", [
        "For x ≠ 1:  x² + x − 2 = (x + 2)(x − 1),",
        "so g(x) = x + 2  and",
        "|g(x) − 3| = |x + 2 − 3| = |x − 1|.",
        "",
        "If 0 < |x − 1| < δ = ε, then |g(x) − 3| < ε.",
        "ε = 0.5, 0.1, 0.01  →  δ = 0.5, 0.1, 0.01.",
        "",
        "Continuous f: just substitute, e.g.",
        "f(x) = x² + 2x + 1 → lim at 3 is f(3) = 16.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    k = np.arange(1, 5)
    err_left = [3 - g(t) for t in xs_left]
    err_right = [g(t) - 3 for t in xs_right]
    ax.plot(k, err_left, "o-", color=BLUE, linewidth=2, label="xₙ = 0.9, 0.99, … (from the left)")
    ax.plot(k + 0.06, err_right, "s--", color=AQUA, linewidth=2, label="xₙ = 1.1, 1.01, … (from the right)")
    for kk, v in zip(k, err_left):
        ax.annotate(f"g = {g(xs_left[kk - 1]):.4f}", (kk, v), textcoords="offset points",
                    xytext=(12, 6), fontsize=9.5, color=BLUE)
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlim(0.8, 4.7)
    ax.set_ylim(3e-5, 0.6)
    style(ax, "Heine: every xₙ → 1 gives g(xₙ) → 3", "step n", "|g(xₙ) − 3|  (log scale)",
          axes_lines=False)
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-2, 2, 400)
    h = np.where(xs < 0, -1, 1)
    ax.plot(xs[xs < 0], h[xs < 0], color=BLUE, linewidth=2.5, label="left side → −1")
    ax.plot(xs[xs >= 0], h[xs >= 0], color=AQUA, linewidth=2.5, label="right side → +1")
    hole(ax, 0, -1, color=BLUE)
    dot(ax, 0, 1, color=AQUA)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "One-sided limits differ: no limit at 0", "x", "h(x)")
    ax.set_ylim(-1.6, 1.8)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    t = np.linspace(0, 20, 1000)
    phi = np.exp(-0.2 * t) * np.cos(3 * t)
    ax.fill_between(t, -np.exp(-0.2 * t), np.exp(-0.2 * t), color=BLUE_LIGHT, alpha=0.45,
                    linewidth=0, label="envelope ± e^(−0.2t) → 0")
    ax.plot(t, phi, color=BLUE, linewidth=1.8, label="φ(t) = e^(−0.2t) cos 3t")
    style(ax, "At infinity: the damped pendulum → 0", "t", "φ(t)")
    ax.set_ylim(-1.1, 1.45)
    ax.legend(loc="upper right", fontsize=9.5)
    save_limits(fig, name="01_function_limits.png")


# ================================================================ limits 02
def sheet_02_one_sided_limits():
    n = np.arange(1, 8)
    x_n = 1 / (2 * np.pi * n)
    y_n = 1 / (np.pi / 2 + 2 * np.pi * n)
    assert np.allclose(np.sin(1 / x_n), 0, atol=1e-14) and np.allclose(np.sin(1 / y_n), 1)
    assert f"{x_n[0]:.6f}" == "0.159155" and f"{y_n[0]:.6f}" == "0.127324"

    fig = new_figure("A two-sided limit exists only when both sides agree",
                     "lim (x→a) f = A  ⇔  lim (x→a⁻) f = lim (x→a⁺) f = A.  "
                     "A limit fails by a jump, by endless oscillation, or by blowing up.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 2, 600)
    ax.plot(xs[xs < 0], -np.ones((xs < 0).sum()), color=BLUE, linewidth=2.5)
    ax.plot(xs[xs >= 0], np.ones((xs >= 0).sum()), color=AQUA, linewidth=2.5)
    hole(ax, 0, -1, color=BLUE)
    dot(ax, 0, 1, color=AQUA)
    for xv in [-0.5, -0.1]:
        dot(ax, xv, -1, color=BLUE, size=6)
    for xv in [0.5, 0.1]:
        dot(ax, xv, 1, color=AQUA, size=6)
    ax.text(-1.9, -0.7, "x → 0⁻:  f = −1", color=BLUE, fontsize=11)
    ax.text(0.25, 1.3, "x → 0⁺:  f = +1", color=AQUA, fontsize=11)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "Jump: −1 ≠ +1, no limit at 0", "x", "f(x)")
    ax.set_ylim(-1.6, 1.8)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(1, 3, 300)
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.2, label="g(x) = x²")
    for xv, color in [(1.9, BLUE), (1.99, BLUE), (2.01, AQUA), (2.1, AQUA)]:
        dot(ax, xv, xv ** 2, color=color, size=6)
    ax.annotate("from the left:\n1.9 → 3.61\n1.99 → 3.9601", (1.9, 3.61), xytext=(1.1, 5.6),
                fontsize=10, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE))
    ax.annotate("from the right:\n2.1 → 4.41\n2.01 → 4.0401", (2.1, 4.41), xytext=(2.3, 1.4),
                fontsize=10, color=AQUA, arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.axhline(4, color=INK2, linestyle=":", linewidth=1.2)
    ax.axvline(2, color=INK2, linestyle=":", linewidth=1.2)
    style(ax, "Both sides → 4, so the limit at 2 is 4", "x", "g(x)", axes_lines=False)
    ax.set_ylim(0.5, 9.5)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "When does lim (x→a) f exist?", [
        "Exactly when the left-hand limit exists,",
        "the right-hand limit exists, and they agree.",
        "",
        "Three ways to fail:",
        "• jump: one-sided limits differ (−1 vs 1)",
        "• oscillation: two sequences xₙ → a give",
        "   different limits of f(xₙ)  (Heine)",
        "• blow-up: 1/x → +∞ from the right,",
        "   −∞ from the left; only |1/x| → ∞",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    xl = np.linspace(-0.2, -0.01, 3000)
    xr = np.linspace(0.01, 0.2, 3000)
    ax.plot(xl, np.sin(1 / xl), color=BLUE_LIGHT, linewidth=1)
    ax.plot(xr, np.sin(1 / xr), color=BLUE, linewidth=1)
    ax.plot(x_n, np.sin(1 / x_n), "o", color=ORANGE, markersize=7, label="xₙ = 1/(2πn): sin = 0")
    ax.plot(y_n, np.sin(1 / y_n), "s", color=AQUA, markersize=7, label="yₙ = 1/(π/2 + 2πn): sin = 1")
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "sin(1/x): two sequences, two limits", "x", "sin(1/x)")
    ax.set_ylim(-1.25, 1.9)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    xl = np.linspace(-1, -0.02, 500)
    xr = np.linspace(0.02, 1, 500)
    ax.plot(xl, 1 / xl, color=BLUE, linewidth=2.2, label="x → 0⁻: 1/x → −∞")
    ax.plot(xr, 1 / xr, color=AQUA, linewidth=2.2, label="x → 0⁺: 1/x → +∞")
    for xv in [0.5, 0.1]:
        dot(ax, xv, 1 / xv, color=AQUA, size=6)
        dot(ax, -xv, -1 / xv, color=BLUE, size=6)
    ax.annotate("x = 0.1 → 10", (0.1, 10), textcoords="offset points", xytext=(12, 4), fontsize=10)
    ax.annotate("x = −0.1 → −10", (-0.1, -10), textcoords="offset points", xytext=(-100, -4),
                fontsize=10)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "Blow-up: infinite one-sided limits", "x", "1/x")
    ax.set_ylim(-20, 20)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    x1 = np.linspace(-1, 3, 1000)
    mask = np.abs(x1 - 1) > 1e-3
    rem = np.where(mask, (x1 ** 2 + x1 - 2) / np.where(mask, x1 - 1, 1), np.nan)
    blow = np.where(mask, 1 / np.where(mask, x1 - 1, 1), np.nan)
    blow = split_at(x1, blow, 1, 0.02)
    ax.plot(x1, rem, color=BLUE, linewidth=2.2, label="(x² + x − 2)/(x − 1): finite limit 3")
    hole(ax, 1, 3, color=BLUE)
    ax.plot(x1, blow, color=ORANGE, linewidth=2.2, label="1/(x − 1): ±∞ at 1")
    ax.axvline(1, color=MUTED, linewidth=0.9, linestyle=":")
    style(ax, "At x = 1: removable hole vs blow-up", "x", "y")
    ax.set_ylim(-10, 10)
    ax.legend(loc="lower right", fontsize=9.5)
    save_limits(fig, name="02_one_sided_limits.png")


# ================================================================ limits 03
def sheet_03_limit_laws():
    f, g = X ** 2, 3 * X + 1
    lf, lg = sp.limit(f, X, 2), sp.limit(g, X, 2)
    checks = {
        "sum": (lf + lg, sp.limit(f + g, X, 2)),
        "difference": (lf - lg, sp.limit(f - g, X, 2)),
        "product": (lf * lg, sp.limit(f * g, X, 2)),
        "quotient": (sp.limit(X ** 2 + 1, X, 1) / sp.limit(X + 3, X, 1),
                     sp.limit((X ** 2 + 1) / (X + 3), X, 1)),
        "constant": (5 * sp.limit(X ** 2 - 3, X, 2), sp.limit(5 * (X ** 2 - 3), X, 2)),
        "power": (sp.limit(X + 2, X, 1) ** 3, sp.limit((X + 2) ** 3, X, 1)),
        "root": (sp.sqrt(sp.limit(X + 3, X, 1)), sp.limit(sp.sqrt(X + 3), X, 1)),
    }
    assert [v[0] for v in checks.values()] == [11, -3, 28, sp.Rational(1, 2), 5, 27, 2]
    assert all(lhs == rhs for lhs, rhs in checks.values())

    num, den = X ** 4 - 3 * X ** 2 + 1, 4 - X
    assert sp.limit(num, X, 1) == -1 and sp.limit(den, X, 1) == 3
    assert sp.limit(num / den, X, 1) == sp.Rational(-1, 3)

    def h(x):
        return (x ** 4 - 3 * x ** 2 + 1) / (4 - x)

    approach = [0.9, 0.99, 0.999, 1.001, 1.01, 1.1]
    assert f"{h(0.9):.4f}" == "-0.2496" and f"{h(1.1):.4f}" == "-0.4020"
    assert sp.limit((X ** 2 - 1) / (X - 1), X, 1) == 2

    fig = new_figure("Limit laws: take the limit of each piece, then combine",
                     "If lim f and lim g exist, sums, differences, products, constant multiples, powers and roots "
                     "pass through the limit; a quotient needs lim g ≠ 0.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(1, 3, 300)
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2, label="f = x²  → 4")
    ax.plot(xs, 3 * xs + 1, color=AQUA, linewidth=2, label="g = 3x + 1  → 7")
    ax.plot(xs, xs ** 2 + 3 * xs + 1, color=ORANGE, linewidth=2.2, label="f + g  → 11 = 4 + 7")
    for v, c in [(4, BLUE), (7, AQUA), (11, ORANGE)]:
        dot(ax, 2, v, color=c, size=7)
    ax.axvline(2, color=INK2, linestyle=":", linewidth=1.2)
    style(ax, "Sum rule at x = 2", "x", "y", axes_lines=False)
    ax.set_ylim(0, 22)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0.5, 1.5, 300)
    ax.plot(xs, xs ** 4 - 3 * xs ** 2 + 1, color=BLUE, linewidth=2, label="numerator x⁴ − 3x² + 1 → −1")
    ax.plot(xs, 4 - xs, color=AQUA, linewidth=2, label="denominator 4 − x → 3 (≠ 0)")
    dot(ax, 1, -1, color=BLUE, size=7)
    dot(ax, 1, 3, color=AQUA, size=7)
    ax.axvline(1, color=INK2, linestyle=":", linewidth=1.2)
    style(ax, "Worked example: the two pieces at x = 1", "x", "y")
    ax.set_ylim(-1.6, 4.6)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "The rules, checked with SymPy", [
        "f = x², g = 3x + 1, x → 2:",
        "`sum         4 + 7 = 11`",
        "`difference  4 - 7 = -3`",
        "`product     4 · 7 = 28`",
        "`quotient    2 / 4 = 1/2   (x²+1)/(x+3), x→1`",
        "`constant    5 · 1 = 5     5(x²-3), x→2`",
        "`power       3³ = 27       (x+2)³, x→1`",
        "`root        √4 = 2        √(x+3), x→1`",
        "",
        "Each time: limit of the combination",
        "= combination of the limits.",
    ], fontsize=12, spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.5, 1.5, 400)
    ax.plot(xs, h(xs), color=BLUE, linewidth=2.2, label="(x⁴ − 3x² + 1)/(4 − x)")
    ax.axhline(-1 / 3, color=ORANGE, linestyle="--", linewidth=1.4, label="−1/3 = (−1)/3")
    ax.axvline(1, color=INK2, linestyle=":", linewidth=1.2)
    for xv in approach:
        dot(ax, xv, h(xv), color=AQUA if xv > 1 else BLUE, size=6)
    ax.annotate(f"h(0.999) = {h(0.999):.4f}\nh(1.001) = {h(1.001):.4f}", (1.0, -1 / 3),
                textcoords="offset points", xytext=(18, 40), fontsize=10)
    style(ax, "…so the quotient → −1/3", "x", "y", axes_lines=False)
    ax.legend(loc="lower left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0, 2, 300)
    ax.plot(xs, xs ** 2 - 1, color=BLUE, linewidth=2, label="x² − 1 → 0")
    ax.plot(xs, xs - 1, color=AQUA, linewidth=2, label="x − 1 → 0")
    dot(ax, 1, 0, color=ORANGE, size=9)
    ax.annotate("0/0: quotient rule\ndoes not apply", (1, 0), textcoords="offset points",
                xytext=(-130, 40), fontsize=10.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.axvline(1, color=INK2, linestyle=":", linewidth=1.2)
    style(ax, "Warning: denominator limit 0", "x", "y")
    ax.set_ylim(-1.3, 3.3)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(0, 2, 300)
    ax.plot(xs, xs + 1, color=BLUE, linewidth=2.2, label="(x² − 1)/(x − 1) = x + 1, x ≠ 1")
    hole(ax, 1, 2)
    ax.annotate("simplify first:\nlimit = 2", (1, 2), textcoords="offset points",
                xytext=(24, -50), fontsize=10.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    style(ax, "…but cancelling the factor rescues it", "x", "y")
    ax.set_ylim(0, 3.5)
    ax.legend(loc="upper left", fontsize=9.5)
    save_limits(fig, name="03_limit_laws.png")


# ================================================================ integrals 00
def sheet_00_integral_as_area():
    def f(x):
        return x ** 2 + 1

    a, b = 0, 2

    def right_sum(n):
        dx = (b - a) / n
        x_right = np.linspace(a + dx, b, n)
        return np.sum(f(x_right) * dx)

    assert right_sum(8) == 5.1875
    assert [f"{right_sum(n):.6f}" for n in (20, 100)] == ["4.870000", "4.706800"]
    exact = sp.integrate(X ** 2 + 1, (X, 0, 2))
    assert exact == sp.Rational(14, 3)
    assert sp.integrate(2 * X, (X, 1, 3)) == 8

    fig = new_figure("The definite integral: rectangles in the limit, an antiderivative in practice",
                     "∫ₐᵇ f(x) dx = lim Σ f(xᵢ) Δx = F(b) − F(a) whenever F′ = f.  "
                     "Example: f(x) = x² + 1 on [0, 2] has area 14/3 ≈ 4.667.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    n = 8
    dx = (b - a) / n
    x_right = np.linspace(a + dx, b, n)
    ax.bar(x_right - dx, f(x_right), width=dx, align="edge", color=BLUE_LIGHT,
           edgecolor=BLUE, linewidth=1, zorder=2, label=f"8 rectangles: {right_sum(8):.4f}")
    xs = np.linspace(0, 2, 300)
    ax.plot(xs, f(xs), color=INK, linewidth=2.2, zorder=3, label="f(x) = x² + 1")
    style(ax, "Right-endpoint Riemann sum, n = 8", "x", "y")
    ax.set_ylim(0, 6.2)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    ns = np.unique(np.round(np.logspace(np.log10(2), np.log10(400), 40)).astype(int))
    ax.plot(ns, [right_sum(k) for k in ns], color=BLUE, linewidth=2, label="right-endpoint sum")
    ax.axhline(14 / 3, color=ORANGE, linestyle="--", linewidth=1.5, label="exact 14/3 ≈ 4.6667")
    for k in (8, 20, 100):
        dot(ax, k, right_sum(k), color=BLUE, size=7)
        ax.annotate(f"n = {k}: {right_sum(k):.4f}", (k, right_sum(k)), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xscale("log")
    style(ax, "Thinner rectangles → exact area", "number of rectangles n (log scale)", "sum",
          axes_lines=False)
    ax.set_ylim(4.4, 7.2)
    ax.legend(loc="upper right", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "Two kinds of integral", [
        "Indefinite: a family of functions",
        "    ∫ f(x) dx = F(x) + C,  where F′ = f",
        "Definite: one number, the signed area",
        "    ∫ₐᵇ f(x) dx = F(b) − F(a)",
        "",
        "Rules for the bounds:",
        "    zero width gives 0:  ∫ₐᵃ f dx = 0",
        "    swapping the bounds flips the sign",
        "    adjacent intervals [a, b], [b, c] add",
        "Linearity: ∫ (k f + g) dx = k ∫ f dx + ∫ g dx",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2.5, 2.5, 300)
    for C, color in [(3, AQUA), (0, BLUE), (-1, ORANGE)]:
        ax.plot(xs, xs ** 2 + C, color=color, linewidth=2, label=f"x² {'+' if C >= 0 else '−'} {abs(C)}")
        seg = np.linspace(0.4, 1.6, 2)
        ax.plot(seg, 1 + C + 2 * (seg - 1), color=INK, linewidth=1.3)
        dot(ax, 1, 1 + C, color=color, size=6)
    ax.text(1.25, -2.4, "same slope 2 at x = 1", fontsize=10, color=INK2)
    style(ax, "∫ 2x dx = x² + C: a family of curves", "x", "y")
    ax.set_ylim(-3, 10)
    ax.legend(loc="upper center", fontsize=9.5, ncol=3)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0, 3.4, 300)
    shade = np.linspace(1, 3, 100)
    ax.fill_between(shade, 2 * shade, color=BLUE_LIGHT, alpha=0.8, linewidth=0,
                    label="area under 2x on [1, 3] = 8")
    ax.plot(xs, 2 * xs, color=BLUE, linewidth=2, label="f(x) = 2x")
    ax.plot(xs, xs ** 2, color=ORANGE, linewidth=2, label="F(x) = x²")
    ax.annotate("", xy=(3.15, 9), xytext=(3.15, 1),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.6))
    ax.plot([1, 3.15], [1, 1], ":", color=ORANGE)
    ax.plot([3, 3.15], [9, 9], ":", color=ORANGE)
    ax.text(3.2, 4.6, "F(3) − F(1)\n= 9 − 1 = 8", fontsize=10, color=ORANGE)
    style(ax, "FTC: the area is the rise of F", "x", "y")
    ax.set_xlim(0, 4.3)
    ax.set_ylim(0, 12)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.add_patch(Rectangle((0, 0), 3, 2, color=BLUE_LIGHT, alpha=0.9, linewidth=0))
    ax.add_patch(Rectangle((0, -3), 2, 3, color=ORANGE_LIGHT, alpha=0.9, linewidth=0))
    ax.plot([0, 3], [2, 2], color=BLUE, linewidth=2.2, label="g(x) = 2 on [0, 3]")
    ax.plot([0, 2], [-3, -3], color=ORANGE, linewidth=2.2, label="f(x) = −3 on [0, 2]")
    ax.text(1.5, 1, "∫₀³ 2 dx = +6", ha="center", va="center", fontsize=12, color=INK)
    ax.text(1.0, -1.5, "∫₀² −3 dx = −6", ha="center", va="center", fontsize=12, color=INK)
    style(ax, "Signed area: below the axis is negative", "x", "y")
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylim(-4, 4)
    ax.legend(loc="upper right", fontsize=9.5)
    save_integrals(fig, name="00_integral_as_area.png")


# ================================================================ integrals 01
def sheet_01_power_rule():
    examples = [2 * X, X ** 3 / 3, -X ** (-4) / 4, sp.Rational(2, 3) * X ** sp.Rational(3, 2),
                5 * X ** 3 / 3 + X ** 2 / 2 - 2 * X]
    integrands = [2, X ** 2, X ** (-5), sp.sqrt(X), 5 * X ** 2 + X - 2]
    assert all(sp.simplify(sp.diff(F, X) - f) == 0 for F, f in zip(examples, integrands))
    xp = sp.symbols("x", positive=True)
    assert sp.integrate(xp ** 2, (xp, 0, 2)) == sp.Rational(8, 3)

    fig = new_figure("Integrating a power: raise the exponent by one, divide by the new exponent",
                     "∫ xⁿ dx = xⁿ⁺¹/(n + 1) + C for n ≠ −1,   ∫ k dx = kx + C,   "
                     "and ∫ 1/x dx = ln|x| + C fills the gap at n = −1.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-3, 3, 400)
    ax.plot(xs, 2 * xs, color=BLUE, linewidth=2, label="f(x) = 2x")
    ax.plot(xs, xs ** 2, color=ORANGE, linewidth=2, label="F(x) = x²")
    for x0 in (-2, -1, 1, 2):
        seg = np.array([x0 - 0.6, x0 + 0.6])
        ax.plot(seg, x0 ** 2 + 2 * x0 * (seg - x0), color=INK, linewidth=1.2,
                label="tangent of F: slope = f" if x0 == -2 else None)
        dot(ax, x0, x0 ** 2, color=ORANGE, size=6)
        dot(ax, x0, 2 * x0, color=BLUE, size=6)
    style(ax, "F′ = f: an antiderivative's slope is f", "x", "y")
    ax.set_ylim(-6.5, 9)
    ax.legend(loc="lower right", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-1.5, 1.3, 400)
    p = 5 * xs ** 2 + xs - 2
    P = 5 * xs ** 3 / 3 + xs ** 2 / 2 - 2 * xs
    roots = np.sort(np.roots([5, 1, -2]).real)
    ax.plot(xs, p, color=BLUE, linewidth=2, label="p(x) = 5x² + x − 2")
    ax.plot(xs, P, color=ORANGE, linewidth=2, label="P(x) = 5x³/3 + x²/2 − 2x")
    for r in roots:
        P_r = 5 * r ** 3 / 3 + r ** 2 / 2 - 2 * r
        ax.plot([r, r], [0, P_r], ":", color=INK2)
        dot(ax, r, 0, color=BLUE, size=6)
        dot(ax, r, P_r, color=ORANGE, size=6)
    ax.text(-1.45, -3.3, "p = 0 exactly where P turns", fontsize=10, color=INK2)
    style(ax, "Polynomials: integrate term by term", "x", "y")
    ax.set_ylim(-3.6, 6.5)
    ax.legend(loc="upper center", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "Why the rule works", [
        "Differentiate the answer and get xⁿ back:",
        r"$\frac{d}{dx}\,\frac{x^{n+1}}{n+1} = \frac{(n+1)\,x^{n}}{n+1} = x^{n}$",
        "",
        "∫ x² dx = x³/3 + C",
        "∫ x⁻⁵ dx = −x⁻⁴/4 + C",
        "∫ x^(1/2) dx = (2/3) x^(3/2) + C",
        "∫ (5x² + x − 2) dx = 5x³/3 + x²/2 − 2x + C",
        "",
        "n = −1 would divide by n + 1 = 0.",
    ], spacing=0.087)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.05, 4, 400)
    for n, color in [(-0.5, BLUE_LIGHT), (-0.8, AQUA), (-0.95, BLUE)]:
        ax.plot(xs, (xs ** (n + 1) - 1) / (n + 1), color=color, linewidth=2,
                label=f"n = {n}: (xⁿ⁺¹ − 1)/(n + 1)")
    ax.plot(xs, np.log(xs), "--", color=ORANGE, linewidth=2.2, label="ln x  (the n → −1 limit)")
    style(ax, "As n → −1 the power rule turns into ln x", "x", "antiderivative with F(1) = 0")
    ax.set_ylim(-3.5, 2.6)
    ax.legend(loc="lower right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    xl = np.linspace(-4, -0.15, 300)
    xr = np.linspace(0.15, 4, 300)
    ax.plot(xl, 1 / xl, color=BLUE, linewidth=2, label="f(x) = 1/x")
    ax.plot(xr, 1 / xr, color=BLUE, linewidth=2)
    ax.plot(xl, np.log(np.abs(xl)), color=ORANGE, linewidth=2, label="F(x) = ln|x|")
    ax.plot(xr, np.log(xr), color=ORANGE, linewidth=2)
    slope_check(ax, xl, np.log(np.abs(xl)), label="slope of ln|x| (numerical)")
    slope_check(ax, xr, np.log(xr), label=None)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "ln|x| works on both sides of 0", "x", "y")
    ax.set_ylim(-5, 5)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(0, 2.4, 300)
    shade = np.linspace(0, 2, 200)
    ax.fill_between(shade, shade ** 2, color=BLUE_LIGHT, alpha=0.85, linewidth=0,
                    label=f"area = 8/3 ≈ {trapezoid(shade ** 2, shade):.4f}")
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2, label="f(x) = x²")
    ax.plot(xs, xs ** 3 / 3, color=ORANGE, linewidth=2, label="F(x) = x³/3")
    dot(ax, 2, 8 / 3, color=ORANGE, size=7)
    ax.annotate("F(2) − F(0) = 8/3", (2, 8 / 3), textcoords="offset points", xytext=(-130, 20),
                fontsize=10.5, color=ORANGE)
    style(ax, "Definite: ∫₀² x² dx = [x³/3]₀² = 8/3", "x", "y")
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper left", fontsize=9.5)
    save_integrals(fig, name="01_power_rule.png")


# ================================================================ integrals 02
def sheet_02_exp_trig_integrals():
    assert sp.simplify(sp.integrate(2 ** (3 * X), X) - 2 ** (3 * X) / (3 * sp.log(2))) == 0
    assert sp.integrate(sp.sin(X), X) == -sp.cos(X) and sp.integrate(sp.cos(X), X) == sp.sin(X)
    assert sp.simplify(sp.diff(sp.tan(X), X) - sp.sec(X) ** 2) == 0
    assert sp.integrate(sp.sin(X), (X, 0, sp.pi)) == 2

    fig = new_figure("Immediate integrals: every derivative rule, read backwards",
                     "∫ a^(kx) dx = a^(kx)/(k ln a) + C,   ∫ eˣ dx = eˣ + C,   ∫ sin x dx = −cos x + C,   "
                     "∫ cos x dx = sin x + C,   ∫ sec² x dx = tan x + C")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 2, 400)
    ax.plot(xs, np.exp(xs), color=BLUE, linewidth=2.4, label="f(x) = eˣ")
    ax.plot(xs, np.exp(xs) + 1, color=ORANGE, linewidth=2, label="F(x) = eˣ + 1")
    slope_check(ax, xs, np.exp(xs) + 1, label="slope of F (numerical)")
    style(ax, "eˣ is its own antiderivative", "x", "y")
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-1, 1, 400)
    f = 2 ** (3 * xs)
    F = 2 ** (3 * xs) / (3 * log(2))
    ax.plot(xs, f, color=BLUE, linewidth=2.4, label="f(x) = 2^(3x)")
    ax.plot(xs, F, color=ORANGE, linewidth=2, label="F(x) = 2^(3x) / (3 ln 2)")
    slope_check(ax, xs, F)
    ax.text(-0.95, 5.2, f"a = 2, k = 3:  k ln a = {3 * log(2):.3f}", fontsize=10.5, color=INK2)
    style(ax, "∫ a^(kx) dx: divide by k ln a", "x", "y")
    ax.set_ylim(0, 8.5)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "The sine–cosine cycle", [
        "Differentiate (go down):",
        "sin x → cos x → −sin x → −cos x → sin x",
        "",
        "Integrate (go up):",
        "∫ sin x dx = −cos x + C",
        "∫ cos x dx = sin x + C",
        "",
        "sin² + cos² = 1  ⇒  tan² + 1 = sec²,",
        "so (tan x)′ = sec² x and ∫ sec² x dx = tan x + C;",
        "likewise ∫ csc² x dx = −cot x + C.",
    ], spacing=0.082)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0, 2 * pi, 400)
    ax.plot(xs, np.sin(xs), color=BLUE, linewidth=2.4, label="sin x")
    ax.plot(xs, np.cos(xs), color=AQUA, linewidth=2, label="cos x = d/dx sin x  (shift left π/2)")
    ax.plot(xs, -np.cos(xs), "--", color=ORANGE, linewidth=2, label="−cos x = ∫ sin x dx  (shift right π/2)")
    ax.set_xticks(np.linspace(0, 2 * pi, 5), ["0", "π/2", "π", "3π/2", "2π"])
    style(ax, "Derivative and integral shift by π/2", "x", "y")
    ax.set_ylim(-1.3, 2.3)
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-1.35, 1.35, 400)
    ax.plot(xs, 1 / np.cos(xs) ** 2, color=BLUE, linewidth=2.4, label="f(x) = sec² x")
    ax.plot(xs, np.tan(xs), color=ORANGE, linewidth=2, label="F(x) = tan x")
    slope_check(ax, xs, np.tan(xs))
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "sec² x is the slope of tan x", "x", "y")
    ax.set_ylim(-5, 8)
    ax.legend(loc="upper center", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(0, pi, 400)
    area = trapezoid(np.sin(xs), xs)
    assert abs(area - 2) < 1e-4
    ax.fill_between(xs, np.sin(xs), color=BLUE_LIGHT, alpha=0.85, linewidth=0, label="area = 2")
    ax.plot(xs, np.sin(xs), color=BLUE, linewidth=2.2, label="sin x")
    ax.plot(xs, -np.cos(xs), color=ORANGE, linewidth=2, label="F(x) = −cos x")
    dot(ax, 0, -1, color=ORANGE, size=7)
    dot(ax, pi, 1, color=ORANGE, size=7)
    ax.text(pi / 2 + 0.25, -0.9, "F(π) − F(0) = 1 − (−1) = 2", ha="center", fontsize=10.5, color=ORANGE)
    ax.set_xticks([0, pi / 2, pi], ["0", "π/2", "π"])
    style(ax, "One arch: ∫ sin x dx over [0, π] = 2", "x", "y")
    ax.set_ylim(-1.2, 1.8)
    ax.legend(loc="upper left", fontsize=9.5)
    save_integrals(fig, name="02_exp_trig_integrals.png")


# ================================================================ integrals 03
def sheet_03_u_substitution():
    I_x = sp.integrate(2 * X * 3 ** (X ** 2 + 5), X)
    assert sp.simplify(I_x - 3 ** (X ** 2 + 5) / sp.log(3)) == 0
    exact = (3 ** 6 - 3 ** 5) / log(3)
    xs = np.linspace(0, 1, 4001)
    us = np.linspace(5, 6, 4001)
    area_x = trapezoid(2 * xs * 3.0 ** (xs ** 2 + 5), xs)
    area_u = trapezoid(3.0 ** us, us)
    assert abs(area_x - exact) < 1e-2 and abs(area_u - exact) < 1e-2
    assert sp.simplify(sp.diff(-sp.log(sp.cos(X)), X) - sp.tan(X)) == 0
    xp = sp.symbols("x", positive=True)
    F_nested = 10 * sp.sqrt(xp + 3) - 40 * sp.log(4 + sp.sqrt(xp + 3))
    assert sp.simplify(sp.diff(F_nested, xp) - 5 / (4 + sp.sqrt(xp + 3))) == 0

    fig = new_figure("u-substitution: the chain rule run backwards",
                     "∫ f′(g(x)) g′(x) dx = f(g(x)) + C.  Set u = g(x) and du = g′(x) dx, "
                     "integrate in u, then substitute back.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(0, 1, 300)
    ax.fill_between(xx, 2 * xx * 3.0 ** (xx ** 2 + 5), color=BLUE_LIGHT, alpha=0.85, linewidth=0,
                    label=f"area = {area_x:.1f}")
    ax.plot(xx, 2 * xx * 3.0 ** (xx ** 2 + 5), color=BLUE, linewidth=2.2,
            label="2x · 3^(x² + 5)")
    style(ax, "In x: ∫₀¹ 2x · 3^(x²+5) dx", "x", "integrand")
    ax.set_ylim(0, 1700)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    uu = np.linspace(4.8, 6.2, 300)
    shade = np.linspace(5, 6, 200)
    ax.fill_between(shade, 3.0 ** shade, color=ORANGE_LIGHT, alpha=0.9, linewidth=0,
                    label=f"area = {area_u:.1f}")
    ax.plot(uu, 3.0 ** uu, color=ORANGE, linewidth=2.2, label="3ᵘ")
    ax.axvline(5, color=INK2, linestyle=":", linewidth=1.2)
    ax.axvline(6, color=INK2, linestyle=":", linewidth=1.2)
    ax.text(5.03, 1120, "x = 0 → u = 5", fontsize=10, color=INK2)
    ax.text(5.97, 1120, "x = 1 → u = 6", ha="right", fontsize=10, color=INK2)
    style(ax, "In u = x² + 5, du = 2x dx: ∫₅⁶ 3ᵘ du", "u", "integrand")
    ax.set_xlim(4.8, 6.2)
    ax.set_ylim(0, 1700)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "The recipe", [
        "1. Choose u: an inner expression whose",
        "    derivative also appears in the integrand.",
        "2. du = u′(x) dx  (or solve for x, find dx).",
        "3. Rewrite everything in u; no x may remain.",
        "4. Integrate:  ∫ 3ᵘ du = 3ᵘ / ln 3 + C.",
        "5. Substitute back:  3^(x²+5) / ln 3 + C.",
        "",
        f"Same area either way: (3⁶ − 3⁵)/ln 3 = {exact:.1f}",
        "(limits change with u: 0 → 5, 1 → 6).",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    xx = np.linspace(-1.4, 1.4, 400)
    F = -np.log(np.abs(np.cos(xx)))
    ax.plot(xx, np.tan(xx), color=BLUE, linewidth=2.4, label="f(x) = tan x = sin x / cos x")
    ax.plot(xx, F, color=ORANGE, linewidth=2, label="F(x) = −ln|cos x|  (u = cos x)")
    slope_check(ax, xx, F)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    style(ax, "∫ tan x dx = −ln|cos x| + C", "x", "y")
    ax.set_ylim(-6, 6)
    ax.legend(loc="lower right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    xx = np.linspace(-3, 12, 400)
    f = 5 / (4 + np.sqrt(xx + 3))
    F = 10 * np.sqrt(xx + 3) - 40 * np.log(4 + np.sqrt(xx + 3))
    ax.plot(xx, f, color=BLUE, linewidth=2.4, label="f(x) = 5 / (4 + √(x + 3))  (left axis)")
    slope_check(ax, xx, F)
    style(ax, "Nested: u = √(x+3), then v = 4 + u", "x", "f(x)", axes_lines=False)
    ax.set_ylim(0.4, 1.5)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(xx, F, color=ORANGE, linewidth=2,
             label="F(x) = 10√(x+3) − 40 ln(4 + √(x+3))  (right axis)")
    ax2.set_ylabel("F(x)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.set_ylim(-60, -38)
    handles = ax.get_legend_handles_labels()
    handles2 = ax2.get_legend_handles_labels()
    ax.legend(handles[0] + handles2[0], handles[1] + handles2[1], loc="upper center", fontsize=8.5)

    panel(fig.add_subplot(gs[1, 2]), "Choosing u well", [
        "∫ e^(3x²+2) · 2x dx",
        "  u = 2x:  ½ ∫ e^(¾u² + 2) u du   (no easier)",
        "  u = 3x² + 2, du = 6x dx:",
        "      ⅓ ∫ eᵘ du = ⅓ e^(3x²+2) + C",
        "",
        "∫ dx/(eˣ + 1) with u = eˣ + 1, two routes:",
        "  du = eˣ dx,  or  x = ln(u − 1), dx = du/(u−1)",
        "  both give  x − ln(eˣ + 1) + C",
        "",
        "If the new integral looks harder, pick another u.",
    ], fontsize=12, spacing=0.082)
    save_integrals(fig, name="03_u_substitution.png")


# ================================================================ integrals 04
def sheet_04_trig_substitution():
    assert sp.integrate(1 / sp.sqrt(9 - X ** 2), X) == sp.asin(X / 3)
    assert sp.simplify(sp.diff(sp.log(X + sp.sqrt(X ** 2 + 4)), X) - 1 / sp.sqrt(X ** 2 + 4)) == 0
    xp = sp.symbols("x", positive=True)
    assert sp.simplify(sp.diff(sp.log(xp + sp.sqrt(xp ** 2 - 1)), xp) - 1 / sp.sqrt(xp ** 2 - 1)) == 0
    assert sp.integrate(sp.sqrt(9 - X ** 2), (X, -3, 3)) == 9 * sp.pi / 2

    fig = new_figure("Trigonometric substitution: let a Pythagorean identity remove the root",
                     "√(a² − x²): x = a sin θ → a cos θ    √(a² + x²): x = a tan θ → a sec θ    "
                     "√(x² − a²): x = a sec θ → a tan θ")
    gs = grid(fig, top=0.84, wspace=0.3)

    def right_triangle(ax, ox, oy, adj, opp, labels, theta_label="θ", color=BLUE):
        """Right angle at (ox + adj, oy); θ at the origin vertex (ox, oy)."""
        pts = np.array([[ox, oy], [ox + adj, oy], [ox + adj, oy + opp]])
        ax.add_patch(Polygon(pts, closed=True, facecolor=to_rgba(BLUE_LIGHT, 0.45), edgecolor=color,
                             linewidth=2))
        s = 0.12 * min(adj, opp) + 0.05
        ax.plot([ox + adj - s, ox + adj - s, ox + adj], [oy, oy + s, oy + s], color=INK, linewidth=1)
        hyp, adj_l, opp_l = labels
        ax.text(ox + adj / 2, oy - 0.25, adj_l, ha="center", va="top", fontsize=10.5)
        ax.text(ox + adj + 0.12, oy + opp / 2, opp_l, ha="left", va="center", fontsize=10.5)
        ax.text(ox + adj / 2 - 0.25, oy + opp / 2 + 0.2, hyp, ha="right", va="bottom", fontsize=10.5)
        ax.text(ox + 0.45, oy + 0.1, theta_label, fontsize=13, color=ORANGE)

    ax = fig.add_subplot(gs[0, 0])
    a, th = 3, 0.75
    t = np.linspace(0, pi, 300)
    ax.plot(a * np.cos(t), a * np.sin(t), color=MUTED, linewidth=1.5, label="y = √(9 − x²)… circle r = 3")
    xa, ya = a * np.cos(th), a * np.sin(th)
    pts = np.array([[0, 0], [xa, 0], [xa, ya]])
    ax.add_patch(Polygon(pts, closed=True, facecolor=to_rgba(BLUE_LIGHT, 0.45), edgecolor=BLUE,
                         linewidth=2))
    ax.plot([xa - 0.2, xa - 0.2, xa], [0, 0.2, 0.2], color=INK, linewidth=1)
    ax.text(0.45, 0.1, "θ", fontsize=14, color=ORANGE)
    ax.text(xa / 2 - 0.15, ya / 2 + 0.15, "hyp a = 3", ha="right", fontsize=10.5, rotation=36)
    ax.text(xa + 0.12, ya / 2, "opp x = 3 sin θ", ha="left", va="center", fontsize=10.5)
    ax.text(xa / 2, -0.2, "adj √(9 − x²) = 3 cos θ", ha="center", va="top", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 5.2)
    ax.set_ylim(-0.9, 3.4)
    ax.axis("off")
    ax.set_title("Sine case: the reference triangle", pad=10)

    ax = fig.add_subplot(gs[0, 1])
    right_triangle(ax, 0, 0, 2.0, 2.4, ("√(x² + 4)", "adj 2", "opp x"))
    right_triangle(ax, 4.4, 0, 1.2, 2.4, ("hyp x", "adj 1", "√(x² − 1)"))
    ax.text(1.0, 3.25, "x = 2 tan θ", ha="center", fontsize=11.5, weight="bold", color=BLUE)
    ax.text(5.0, 3.25, "x = sec θ", ha="center", fontsize=11.5, weight="bold", color=BLUE)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 7.6)
    ax.set_ylim(-0.9, 3.7)
    ax.axis("off")
    ax.set_title("Tangent and secant cases", pad=10)

    panel(fig.add_subplot(gs[0, 2]), "The three worked examples", [
        "∫ dx/√(9 − x²):  x = 3 sin θ, root = 3 cos θ",
        "   ∫ dθ = θ  →  arcsin(x/3) + C",
        "∫ dx/√(x² + 4):  x = 2 tan θ, root = 2 sec θ",
        "   ∫ sec θ dθ  →  ln|x + √(x² + 4)| + C",
        "∫ dx/√(x² − 1):  x = sec θ, root = tan θ",
        "   ∫ sec θ dθ  →  ln|x + √(x² − 1)| + C",
        "",
        "Read sin, cos, tan, sec of θ off the triangle",
        "to return from θ to x.",
    ], fontsize=12, spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    x1 = np.linspace(-2.95, 2.95, 400)
    x2 = np.linspace(-4, 4, 400)
    x3 = np.linspace(1.02, 4, 300)
    ax.plot(x1, 1 / np.sqrt(9 - x1 ** 2), color=BLUE, linewidth=2.2, label="1/√(9 − x²)")
    ax.plot(x2, 1 / np.sqrt(x2 ** 2 + 4), color=AQUA, linewidth=2.2, label="1/√(x² + 4)")
    ax.plot(x3, 1 / np.sqrt(x3 ** 2 - 1), color=ORANGE, linewidth=2.2, label="1/√(x² − 1)")
    style(ax, "The three integrands", "x", "f(x)")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    F1 = np.arcsin(x1 / 3)
    F2 = np.log(x2 + np.sqrt(x2 ** 2 + 4))
    F3 = np.log(x3 + np.sqrt(x3 ** 2 - 1))
    ax.plot(x1, F1, color=BLUE, linewidth=2.2, label="arcsin(x/3)")
    ax.plot(x2, F2, color=AQUA, linewidth=2.2, label="ln(x + √(x² + 4))")
    ax.plot(x3, F3, color=ORANGE, linewidth=2.2, label="ln(x + √(x² − 1))")
    for xs_, F_, f_ in [(x1, F1, 1 / np.sqrt(9 - x1 ** 2)), (x2, F2, 1 / np.sqrt(x2 ** 2 + 4)),
                        (x3, F3, 1 / np.sqrt(x3 ** 2 - 1))]:
        inner = slice(5, -5)
        assert np.allclose(np.gradient(F_, xs_)[inner], f_[inner], rtol=2e-2)
    style(ax, "Their antiderivatives (slopes checked)", "x", "F(x)")
    ax.set_ylim(-2.4, 2.9)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(-3, 3, 2001)
    ys = np.sqrt(9 - xs ** 2)
    area = trapezoid(ys, xs)
    assert abs(area - 9 * pi / 2) < 1e-2
    ax.fill_between(xs, ys, color=BLUE_LIGHT, alpha=0.85, linewidth=0,
                    label=f"area = 9π/2 ≈ {9 * pi / 2:.4f}")
    ax.plot(xs, ys, color=BLUE, linewidth=2.2, label="y = √(9 − x²)")
    ax.text(0, 1.1, "[x/2 · √(9−x²) + 9/2 · arcsin(x/3)]₋₃³", ha="center", fontsize=10)
    ax.set_aspect("equal")
    style(ax, "Check: the semicircle of radius 3", "x", "y")
    ax.set_ylim(0, 4.6)
    ax.legend(loc="upper left", fontsize=9.5)
    save_integrals(fig, name="04_trig_substitution.png")


# ================================================================ integrals 05
def sheet_05_integration_by_parts():
    xp = sp.symbols("x", positive=True)
    assert sp.simplify(sp.integrate(xp * sp.exp(xp), xp) - (xp - 1) * sp.exp(xp)) == 0
    assert sp.simplify(sp.integrate(sp.log(xp), xp) - (xp * sp.log(xp) - xp)) == 0
    cyc = (-sp.cos(X) * 2 ** X + sp.log(2) * sp.sin(X) * 2 ** X) / (sp.log(2) ** 2 + 1)
    assert sp.simplify(sp.integrate(sp.sin(X) * 2 ** X, X) - cyc) == 0
    assert sp.simplify(sp.integrate(xp * sp.exp(xp) / (xp + 1) ** 2, xp) - sp.exp(xp) / (xp + 1)) == 0
    int_u_dv = float(sp.integrate(xp * sp.exp(xp), (xp, 0, 1)))       # ∫ u dv with u = x, v = eˣ
    int_v_du = float(sp.integrate(sp.exp(xp), (xp, 0, 1)))            # ∫ v du
    assert abs(int_u_dv - 1) < 1e-12 and abs(int_u_dv + int_v_du - e) < 1e-12

    fig = new_figure("Integration by parts: the product rule, reversed",
                     "∫ u dv = uv − ∫ v du.  Choose u to get simpler when differentiated (LIATE) and dv easy "
                     "to integrate; the new integral must be easier.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 1, 300)
    u, v = t, np.exp(t)
    ax.fill_between(u, v, color=BLUE_LIGHT, alpha=0.9, linewidth=0,
                    label=f"∫ v du = ∫₀¹ eˣ dx = e − 1 ≈ {int_v_du:.3f}")
    ax.fill_betweenx(v, 0, u, color=ORANGE_LIGHT, alpha=0.95, linewidth=0,
                     label=f"∫ u dv = ∫₀¹ x eˣ dx = {int_u_dv:.0f}")
    ax.plot(u, v, color=INK, linewidth=2.2)
    ax.add_patch(Rectangle((0, 0), 1, e, fill=False, edgecolor=INK2, linewidth=1.2, linestyle="--"))
    ax.text(1.03, e, "uv = 1 · e", va="center", fontsize=10.5, color=INK2)
    ax.text(0.5, 0.45, "u = x,  v = eˣ", ha="center", fontsize=10.5)
    style(ax, "The two pieces fill the box uv", "u = x", "v = eˣ", axes_lines=False)
    ax.set_xlim(0, 1.35)
    ax.set_ylim(0, 3.9)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-3, 2, 400)
    F = np.exp(xs) * (xs - 1)
    ax.plot(xs, xs * np.exp(xs), color=BLUE, linewidth=2.4, label="f(x) = x eˣ")
    ax.plot(xs, F, color=ORANGE, linewidth=2, label="F(x) = eˣ(x − 1)")
    slope_check(ax, xs, F)
    style(ax, "u = x, dv = eˣ dx:  ∫ x eˣ dx = eˣ(x − 1) + C", "x", "y")
    ax.set_ylim(-1.5, 15.5)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[0, 2]), "Where it comes from", [
        "Product rule:  (uv)′ = u′v + uv′",
        "Integrate and rearrange:",
        "    ∫ u v′ dx = uv − ∫ u′ v dx",
        "",
        "LIATE: pick u in this order",
        "Logarithmic, Inverse trig, Algebraic,",
        "Trigonometric, Exponential; the rest is dv.",
        "",
        "∫ ln x dx:  u = ln x, dv = dx  →  x ln x − x + C",
        "∫ x sin x dx:  u = x  →  −x cos x + sin x + C",
    ], fontsize=12, spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.05, 4, 400)
    F = xs * (np.log(xs) - 1)
    ax.plot(xs, np.log(xs), color=BLUE, linewidth=2.4, label="f(x) = ln x  (= ln x · 1)")
    ax.plot(xs, F, color=ORANGE, linewidth=2, label="F(x) = x(ln x − 1)")
    slope_check(ax, xs, F)
    style(ax, "u = ln x, dv = dx: ∫ ln x dx", "x", "y")
    ax.set_ylim(-3, 2.2)
    ax.legend(loc="lower right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-2 * pi, 2 * pi, 600)
    l2 = log(2)
    F = 2 ** xs * (l2 * np.sin(xs) - np.cos(xs)) / (l2 ** 2 + 1)
    ax.plot(xs, np.sin(xs) * 2 ** xs, color=BLUE, linewidth=2.4, label="f(x) = sin x · 2ˣ")
    ax.plot(xs, F, color=ORANGE, linewidth=2, label="F = 2ˣ(ln2 sin x − cos x)/(ln²2 + 1)")
    slope_check(ax, xs, F)
    ax.set_xticks(np.linspace(-2 * pi, 2 * pi, 5), ["−2π", "−π", "0", "π", "2π"])
    style(ax, "Cyclic trick: I reappears, solve for it", "x", "y")
    ax.set_ylim(-60, 60)
    ax.legend(loc="lower left", fontsize=9)

    panel(fig.add_subplot(gs[1, 2]), "Guidelines, not rules", [
        "∫ x eˣ / (x + 1)² dx",
        "",
        "Habit: u = x/(x + 1)², dv = eˣ dx",
        "→ a new integral with (x + 1)³ below: harder.",
        "",
        "Better: u = x eˣ,  dv = dx/(x + 1)²",
        "du = eˣ(x + 1) dx,  v = −1/(x + 1)",
        "→ −x eˣ/(x + 1) + ∫ eˣ dx = eˣ/(x + 1) + C",
        "",
        "If the new integral got harder, choose again.",
    ], fontsize=12, spacing=0.082)
    save_integrals(fig, name="05_integration_by_parts.png")


# ================================================================ integrals 06
def sheet_06_partial_fractions():
    A, B, C = sp.symbols("A B C")
    sol = sp.solve([sp.Eq(A + B, 9), sp.Eq(-A + 3 * B + C, 3), sp.Eq(A + 3 * C, -2)], [A, B, C])
    assert sol == {A: sp.Rational(70, 13), B: sp.Rational(47, 13), C: sp.Rational(-32, 13)}
    denom = X ** 3 + 2 * X ** 2 - 2 * X + 3
    assert sp.factor(denom) == (X + 3) * (X ** 2 - X + 1)
    assert sp.discriminant(X ** 2 - X + 1, X) == -3
    integrand = (9 * X ** 2 + 3 * X - 2) / denom
    ours = (sp.Rational(70, 13) * sp.log(X + 3) + sp.Rational(47, 26) * sp.log(X ** 2 - X + 1)
            - sp.Rational(17, 39) * sp.sqrt(3) * sp.atan(sp.sqrt(3) / 3 * (2 * X - 1)))
    assert sp.simplify(sp.diff(ours, X) - integrand) == 0
    q, r = sp.div(X ** 3, X ** 2 - 1, X)
    assert q == X and r == X

    f = numeric(integrand)
    a_, b_, c_ = 70 / 13, 47 / 13, -32 / 13

    fig = new_figure("Partial fractions: split a rational function into pieces you can integrate",
                     "(9x² + 3x − 2) / ((x + 3)(x² − x + 1)) = (70/13)/(x + 3) + ((47/13)x − 32/13)/(x² − x + 1);"
                     "  each piece integrates to a log or an arctan.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-6, 4, 600)
    Q = xs ** 3 + 2 * xs ** 2 - 2 * xs + 3
    ax.plot(xs, Q, color=INK, linewidth=2.4, label="Q(x) = x³ + 2x² − 2x + 3")
    ax.plot(xs, xs + 3, color=BLUE, linewidth=2, label="x + 3  (root −3)")
    ax.plot(xs, xs ** 2 - xs + 1, color=AQUA, linewidth=2, label="x² − x + 1  (no real root)")
    dot(ax, -3, 0, color=ORANGE, size=8)
    ax.text(-2.8, -8, "discriminant 1 − 4 = −3 < 0", fontsize=10, color=AQUA)
    style(ax, "Step 1: factor the denominator", "x", "y")
    ax.set_ylim(-20, 30)
    ax.legend(loc="upper left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-2.6, 5, 500)
    p1 = a_ / (xs + 3)
    p2 = (b_ * xs + c_) / (xs ** 2 - xs + 1)
    ax.plot(xs, p1, color=BLUE, linewidth=2, label="(70/13) / (x + 3)")
    ax.plot(xs, p2, color=AQUA, linewidth=2, label="((47/13)x − 32/13) / (x² − x + 1)")
    ax.plot(xs, f(xs), color=INK, linewidth=2.6, alpha=0.35, label="original integrand")
    ax.plot(xs, p1 + p2, "--", color=ORANGE, linewidth=1.8, label="sum of the two pieces")
    assert np.allclose(p1 + p2, f(xs))
    style(ax, "Step 2: the pieces add back to f", "x", "y")
    ax.set_ylim(-4.5, 13)
    ax.legend(loc="upper right", fontsize=9)

    panel(fig.add_subplot(gs[0, 2]), "Templates and coefficients", [
        "linear  ax + b        →  A/(ax + b)",
        "quadratic, no root  →  (Ax + B)/(ax² + bx + c)",
        "repeated (ax + b)ⁿ   →  A₁/(ax+b) + … + Aₙ/(ax+b)ⁿ",
        "",
        "Clear denominators and match powers of x:",
        "x²:  A + B = 9",
        "x¹: −A + 3B + C = 3",
        "x⁰:  A + 3C = −2",
        "→  A = 70/13,  B = 47/13,  C = −32/13",
    ], fontsize=12, spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2.6, 6, 500)
    t1 = 70 / 13 * np.log(np.abs(xs + 3))
    t2 = 47 / 26 * np.log(xs ** 2 - xs + 1)
    t3 = -17 * sqrt(3) / 39 * np.arctan(sqrt(3) / 3 * (2 * xs - 1))
    ax.plot(xs, t1, color=BLUE, linewidth=2, label="(70/13) ln|x + 3|   (u = x + 3)")
    ax.plot(xs, t2, color=AQUA, linewidth=2, label="(47/26) ln(x² − x + 1)   (u′/u)")
    ax.plot(xs, t3, color=ORANGE, linewidth=2, label="−(17√3/39) arctan((2x − 1)/√3)")
    ax.plot(xs, t1 + t2 + t3, color=INK, linewidth=2.6, label="sum = F(x)")
    style(ax, "Step 3: integrate each piece", "x", "y")
    ax.set_ylim(-6, 22)
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0, 6, 400)
    F = numeric(ours)(xs)
    ax.plot(xs, f(xs), color=BLUE, linewidth=2.6, label="integrand f(x)")
    slope_check(ax, xs, F, label="slope of F (numerical)", color=ORANGE)
    style(ax, "Check: the slope of F is f", "x", "y")
    ax.set_ylim(0, 3.6)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(-3, 3, 900)
    yr = split_at(xs, xs ** 3 / (xs ** 2 - 1), 1, 0.03)
    yr = split_at(xs, yr, -1, 0.03)
    ax.plot(xs, yr, color=BLUE, linewidth=2.2, label="x³/(x² − 1)")
    ax.plot(xs, xs, "--", color=ORANGE, linewidth=1.8, label="quotient x (slant asymptote)")
    for pole in (-1, 1):
        ax.axvline(pole, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(1.2, -7.8, "x³/(x² − 1)\n= x + x/(x² − 1)", fontsize=10.5)
    style(ax, "Improper? Divide first", "x", "y")
    ax.set_ylim(-9, 9)
    ax.legend(loc="upper left", fontsize=9.5)
    save_integrals(fig, name="06_partial_fractions.png")


if __name__ == "__main__":
    sheet_00_sequence_limits()
    sheet_01_function_limits()
    sheet_02_one_sided_limits()
    sheet_03_limit_laws()
    sheet_00_integral_as_area()
    sheet_01_power_rule()
    sheet_02_exp_trig_integrals()
    sheet_03_u_substitution()
    sheet_04_trig_substitution()
    sheet_05_integration_by_parts()
    sheet_06_partial_fractions()
