#!/usr/bin/env python3
"""Concept sheets for coursera/01_optimization (derivatives and optimization, week 1).

One page per notebook that summarizes its central idea. Every number is computed here from
the same functions, parameters and random seeds the notebook uses, with asserts against the
values the notebook prints. Images are written to coursera/01_optimization/img/.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_coursera_01_optimization_concept_images.py
"""
from functools import partial

import numpy as np
import sympy as sp

from viz_style import (REPO, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "01_optimization"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"


# ================================================================ helpers
def tangent(ax, f, slope, x0, half_width, color, label=None):
    """Short tangent segment of f at x0 with the given slope."""
    xs = np.array([x0 - half_width, x0 + half_width])
    ax.plot(xs, f(x0) + slope * (xs - x0), color=color, linewidth=1.8, linestyle="--", label=label)
    ax.plot(x0, f(x0), "o", color=color, markersize=7, zorder=4)


def sign_shading(ax, xs, ys):
    """Shade where a derivative curve is negative (blue) or positive (orange)."""
    ax.fill_between(xs, ys, 0, where=ys < 0, color=BLUE_LIGHT, alpha=0.8, linewidth=0)
    ax.fill_between(xs, ys, 0, where=ys > 0, color=ORANGE_LIGHT, alpha=0.8, linewidth=0)
    ax.axhline(0, color=MUTED, linewidth=1)


def square_patches(ax, house, lines, colors, ylim):
    """The power-line picture: each connection cost drawn as a square of side |x - p|."""
    from matplotlib.patches import Rectangle
    for p, color in zip(lines, colors):
        d = abs(house - p)
        ax.add_patch(Rectangle((min(house, p), 0), d, d, facecolor=color, alpha=0.35,
                               edgecolor=color, linewidth=1.5))
        ax.plot(p, 0, "o", color=color, markersize=8, clip_on=False, zorder=4)
    ax.axvline(house, color=INK, linewidth=1.3)
    ax.set_ylim(0, ylim)
    ax.set_aspect("equal")
    ax.grid(True)


# ================================================================ notebook 00
def sheet_00_slope_zero():
    def temp(x):
        return 0.5 * (x - 3) ** 2 + 75

    def bumpy(x):
        return np.sin(3 * x) + 0.3 * x ** 2 + 78

    x = sp.symbols("x")
    T = sp.Rational(1, 2) * (x - 3) ** 2 + 75
    Tp = sp.diff(T, x)
    assert sp.solve(Tp, x) == [3]
    walk = [1.0, 0.5, 1.5, 2.5, 3.0]
    assert [round(temp(v)) for v in walk] == [77, 78, 76, 75, 75]

    xs_scan = np.linspace(-2.5, 2.5, 4000)              # the notebook's scan
    dT = np.gradient(bumpy(xs_scan), xs_scan)
    cands = xs_scan[1:][np.sign(dT[:-1]) != np.sign(dT[1:])]
    assert np.round(cands, 3).tolist() == [-2.447, -1.685, -0.491, 0.562, 1.472]
    gmin = cands[np.argmin(bumpy(cands))]
    assert round(gmin, 3) == -0.491 and round(bumpy(gmin), 3) == 77.077

    fig = new_figure("At a minimum the slope is zero, so f′(x) = 0 marks the candidates",
                     "The sign of f′ points the way: f′ < 0 means lower to the right, f′ > 0 lower "
                     "to the left; compare all f′ = 0 points to find the global minimum.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 6, 300)
    ax.plot(xs, temp(xs), color=INK2, linewidth=2.2, label="T(x) = ½(x − 3)² + 75")
    for xv, color in [(1.5, BLUE), (4.5, ORANGE), (3.0, AQUA)]:
        slope = float(Tp.subs(x, xv))
        tangent(ax, temp, slope, xv, 1.1, color)
    for (tx, ty, ha), text, color in [((0.1, 74.0, "left"), "slope −1.5", BLUE),
                                      ((5.9, 74.0, "right"), "slope +1.5", ORANGE),
                                      ((3.0, 75.45, "center"), "slope 0", AQUA)]:
        ax.text(tx, ty, text, ha=ha, fontsize=10.5, color=color, weight="bold")
    ax.set_ylim(73.5, 80.5)
    ax.set_xlabel("position on the bench x")
    ax.set_ylabel("temperature (°C)")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Tangents: the flat one sits at the bottom")

    ax = fig.add_subplot(gs[0, 1])
    d = xs - 3
    sign_shading(ax, xs, d)
    ax.plot(xs, d, color=INK2, linewidth=2.2, label="T′(x) = x − 3")
    ax.plot(3, 0, "o", color=AQUA, markersize=9, zorder=4)
    ax.text(1.4, 0.9, "T′ < 0\ncolder → right", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(4.6, -1.4, "T′ > 0\ncolder ← left", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.annotate("T′(3) = 0: coldest", (3, 0), xytext=(3.3, -2.6), fontsize=10.5, color=AQUA,
                weight="bold", arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("position x")
    ax.set_ylabel("slope T′(x)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The derivative tells you which way to walk")

    temps = [temp(v) for v in walk]
    panel(fig.add_subplot(gs[0, 2]), "The sauna search", [
        f"start   x = 1.0:  {temps[0]:.3f} °C",
        f"left    x = 0.5:  {temps[1]:.3f} °C  hotter",
        f"right   x = 1.5:  {temps[2]:.3f} °C  colder",
        f"        x = 2.5:  {temps[3]:.3f} °C",
        f"coldest x = 3.0:  {temps[4]:.3f} °C",
        "",
        "Stop when every direction is hotter.",
        r"There $T'(x) = x - 3 = 0$, so $x = 3$.",
        "",
        "In ML the 'temperature' is the error of",
        "a model; the coldest spot is the best fit.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    xb = np.linspace(-2.5, 2.5, 400)
    ax.plot(xb, bumpy(xb), color=INK2, linewidth=2.2, label="sin 3x + 0.3x² + 78")
    ax.plot(cands, bumpy(cands), "o", color=ORANGE, markersize=8, zorder=4, label="candidates: f′ = 0")
    ax.plot(gmin, bumpy(gmin), "*", color=AQUA, markersize=20, zorder=5,
            label=f"global min x = {gmin:.3f}")
    ax.set_ylim(76.6, 81.6)
    ax.set_xlabel("position x")
    ax.set_ylabel("temperature")
    ax.legend(loc="upper center", fontsize=9.5, ncol=1)
    ax.grid(True)
    ax.set_title("A bumpy bench: five flat points")

    ax = fig.add_subplot(gs[1, 1])
    ds = np.gradient(bumpy(xb), xb)
    sign_shading(ax, xb, ds)
    ax.plot(xb, ds, color=INK2, linewidth=2.2, label="f′(x) (numerical)")
    ax.plot(cands, np.zeros_like(cands), "o", color=ORANGE, markersize=8, zorder=4,
            label="sign changes")
    ax.set_ylim(-4.6, 6.2)
    ax.set_xlabel("position x")
    ax.set_ylabel("slope f′(x)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Candidates = where f′ changes sign")

    kinds = ["max" if ds_before > 0 else "min"
             for ds_before in [np.gradient(bumpy(xs_scan), xs_scan)[np.searchsorted(xs_scan, c) - 3]
                               for c in cands]]
    lines = [f"x = {c:+.3f}   f = {bumpy(c):.3f}   {k}"
             + ("  ← global" if c == gmin else "") for c, k in zip(cands, kinds)]
    panel(fig.add_subplot(gs[1, 2]), "Local vs global", [
        "Every zero-slope point is a candidate,",
        "a local min or a local max:",
        "",
        *[f"`{line}`" for line in lines],
        "",
        "f′ = 0 cannot tell them apart;",
        "comparing the values f(x) can.",
    ], spacing=0.08)
    save(fig, name="00_slope_zero_candidates.png")


# ================================================================ notebook 01
def sheet_01_one_powerline():
    x = sp.symbols("x")
    f = x ** 2
    fp = sp.diff(f, x)
    assert sp.solve(fp, x) == [0] and f.subs(x, 0) == 0
    positions = [1.0, 4.0, 6.0]                          # the notebook's preview
    total = sum((x - p) ** 2 for p in positions)
    best = float(sp.solve(sp.diff(total, x), x)[0])
    assert abs(best - np.mean(positions)) < 1e-12
    coeffs = sp.Poly(sp.expand(total), x).all_coeffs()
    assert [float(c) for c in coeffs] == [3.0, -22.0, 53.0]

    fig = new_figure("One power line: the cost x² is smallest right at the line",
                     "Connecting at distance x costs f(x) = x²;  f′(x) = 2x = 0 gives x = 0.  "
                     "With several lines you minimize the sum of squared distances.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-5, 5, 300)
    ax.plot(xs, xs ** 2, color=INK2, linewidth=2.2, label="cost f(x) = x²")
    for xv, color in [(-3.0, BLUE), (0.0, AQUA), (2.0, ORANGE)]:
        slope = float(fp.subs(x, xv))
        tangent(ax, lambda v: v ** 2, slope, xv, 1.3, color)
    for (tx, ty, ha), text, color in [((-2.5, 10.5, "left"), "f′(−3) = −6", BLUE),
                                      ((2.4, 1.5, "left"), "f′(2) = 4", ORANGE),
                                      ((0.0, -3.2, "center"), "f′(0) = 0", AQUA)]:
        ax.text(tx, ty, text, ha=ha, fontsize=10.5, color=color, weight="bold")
    ax.set_ylim(-4, 26)
    ax.set_xlabel("house position x (distance from the line)")
    ax.set_ylabel("cost")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The flat tangent is at x = 0")

    ax = fig.add_subplot(gs[0, 1])
    sign_shading(ax, xs, 2 * xs)
    ax.plot(xs, 2 * xs, color=INK2, linewidth=2.2, label="f′(x) = 2x")
    ax.plot(0, 0, "o", color=AQUA, markersize=9, zorder=4)
    ax.text(-2.8, 3.0, "f′ < 0: move right\n(toward the line)", ha="center", fontsize=10.5,
            color=BLUE, weight="bold")
    ax.text(2.8, -5.0, "f′ > 0: move left\n(toward the line)", ha="center", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.set_xlabel("house position x")
    ax.set_ylabel("slope f′(x)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Downhill always points to the line")

    panel(fig.add_subplot(gs[0, 2]), "The cost model", [
        "A house must be wired to a power line.",
        "The farther away, the longer and thicker",
        "the cable: cost = (distance)².",
        "",
        r"$f(x) = x^2,\qquad f'(x) = 2x$",
        r"$f'(x) = 0 \;\Rightarrow\; x = 0,\quad f(0) = 0$",
        "",
        "Build on the line, pay nothing.",
        "This is the squared error used to train",
        "linear regression and neural networks.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xd = np.linspace(0, 4, 200)
    ax.plot(xd, xd, color=BLUE, linewidth=2.2, label="distance |x|")
    ax.plot(xd, xd ** 2, color=ORANGE, linewidth=2.2, label="squared distance x²")
    for v in [1, 2, 3, 4]:
        ax.plot([v, v], [v, v ** 2], color=MUTED, linewidth=1, linestyle=":")
        ax.annotate(f"{v:g} → {v ** 2:g}", (v, v ** 2), textcoords="offset points",
                    xytext=(-6, 6), ha="right", fontsize=10)
    ax.set_ylim(0, 17.5)
    ax.set_xlabel("distance to the line")
    ax.set_ylabel("cost")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Squaring punishes far distances more")

    ax = fig.add_subplot(gs[1, 1])
    xs3 = np.linspace(-1, 9, 300)
    for p, color in zip(positions, [BLUE, ORANGE, AQUA]):
        ax.plot(xs3, (xs3 - p) ** 2, color=color, linewidth=1.5, alpha=0.75,
                label="(x − pᵢ)², one per line" if p == positions[0] else None)
        ax.plot(p, 0, "o", color=color, markersize=7, clip_on=False, zorder=4)
    tot = sum((xs3 - p) ** 2 for p in positions)
    ax.plot(xs3, tot, color=INK, linewidth=2.5, label="sum = total cost")
    ax.plot(best, sum((best - p) ** 2 for p in positions), "*", color=INK, markersize=16, zorder=5)
    ax.text(best, 30, f"min at x = {best:.3f}\n= mean of 1, 4, 6", ha="center", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("house position x")
    ax.set_ylabel("cost")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Preview: three lines at 1, 4, 6")

    panel(fig.add_subplot(gs[1, 2]), "Toward many power lines", [
        "You can't sit on all lines at once, so",
        "minimize the sum of squared distances:",
        r"$f(x) = (x-p_1)^2 + \cdots + (x-p_n)^2$",
        "",
        "Lines at 1, 4, 6:",
        r"$f(x) = 3x^2 - 22x + 53$",
        r"$f'(x) = 6x - 22 = 0 \;\Rightarrow\; x = 11/3$",
        "",
        "The best spot is the average position:",
        "the least-squares idea, solved next for",
        "two and three lines.",
    ], spacing=0.08)
    save(fig, name="01_one_powerline.png")


# ================================================================ notebook 02
def sheet_02_two_powerlines():
    a, b = 2.0, 8.0

    def cost(x):
        return (x - a) ** 2 + (x - b) ** 2

    x, A, B = sp.symbols("x a b")
    f = (x - A) ** 2 + (x - B) ** 2
    sol = sp.solve(sp.diff(f, x), x)[0]
    assert sp.simplify(sol - (A + B) / 2) == 0 and sol.subs({A: 2, B: 8}) == 5
    mid = (a + b) / 2
    assert cost(mid) == 18 and [cost(v) for v in (3.0, 5.0, 7.0)] == [26, 18, 26]

    fig = new_figure("Two power lines: build at the midpoint (a + b) / 2",
                     "f(x) = (x − a)² + (x − b)²,   f′(x) = 2(x − a) + 2(x − b) = 0  ⇒  x = (a + b)/2;  "
                     "with a = 2, b = 8 the house goes at x = 5.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 12, 300)
    ax.plot(xs, (xs - a) ** 2, color=BLUE, linewidth=1.5, alpha=0.7, label="(x − a)²")
    ax.plot(xs, (xs - b) ** 2, color=ORANGE, linewidth=1.5, alpha=0.7, label="(x − b)²")
    ax.plot(xs, cost(xs), color=INK, linewidth=2.5, label="total cost f(x)")
    ax.plot(mid, cost(mid), "*", color=AQUA, markersize=18, zorder=5)
    ax.text(mid, 32, f"min f(5) = {cost(mid):.0f}", ha="center", fontsize=10.5)
    for p, color in [(a, BLUE), (b, ORANGE)]:
        ax.axvline(p, color=color, linestyle=":", linewidth=1.3)
    ax.set_ylim(0, 110)
    ax.set_xlabel("house position x")
    ax.set_ylabel("cost")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Two parabolas add up to one")

    sub = gs[0, 1].subgridspec(1, 2, wspace=0.12)
    square_axes = []
    for k, house in enumerate([3.0, 5.0]):
        ax = fig.add_subplot(sub[k])
        square_axes.append(ax)
        square_patches(ax, house, [a, b], [BLUE, ORANGE], ylim=6.5)
        ax.set_xlim(1, 9)
        ax.set_xticks([2, 5, 8])
        ax.set_yticks([0, 2, 4, 6])
        ax.set_xlabel("x")
        ax.set_title(f"x = {house:g}: area {cost(house):.0f}", fontsize=12)
    left, right = square_axes[0].get_position(), square_axes[1].get_position()
    fig.text(0.5 * (left.x0 + right.x1), right.y1 + 0.055,
             "Cost as two square areas", ha="center", fontsize=13, weight="bold")

    panel(fig.add_subplot(gs[0, 2]), "Setting the slope to zero", [
        r"$f(x) = (x-a)^2 + (x-b)^2$",
        r"$f'(x) = 2(x-a) + 2(x-b) = 0$",
        r"$(x-a) + (x-b) = 0$",
        r"$2x = a + b$",
        r"$x = (a+b)\,/\,2$",
        "",
        r"The $x^2$ coefficient is +2: the parabola",
        "opens upward, so this is the minimum.",
        "a = 2, b = 8:  x = 5,  cost 9 + 9 = 18.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    houses = np.arange(2, 9)
    blue_part = (houses - a) ** 2
    orange_part = (houses - b) ** 2
    ax.bar(houses, blue_part, width=0.62, color=BLUE, label="blue square (x − a)²", zorder=2)
    ax.bar(houses, orange_part, width=0.62, bottom=blue_part, color=ORANGE,
           label="orange square (x − b)²", zorder=2)
    for h, t in zip(houses, blue_part + orange_part):
        ax.text(h, t + 1, f"{t:.0f}", ha="center", fontsize=10,
                weight="bold" if h == mid else None)
    ax.set_ylim(0, 46)
    ax.set_xticks(houses)
    ax.set_xlabel("house position x")
    ax.set_ylabel("total area = cost")
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    ax.grid(axis="y", zorder=0)
    ax.set_title("The middle balances the two squares")

    ax = fig.add_subplot(gs[1, 1])
    xs2 = np.linspace(0, 10, 200)
    ax.plot(xs2, 2 * (xs2 - a), color=BLUE, linewidth=1.6, label="2(x − a)")
    ax.plot(xs2, 2 * (xs2 - b), color=ORANGE, linewidth=1.6, label="2(x − b)")
    sign_shading(ax, xs2, 4 * xs2 - 20)
    ax.plot(xs2, 4 * xs2 - 20, color=INK, linewidth=2.5, label="f′(x) = 4x − 20")
    ax.plot(mid, 0, "o", color=AQUA, markersize=9, zorder=5)
    ax.annotate("pulls cancel at x = 5", (mid, 0), xytext=(5.6, -12), fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(-22, 22)
    ax.set_xlabel("house position x")
    ax.set_ylabel("slope")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The two slopes cancel at the midpoint")

    panel(fig.add_subplot(gs[1, 2]), "Why signs and order don't matter", [
        "Distance to the blue line: x − a;",
        "to the orange line: b − x.",
        r"$(b-x)^2 = (x-b)^2$, so write every",
        "term as (x − line)², whatever the side.",
        "",
        "Too far left: the orange square is huge.",
        "Too far right: the blue square is huge.",
        "",
        "For n lines the same steps give the",
        "average of the positions (next notebook).",
    ], spacing=0.083)
    save(fig, name="02_two_powerlines_midpoint.png")


# ================================================================ notebook 03
def sheet_03_three_powerlines():
    a, b, c = 1.0, 4.0, 9.0

    def cost(x):
        return (x - a) ** 2 + (x - b) ** 2 + (x - c) ** 2

    avg = (a + b + c) / 3
    assert f"{avg:.4f}" == "4.6667" and f"{cost(avg):.4f}" == "32.6667"
    x, A, B, C = sp.symbols("x a b c")
    sol = sp.solve(sp.diff((x - A) ** 2 + (x - B) ** 2 + (x - C) ** 2, x), x)[0]
    assert sol.subs({A: 1, B: 4, C: 9}) == sp.Rational(14, 3)

    rng = np.random.default_rng(0)                       # the notebook's seed and draw
    lines7 = rng.uniform(0, 20, size=7)
    assert np.round(lines7, 2).tolist() == [12.74, 5.4, 0.82, 0.33, 16.27, 18.26, 12.13]
    assert f"{lines7.mean():.4f}" == "9.4197"

    fig = new_figure("Three power lines: build at the average (a + b + c) / 3",
                     "f(x) = (x − a)² + (x − b)² + (x − c)²,  f′(x) = 0  ⇒  x = (a + b + c)/3.  "
                     "For n lines Σ(x − aᵢ)², the square loss, is minimized at the mean.")
    gs = grid(fig, top=0.84, wspace=0.3)

    colors = [BLUE, ORANGE, AQUA]
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-3, 13, 300)
    for p, color, name in zip([a, b, c], colors, "abc"):
        ax.plot(xs, (xs - p) ** 2, color=color, linewidth=1.5, alpha=0.75, label=f"(x − {name})²")
    ax.plot(xs, cost(xs), color=INK, linewidth=2.5, label="sum = total cost")
    ax.plot(avg, cost(avg), "*", color=ORANGE, markersize=18, zorder=5)
    ax.text(avg, 50, f"min at x = {avg:.2f}\ncost {cost(avg):.2f}", ha="center", fontsize=10.5)
    ax.set_ylim(0, 205)
    ax.set_xlabel("house position x  (lines at 1, 4, 9)")
    ax.set_ylabel("cost")
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    ax.grid(True)
    ax.set_title("Stack a parabola over each line")

    ax = fig.add_subplot(gs[0, 1])
    xs2 = np.linspace(-1, 11, 200)
    for p, color in zip([a, b, c], colors):
        ax.plot(xs2, 2 * (xs2 - p), color=color, linewidth=1.4, alpha=0.8)
    sign_shading(ax, xs2, 6 * xs2 - 28)
    ax.plot(xs2, 6 * xs2 - 28, color=INK, linewidth=2.5, label="f′(x) = 6x − 28")
    ax.plot(avg, 0, "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate("f′ = 0 at x = 14/3", (avg, 0), xytext=(6.2, -18), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.text(10.6, 2 * (10.6 - a) - 4.5, "2(x − a)", color=BLUE, fontsize=9.5, ha="right")
    ax.text(10.6, 2 * (10.6 - c) + 2.5, "2(x − c)", color=AQUA, fontsize=9.5, ha="right")
    ax.set_ylim(-35, 40)
    ax.set_xlabel("house position x")
    ax.set_ylabel("slope")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Three slopes sum to one line")

    panel(fig.add_subplot(gs[0, 2]), "Setting the slope to zero", [
        r"$f'(x) = 2(x-a) + 2(x-b) + 2(x-c) = 0$",
        r"$(x-a) + (x-b) + (x-c) = 0$",
        r"$3x - a - b - c = 0$",
        r"$x = (a+b+c)\,/\,3$",
        "",
        "Lines at 1, 4, 9:",
        r"$f'(x) = 6x - 28 \;\Rightarrow\; x = 14/3 \approx 4.667$",
        f"minimum cost f(14/3) = {cost(avg):.4f}",
    ], spacing=0.092)

    ax = fig.add_subplot(gs[1, 0])
    houses = [2.0, avg, 9.0]                              # the notebook's three snapshots
    bottoms = np.zeros(3)
    for p, color, name in zip([a, b, c], colors, "abc"):
        part = np.array([(h - p) ** 2 for h in houses])
        ax.bar(range(3), part, width=0.6, bottom=bottoms, color=color, label=f"(x − {name})²",
               zorder=2)
        bottoms += part
    for k, t in enumerate(bottoms):
        ax.text(k, t + 2, f"{t:.1f}", ha="center", fontsize=10.5, weight="bold" if k == 1 else None)
    ax.set_xticks(range(3), ["x = 2", "x = 14/3\n(average)", "x = 9"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("total square area = cost")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("The far line owns the biggest square")

    ax = fig.add_subplot(gs[1, 1])
    xs7 = np.linspace(-2, 22, 400)
    tot7 = sum((xs7 - p) ** 2 for p in lines7)
    ax.plot(xs7, tot7, color=INK, linewidth=2.5, label="Σ (x − aᵢ)²,  7 random lines")
    ax.plot(lines7, np.zeros_like(lines7), "|", color=BLUE, markersize=22, markeredgewidth=2.5,
            clip_on=False, zorder=4, label="line positions aᵢ")
    m = lines7.mean()
    ax.plot(m, sum((m - p) ** 2 for p in lines7), "*", color=ORANGE, markersize=18, zorder=5)
    ax.annotate(f"min at mean = {m:.4f}", (m, sum((m - p) ** 2 for p in lines7)),
                xytext=(11.5, 1750), fontsize=10.5, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(0, 2600)
    ax.set_xlabel("house position x")
    ax.set_ylabel("cost")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Any n lines: the minimum is the mean")

    panel(fig.add_subplot(gs[1, 2]), "The square loss", [
        "f(x) = Σᵢ (x − aᵢ)²,   i = 1 … n",
        "f′(x) = 2 Σᵢ (x − aᵢ) = 0",
        "⇒  x = (a₁ + … + aₙ) / n = ā",
        "",
        "Seven lines drawn with seed 0:",
        "solving f′ = 0 and averaging both give",
        f"x = {m:.4f}.",
        "",
        "Sum of squared errors: the loss behind",
        "least squares and linear regression.",
    ], spacing=0.083)
    save(fig, name="03_three_powerlines_average.png")


# ================================================================ notebook 04 part 1
def sheet_04_log_loss_part_1():
    def win_prob(p):
        return p ** 7 * (1 - p) ** 3

    coins = [0.7, 0.5, 0.3]
    wins = [win_prob(p) for p in coins]
    assert [f"{w:.6f}" for w in wins] == ["0.002224", "0.000977", "0.000075"]
    P = sp.symbols("P", positive=True)
    G = P ** 7 * (1 - P) ** 3
    assert sp.factor(sp.diff(G, P)) == -P ** 6 * (P - 1) ** 2 * (10 * P - 7)
    assert sp.solve(sp.diff(7 * sp.log(P) + 3 * sp.log(1 - P), P), P) == [sp.Rational(7, 10)]

    fig = new_figure("Best coin: maximize the likelihood, or minimize the log loss",
                     "Win only on 7 heads then 3 tails:  G(P) = P⁷(1 − P)³.  log G = 7 log P + 3 log(1 − P) "
                     "has the same peak, P = 0.7, and is far easier to differentiate.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(3), np.array(wins) * 1000, width=0.6, color=[ORANGE, BLUE, BLUE], zorder=2)
    for k, w in enumerate(wins):
        ax.text(k, w * 1000 + 0.05, f"{w:.6f}", ha="center", fontsize=10.5)
    ax.set_xticks(range(3), ["coin 1\nP(H) = 0.7", "coin 2\nP(H) = 0.5", "coin 3\nP(H) = 0.3"])
    ax.set_ylim(0, 2.6)
    ax.set_ylabel("P(H)⁷ · P(T)³   (× 10⁻³)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Three coins: coin 1 wins")

    ps = np.linspace(0.001, 0.999, 400)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ps, win_prob(ps), color=BLUE, linewidth=2.4, label="G(P) = P⁷(1 − P)³")
    tangent(ax, win_prob, 0.0, 0.7, 0.13, AQUA)
    for p, w in zip(coins, wins):
        ax.plot(p, w, "o", color=ORANGE if p == 0.7 else INK2, markersize=7, zorder=5)
    ax.annotate(f"G′(0.7) = 0\nG(0.7) = {win_prob(0.7):.6f}", (0.7, win_prob(0.7)),
                xytext=(0.12, 0.0019), fontsize=10.5, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(0, 0.0026)
    ax.set_xlabel("P = P(heads)")
    ax.set_ylabel("win probability")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Over all coins: the peak is P = 0.7")

    panel(fig.add_subplot(gs[0, 2]), "The direct way (product rule)", [
        r"$G'(P) = 7P^6(1-P)^3 - 3P^7(1-P)^2$",
        r"$\quad\;\; = -P^6\,(P-1)^2\,(10P-7)$",
        "",
        "Zeros: P = 0, P = 1, P = 0.7.",
        "P = 0 (never heads) and P = 1 (never",
        "tails) can't win, so P = 0.7.",
        "",
        "Correct, but it takes the product rule,",
        "the chain rule and a factorization.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    loss = -(7 * np.log(ps) + 3 * np.log(1 - ps))
    ax.plot(ps, loss, color=ORANGE, linewidth=2.4, label="log loss = −log G(P)")
    ax.axvline(0.7, color=INK2, linestyle="--", linewidth=1.4,
               label=f"min at P = 0.7:  {-np.log(win_prob(0.7)):.3f}")
    ax.plot(0.7, -np.log(win_prob(0.7)), "o", color=AQUA, markersize=9, zorder=5)
    ax.set_ylim(0, 30)
    ax.set_xlabel("P = P(heads)")
    ax.set_ylabel("log loss")
    ax.legend(loc="upper left", bbox_to_anchor=(0.07, 1.0), fontsize=9.5)
    ax.grid(True)
    ax.set_title("Flip the sign: minimize the log loss")

    ax = fig.add_subplot(gs[1, 1])
    d = 7 / ps - 3 / (1 - ps)
    sign_shading(ax, ps, d)
    ax.plot(ps, d, color=INK, linewidth=2.4, label="7/P − 3/(1 − P)")
    ax.plot(0.7, 0, "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate("zero at P = 0.7", (0.7, 0), xytext=(0.35, -40), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(-80, 80)
    ax.set_xlabel("P = P(heads)")
    ax.set_ylabel("d/dP log G")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The easy derivative of log G")

    panel(fig.add_subplot(gs[1, 2]), "The log trick and the log loss", [
        r"$(\log G)' = 7/P \,-\, 3/(1-P) = 0$",
        r"$7(1-P) = 3P \;\Rightarrow\; P = 0.7$",
        "",
        r"log loss $= -\log G = -[\,7\log P + 3\log(1-P)\,]$",
        "is positive (log P < 0) and is minimized",
        "at the same P = 0.7.",
        "",
        "h heads, t tails:  P = h / (h + t),",
        "the observed frequency.",
    ], spacing=0.088)
    save(fig, name="04_log_loss_best_coin.png")


# ================================================================ notebook 04 part 2
def sheet_04_log_loss_part_2():
    P = sp.symbols("P", positive=True)
    G = P ** 7 * (1 - P) ** 3
    logG = 7 * sp.log(P) + 3 * sp.log(1 - P)
    expr = P ** 3 * (1 - P) ** 2 * (P + sp.Rational(1, 2)) ** 4 * (2 - P) ** 3
    len_direct = len(str(sp.expand(sp.diff(expr, P))))
    len_log = len(str(sp.simplify(sp.diff(sp.log(expr), P))))
    assert (len_direct, len_log) == (110, 59)             # the notebook's printout
    assert sp.solve(sp.diff(logG, P), P) == [sp.Rational(7, 10)]

    rng = np.random.default_rng(0)                         # the notebook's seed and draw
    probs = rng.uniform(0.01, 0.5, size=2000)
    assert np.prod(probs) == 0.0
    log_sum = np.sum(np.log(probs))
    assert f"{log_sum:.2f}" == "-3249.10"
    running = np.cumprod(probs)
    first_zero = int(np.argmax(running == 0))
    running_log10 = np.cumsum(np.log(probs)) / np.log(10)

    fig = new_figure("Why the logarithm? Products become sums",
                     "log(∏ pᵢ) = Σ log pᵢ:  sums are easy to differentiate, and a sum of logs "
                     "never underflows the way a product of tiny probabilities does.")
    gs = grid(fig, top=0.84, wspace=0.42)

    panel(fig.add_subplot(gs[0, 0]), "The coin example was ML", [
        "data:  10 flips, 7 heads then 3 tails",
        "model:  a coin with P(heads) = P",
        "fit:  the model most likely to give the",
        "data, by minimizing the log loss",
        "",
        r"$-\log G(P) = -[\,7\log P + 3\log(1-P)\,]$",
        "",
        "Result: P = 0.7.  Two reasons for the",
        "log: easier calculus, safer numbers.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[0, 1])
    ps = np.linspace(0.02, 0.98, 400)
    g = ps ** 7 * (1 - ps) ** 3
    ax.plot(ps, g / g.max(), color=BLUE, linewidth=2.4, label="G(P), scaled to peak 1")
    ax.set_ylabel("G(P) / max G", color=BLUE)
    ax.set_xlabel("P = P(heads)")
    ax.set_ylim(0, 1.4)
    ax2 = ax.twinx()
    ax2.plot(ps, 7 * np.log(ps) + 3 * np.log(1 - ps), color=AQUA, linewidth=2.4, label="log G(P)")
    ax2.set_ylabel("log G(P)", color=AQUA)
    ax2.set_ylim(-25, 5)
    ax2.spines["right"].set_visible(True)
    ax.axvline(0.7, color=ORANGE, linestyle="--", linewidth=1.4)
    ax.text(0.72, 1.3, "both peak\nat P = 0.7", ha="left", va="center", fontsize=10.5, color=ORANGE,
            weight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("log is increasing: same optimum")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0, 1], [len_direct, len_log], width=0.55, color=[ORANGE, AQUA], zorder=2)
    for k, v in enumerate([len_direct, len_log]):
        ax.text(k, v + 2, f"{v} characters", ha="center", fontsize=10.5)
    ax.set_xticks([0, 1], ["d/dP of the product\n(expanded)", "d/dP of its log\n(simplified)"])
    ax.set_ylim(0, 130)
    ax.set_ylabel("length of the expression")
    ax.grid(axis="y", zorder=0)
    ax.set_title("P³(1−P)²(P+½)⁴(2−P)³: derivative size")

    ax = fig.add_subplot(gs[1, 0])
    pg = np.linspace(0.05, 0.95, 400)
    t1, t2 = 7 / pg, -3 / (1 - pg)
    ax.plot(pg, t1, color=BLUE, linewidth=1.8, label="7/P   from 7 log P")
    ax.plot(pg, t2, color=ORANGE, linewidth=1.8, label="−3/(1 − P)   from 3 log(1 − P)")
    ax.plot(pg, t1 + t2, color=INK, linewidth=2.5, label="sum = d/dP log G")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(0.7, 0, "o", color=AQUA, markersize=9, zorder=5)
    ax.annotate("zero at P = 0.7", (0.7, 0), xytext=(0.72, 30), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_ylim(-60, 80)
    ax.set_xlabel("P = P(heads)")
    ax.set_ylabel("slope")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("One simple term per factor")

    ax = fig.add_subplot(gs[1, 1])
    n = np.arange(1, len(probs) + 1)
    with np.errstate(divide="ignore"):
        prod_log10 = np.where(running > 0, np.log10(np.where(running > 0, running, 1.0)), np.nan)
    ax.plot(n, running_log10, color=AQUA, linewidth=2.4, label="Σ log₁₀ pᵢ  (sum of logs)")
    ax.plot(n, prod_log10, color=ORANGE, linewidth=2.4, label="log₁₀ of the float product")
    ax.axhline(np.log10(np.nextafter(0, 1)), color=INK2, linestyle=":", linewidth=1.3)
    ax.text(1990, np.log10(np.nextafter(0, 1)) + 25, "smallest float64 ≈ 10⁻³²⁴",
            ha="right", fontsize=9.5, color=INK2)
    ax.plot(first_zero + 1, running_log10[first_zero], "x", color=ORANGE, markersize=12,
            markeredgewidth=3, zorder=5)
    ax.annotate(f"product = 0.0 from n = {first_zero + 1}", (first_zero + 1, running_log10[first_zero]),
                xytext=(700, -140), fontsize=10.5, color=ORANGE, weight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_xlim(0, 2000)
    ax.set_ylim(-1500, 50)
    ax.set_xlabel("number of probabilities multiplied, n")
    ax.set_ylabel("log₁₀ (product)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("2000 probabilities in [0.01, 0.5]: underflow")

    panel(fig.add_subplot(gs[1, 2]), "The practical rule", [
        "A complicated product?  Take the log,",
        "especially before differentiating.",
        "",
        "1. Calculus: one product-rule mess",
        "   becomes a sum of simple fractions.",
        "2. Numerics, the 2000 probabilities:",
        "   np.prod → 0.0 (underflow)",
        f"   Σ log pᵢ = {log_sum:.2f} (natural log)",
        "",
        "That is why ML losses are written as",
        "(negative) log-likelihoods.",
    ], spacing=0.08)
    save(fig, name="04_log_loss_why_logarithm.png")


if __name__ == "__main__":
    sheet_00_slope_zero()
    sheet_01_one_powerline()
    sheet_02_two_powerlines()
    sheet_03_three_powerlines()
    sheet_04_log_loss_part_1()
    sheet_04_log_loss_part_2()
