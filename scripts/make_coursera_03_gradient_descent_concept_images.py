#!/usr/bin/env python3
"""Concept sheets for coursera/03_gradient_descent (gradient descent in one and two variables).

One page per notebook that summarizes its central idea. Every number is computed here from
the same functions, starting points, learning rates, data and random seeds the notebook
uses, with asserts against the values the notebook prints.
Images are written to coursera/03_gradient_descent/img/.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_coursera_03_gradient_descent_concept_images.py
"""
from functools import partial

import numpy as np
import pandas as pd

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save
from matplotlib.colors import LinearSegmentedColormap

LESSON = REPO / "coursera" / "03_gradient_descent"
save = partial(_save, folder=LESSON / "img")

LIGHT_SEQ = LinearSegmentedColormap.from_list("light_seq", [SEQ(x) for x in np.linspace(0, 0.5, 6)])
OMEGA = 0.5671432904          # the Omega constant W(1), the minimum of e^x - ln x


# ================================================================ helpers
def f_exp_log(x):
    return np.exp(x) - np.log(x)


def fprime_exp_log(x):
    return np.exp(x) - 1 / x


def gd_1d(fprime, x0, alpha, n_steps=2000, tol=1e-8):
    """Notebook 01's gradient_descent: stop when |alpha f'(x)| < tol."""
    x, history = x0, [x0]
    for _ in range(n_steps):
        step = alpha * fprime(x)
        x = x - step
        history.append(x)
        if abs(step) < tol:
            break
    return np.array(history)


def gd_fixed(fprime, x0, alpha, n_steps):
    """Notebook 03's gradient_descent: a fixed number of steps, no stopping rule."""
    x, history = x0, [x0]
    for _ in range(n_steps):
        x = x - alpha * fprime(x)
        history.append(x)
    return np.array(history)


def gd_2d(grad, start, alpha, n_steps, tol):
    """The 2D loop shared by notebooks 04-07: stop when the step norm drops below tol."""
    pos = np.array(start, dtype=float)
    path = [pos.copy()]
    for _ in range(n_steps):
        step = alpha * grad(*pos)
        pos = pos - step
        path.append(pos.copy())
        if np.linalg.norm(step) < tol:
            break
    return np.array(path)


def T_sauna(x, y):
    return 85 - (1 / 90) * x ** 2 * (x - 6) * y ** 2 * (y - 6)


def grad_T_sauna(x, y):
    tx = -(1 / 90) * x * (3 * x - 12) * y ** 2 * (y - 6)
    ty = -(1 / 90) * x ** 2 * (x - 6) * y * (3 * y - 12)
    return np.array([tx, ty])


def f_quartic(x):
    return 3 * x ** 4 + 4 * x ** 3 - 12 * x ** 2


def fprime_quartic(x):
    return 12 * x ** 3 + 12 * x ** 2 - 24 * x


def light_contours(ax, X, Y, Z, levels, filled=True, labels=False):
    if filled:
        ax.contourf(X, Y, Z, levels=levels, cmap=LIGHT_SEQ, extend="both")
    cs = ax.contour(X, Y, Z, levels=levels, colors=MUTED, linewidths=0.6)
    if labels:
        ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return cs


def path_line(ax, path, color=ORANGE, label=None, markersize=3.5, lw=1.6, **kw):
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, markersize=markersize, linewidth=lw,
            label=label, zorder=4, **kw)


# ================================================================ notebook 00
def sheet_00_naive_search():
    def naive_search(x0, h, max_steps=100):
        x, history = x0, [x0]
        for _ in range(max_steps):
            best_val, best_x = min([(f_exp_log(x - h), x - h), (f_exp_log(x + h), x + h)])
            if best_val < f_exp_log(x):
                x = best_x
                history.append(x)
            else:
                break
        return x, np.array(history)

    x_star, hist = naive_search(1.5, 0.1)
    assert f"{x_star:.4f}" == "0.6000" and len(hist) - 1 == 9
    assert f"{f_exp_log(x_star):.5f}" == "2.33294"
    x_small, hist_small = naive_search(1.5, 0.01, max_steps=1000)
    assert f"{x_small:.4f}" == "0.5700" and len(hist_small) - 1 == 93
    x_low, _ = naive_search(0.2, 0.1)
    assert f"{x_low:.4f}" == "0.6000"

    fig = new_figure("Gradient descent, part 1: when f′(x) = 0 has no formula, search",
                     "f(x) = eˣ − ln x has one minimum, at the Omega constant Ω ≈ 0.5671, "
                     "but eˣ = 1/x has no elementary solution. Step toward lower values instead.")
    gs = grid(fig, top=0.84, wspace=0.3)

    xs = np.linspace(0.05, 2.0, 400)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xs, f_exp_log(xs), color=BLUE, linewidth=2.2, label="f(x) = eˣ − ln x")
    ax.plot(hist, f_exp_log(hist), "o-", color=ORANGE, markersize=6, linewidth=1.5,
            label="naive search from x₀ = 1.5, h = 0.1", zorder=3)
    ax.vlines(OMEGA, 2, 5.2, color=INK2, linestyle="--", linewidth=1.2, label="true minimum Ω")
    ax.annotate("start", (1.5, f_exp_log(1.5)), textcoords="offset points", xytext=(8, -4),
                fontsize=10, color=ORANGE)
    ax.annotate(f"stops at x = {x_star:.1f} after 9 steps", (x_star, f_exp_log(x_star)),
                xytext=(1.0, 2.15), fontsize=10, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, shrinkB=4))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(2, 6.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper center", fontsize=9, frameon=True, framealpha=1, facecolor=SURFACE,
              edgecolor="none")
    ax.grid(True)
    ax.set_title("Walk downhill in fixed steps of h")

    ax = fig.add_subplot(gs[0, 1])
    xs2 = np.linspace(0.15, 1.5, 400)
    ax.plot(xs2, np.exp(xs2), color=BLUE, linewidth=2.2, label="eˣ")
    ax.plot(xs2, 1 / xs2, color=AQUA, linewidth=2.2, label="1/x")
    ax.plot(OMEGA, np.exp(OMEGA), "o", color=ORANGE, markersize=9, zorder=3)
    ax.annotate("Ω = W(1) ≈ 0.567143\nx·eˣ = 1, no elementary form", (OMEGA, np.exp(OMEGA)),
                xytext=(0.62, 4.15), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_ylim(0, 5)
    ax.set_xlabel("x")
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True)
    ax.set_title("f′(x) = 0  means  eˣ = 1/x")

    panel(fig.add_subplot(gs[0, 2]), "Why not just solve it?", [
        r"$f(x)=e^{x}-\ln x,\quad f'(x)=e^{x}-\dfrac{1}{x}$",
        "",
        r"$f'(x)=0 \;\Leftrightarrow\; x\,e^{x}=1$",
        "",
        "The answer is the Lambert W value",
        "Ω = W(1) ≈ 0.567143: no closed form",
        "built from +, ×, powers, exp or log.",
        "",
        "In many variables, ∇f = 0 is usually",
        "even harder. So we iterate instead.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xz = np.linspace(0.42, 0.78, 300)
    ax.plot(xz, f_exp_log(xz), color=BLUE, linewidth=2.2)
    for xp, label, color, off, ha in [(0.5, "x − h = 0.5", AQUA, (-10, -6), "right"),
                                      (0.6, "x = 0.6", ORANGE, (10, -6), "left"),
                                      (0.7, "x + h = 0.7", AQUA, (10, -6), "left")]:
        ax.plot(xp, f_exp_log(xp), "o", color=color, markersize=9, zorder=3)
        ax.annotate(f"{label}\nf = {f_exp_log(xp):.4f}", (xp, f_exp_log(xp)),
                    textcoords="offset points", xytext=off, ha=ha, va="top", fontsize=9.5,
                    color=INK2)
    ax.axvline(OMEGA, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(OMEGA - 0.006, 2.395, "Ω", ha="right", fontsize=11, color=INK2)
    ax.set_ylim(2.31, 2.41)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.set_title("Stop when both neighbours are higher")

    ax = fig.add_subplot(gs[1, 1])
    hs = [0.2, 0.1, 0.05, 0.02, 0.01]
    runs = [naive_search(1.5, h, max_steps=1000) for h in hs]
    errors = [abs(x - OMEGA) for x, _ in runs]
    steps = [len(p) - 1 for _, p in runs]
    pos = np.arange(len(hs))
    ax.bar(pos, errors, width=0.6, color=[ORANGE if h in (0.1, 0.01) else BLUE for h in hs],
           zorder=2)
    for p, e, s in zip(pos, errors, steps):
        ax.text(p, e * 1.15, f"{e:.4f}\n{s} steps", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.5)
    ax.set_xticks(pos, [f"h = {h:g}" for h in hs])
    ax.set_xlabel("fixed step size (start x₀ = 1.5)")
    ax.set_ylabel("final error |x − Ω|")
    ax.grid(axis="y")
    ax.set_title("Smaller h: closer, but many more steps")

    panel(fig.add_subplot(gs[1, 2]), "The naive search, and its flaw", [
        "1. start at some x",
        "2. look at x − h and x + h",
        "3. move to the lower one if it beats f(x)",
        "4. stop when both neighbours are higher",
        "",
        "h = 0.1 stops at 0.6 (9 steps), also from",
        "x₀ = 0.2; h = 0.01 reaches 0.57 (93 steps).",
        "The step ignores how steep f is. Next:",
        r"$x_{\mathrm{new}} = x_{\mathrm{old}} - \alpha\, f'(x_{\mathrm{old}})$",
    ], spacing=0.085)
    save(fig, name="00_naive_search.png")


# ================================================================ notebook 01
def sheet_01_update_rule():
    x0, alpha = 0.05, 0.005
    hist = gd_1d(fprime_exp_log, x0, alpha)
    assert len(hist) - 1 == 542 and f"{hist[-1]:.6f}" == "0.567143"
    assert np.round(hist[:5], 4).tolist() == [0.05, 0.1447, 0.1735, 0.1964, 0.2158]
    assert f"{fprime_exp_log(0.05):.4f}" == "-18.9487"
    alphas = [0.001, 0.005, 0.02, 0.06]
    runs = {a: gd_1d(fprime_exp_log, x0, a, n_steps=20000) for a in alphas}
    assert len(runs[0.001]) - 1 == 2410 and len(runs[0.02]) - 1 == 139

    fig = new_figure("Gradient descent: step against the slope, scaled by α",
                     "xₖ = xₖ₋₁ − α·f′(xₖ₋₁).  On f(x) = eˣ − ln x from x₀ = 0.05 with "
                     "α = 0.005 it reaches the Omega constant 0.567143 in 542 steps.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0.03, 1.5, 400)
    ax.plot(xs, f_exp_log(xs), color=BLUE, linewidth=2.2, label="f(x) = eˣ − ln x")
    for xt, color in [(0.1447, AQUA), (1.3, ORANGE)]:
        slope = fprime_exp_log(xt)
        seg = np.linspace(xt - 0.17, xt + 0.17, 2)
        ax.plot(seg, f_exp_log(xt) + slope * (seg - xt), color=color, linewidth=2)
        ax.plot(xt, f_exp_log(xt), "o", color=color, markersize=7, zorder=3)
    ax.annotate("f′ < 0: move right", (0.1447, f_exp_log(0.1447)), textcoords="offset points",
                xytext=(22, 6), fontsize=10.5, color=AQUA, weight="bold")
    ax.annotate("f′ > 0: move left", (1.3, f_exp_log(1.3)), textcoords="offset points",
                xytext=(-130, 22), fontsize=10.5, color=ORANGE, weight="bold")
    ax.axvline(OMEGA, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(OMEGA + 0.02, 4.2, "Ω", fontsize=11, color=INK2)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(2, 4.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.set_title("−f′ always points toward the minimum")

    ax = fig.add_subplot(gs[0, 1])
    steps = np.abs(np.diff(hist))
    ax.plot(hist[:-1], steps, color=BLUE, linewidth=2)
    for k, off, ha in [(0, (10, 2), "left"), (1, (6, -16), "left")]:
        ax.plot(hist[k], steps[k], "o", color=ORANGE, markersize=7, zorder=3)
        ax.annotate(f"step {k + 1}: {steps[k]:.4f}", (hist[k], steps[k]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=9.5)
    ax.annotate(f"last step: {steps[-1]:.0e}", (hist[-2], steps[-1]), textcoords="offset points",
                xytext=(-12, 0), ha="right", va="center", fontsize=9.5)
    ax.axvline(OMEGA, color=INK2, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 0.5)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("current position xₖ₋₁")
    ax.set_ylabel("step length α·|f′(xₖ₋₁)|")
    ax.grid(True)
    ax.set_title("Steep: big step. Flat: small step (golf)")

    panel(fig.add_subplot(gs[0, 2]), "The update, by hand", [
        r"$x_k = x_{k-1} - \alpha\, f'(x_{k-1}),\quad f'(x)=e^{x}-\dfrac{1}{x}$",
        "",
        "x₀ = 0.05,  α = 0.005",
        "f′(0.05) = 1.0513 − 20 = −18.949",
        "x₁ = 0.05 − 0.005·(−18.949) = 0.1447",
        "f′(0.1447) = 1.1557 − 6.911 = −5.753",
        "x₂ = 0.1447 − 0.005·(−5.753) = 0.1735",
        "",
        "Never solves f′ = 0: it only evaluates f′.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    colors = {0.001: BLUE_LIGHT, 0.005: BLUE, 0.02: AQUA, 0.06: ORANGE}
    for a in alphas:
        h = runs[a]
        ax.plot(np.arange(len(h)), np.abs(h - OMEGA), color=colors[a], linewidth=2,
                label=f"α = {a}: {len(h) - 1} steps")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-9, 2)
    ax.set_xlabel("iteration k (log scale)")
    ax.set_ylabel("distance to minimum |xₖ − Ω|")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Learning rate: too small crawls")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xs, f_exp_log(xs), color=INK2, linewidth=1.8)
    slow, fast = runs[0.005][:7], runs[0.06][:7]
    ax.plot(slow, f_exp_log(slow), "o-", color=BLUE, markersize=6, linewidth=1.5,
            label="α = 0.005: first 6 steps")
    ax.plot(fast, f_exp_log(fast), "o-", color=ORANGE, markersize=6, linewidth=1.5,
            label=f"α = 0.06: x₁ = {fast[1]:.3f}, past Ω")
    ax.axvline(OMEGA, color=INK2, linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(2, 4.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Too large overshoots the minimum")

    panel(fig.add_subplot(gs[1, 2]), "The algorithm", [
        "1. choose a learning rate α and a start x₀",
        r"2. repeat  $x_k = x_{k-1} - \alpha\, f'(x_{k-1})$",
        "3. stop when the step is tiny (< 10⁻⁸ here)",
        "",
        "Iterations to converge from x₀ = 0.05:",
        f"α = 0.001: {len(runs[0.001]) - 1},  α = 0.005: {len(runs[0.005]) - 1},",
        f"α = 0.02: {len(runs[0.02]) - 1},  α = 0.06: {len(runs[0.06]) - 1}",
        "",
        "Same answer, Ω = 0.567143, every time.",
    ], spacing=0.085)
    save(fig, name="01_update_rule.png")


# ================================================================ notebook 03
def sheet_03_pitfalls():
    bowl = lambda x: 2 * x                                   # f(x) = x², as in the notebook
    paths = {0.01: gd_fixed(bowl, 5.0, 0.01, 50), 0.3: gd_fixed(bowl, 5.0, 0.3, 50),
             1.01: gd_fixed(bowl, 5.0, 1.01, 50)}
    assert [f"{paths[a][-1]:.4f}" for a in (0.01, 0.3, 1.01)] == ["1.8208", "0.0000", "13.4579"]
    path_global = gd_fixed(fprime_quartic, -1.0, 0.01, 200)
    path_local = gd_fixed(fprime_quartic, 0.5, 0.01, 200)
    assert f"{path_global[-1]:.4f}" == "-2.0000" and f"{path_local[-1]:.4f}" == "1.0000"

    rng = np.random.default_rng(0)                           # the notebook's seed and draw
    starts = rng.uniform(-3, 2, size=12)
    finals = np.array([gd_fixed(fprime_quartic, s, 0.01, 300)[-1] for s in starts])
    assert np.isfinite(finals).all()
    n_global_12 = int((finals < -1).sum())
    assert n_global_12 == 5 and f"{starts[1]:.2f}" == "-1.65"
    rng = np.random.default_rng(1)                           # exercise 5
    finals20 = [gd_fixed(fprime_quartic, s, 0.01, 300)[-1] for s in rng.uniform(-3, 2, 20)]
    n_global_20 = sum(1 for x in finals20 if np.isfinite(x) and x < -1)
    assert n_global_20 == 15

    fig = new_figure("Gradient descent, the two pitfalls: the learning rate and local minima",
                     "Too large an α diverges, too small crawls; and downhill-only steps stop "
                     "in the nearest valley. Random restarts help, with no guarantee.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    styles = {0.01: (BLUE, "too small  α = 0.01"), 0.3: (AQUA, "just right  α = 0.3"),
              1.01: (ORANGE, "too large  α = 1.01")}
    for a, (color, label) in styles.items():
        ax.plot(np.abs(paths[a]), "o-", color=color, markersize=3, linewidth=1.5,
                label=f"{label}: x₅₀ = {paths[a][-1]:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-21, 1e3)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("distance to the minimum |xₖ|")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)
    ax.set_title("f(x) = x² from x₀ = 5: three regimes")

    ax = fig.add_subplot(gs[0, 1])
    a_grid = np.linspace(0, 1.25, 400)
    factor = np.abs(1 - 2 * a_grid)
    ax.plot(a_grid, factor, color=BLUE, linewidth=2.2)
    ax.axhline(1, color=INK2, linestyle=":", linewidth=1.2)
    ax.axvspan(0, 1, color=AQUA, alpha=0.1, linewidth=0)
    ax.axvspan(1, 1.25, color=ORANGE, alpha=0.12, linewidth=0)
    for a, (color, _) in styles.items():
        ax.plot(a, abs(1 - 2 * a), "o", color=color, markersize=9, zorder=3)
        ax.annotate(f"α = {a}: {abs(1 - 2 * a):.2f}", (a, abs(1 - 2 * a)), textcoords="offset points",
                    xytext=(8, 8) if a != 1.01 else (-12, 10), ha="left" if a != 1.01 else "right",
                    fontsize=10)
    ax.text(0.5, 1.33, "converges: 0 < α < 1", ha="center", fontsize=10, color=AQUA, weight="bold")
    ax.text(1.125, 1.33, "diverges", ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.set_ylim(0, 1.45)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("learning rate α")
    ax.set_ylabel("|1 − 2α|, shrink factor per step")
    ax.grid(True)
    ax.set_title("xₖ = (1 − 2α)·xₖ₋₁ on the bowl")

    panel(fig.add_subplot(gs[0, 2]), "Choosing α", [
        "Too large: overshoots, oscillates, can",
        "diverge (α = 1.01: |x₅₀| = 13.46).",
        "Too small: crawls (α = 0.01: x₅₀ = 1.82).",
        "Just right: fast and stable (α = 0.3).",
        "",
        "For x² the step multiplies x by (1 − 2α),",
        "so the rule is exact: |1 − 2α| < 1.",
        "In general it is a research problem;",
        "adaptive methods change α as they go.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-3, 2, 400)
    ax.plot(xs, f_quartic(xs), color=INK2, linewidth=1.8, label="f(x) = 3x⁴ + 4x³ − 12x²")
    ax.plot(path_global, f_quartic(path_global), "o-", color=AQUA, markersize=4,
            label="start −1.0 → global min x = −2")
    ax.plot(path_local, f_quartic(path_local), "o-", color=ORANGE, markersize=4,
            label="start 0.5 → local min x = 1")
    ax.annotate("f(−2) = −32", (-2, -32), textcoords="offset points", xytext=(18, -4),
                fontsize=10, color=AQUA)
    ax.annotate("f(1) = −5", (1, -5), textcoords="offset points", xytext=(-6, -22),
                ha="center", fontsize=10, color=ORANGE)
    ax.set_ylim(-40, 45)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True)
    ax.set_title("Same algorithm, different start (α = 0.01)")

    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(-3.1, 0, color=AQUA, alpha=0.12, linewidth=0)
    ax.axvspan(0, 2, color=ORANGE, alpha=0.12, linewidth=0)
    ax.text(-1.5, 1.6, "basin of x = −2", ha="center", fontsize=10.5, color=AQUA, weight="bold")
    ax.text(1.0, 1.6, "basin of x = 1", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.axvline(0, color=INK2, linestyle=":", linewidth=1.2)
    ax.text(0.05, -2.35, "local max x = 0", fontsize=9.5, color=INK2)
    for s, xf in zip(starts, finals):
        color = AQUA if xf < -1 else ORANGE
        ax.annotate("", xy=(s, xf), xytext=(s, 0), arrowprops=dict(arrowstyle="->", color=color,
                                                                     linewidth=1.3))
        ax.plot(s, 0, "o", color=INK, markersize=5, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-3.1, 2)
    ax.set_ylim(-2.6, 2)
    ax.set_xlabel("random start x₀ (seed 0)")
    ax.set_ylabel("where it ends: x after 300 steps")
    ax.grid(True)
    ax.set_title(f"12 random starts: {n_global_12} find the global min")

    panel(fig.add_subplot(gs[1, 2]), "Local minima and random restarts", [
        r"$f'(x) = 12x(x+2)(x-1)$",
        "x = −2: global min (f = −32)",
        "x = 0: local max,  x = 1: local min (f = −5)",
        "",
        "Gradient descent only moves downhill,",
        "so it ends in the valley it starts in.",
        "Restarts: run from many x₀, keep the best.",
        f"12 starts (seed 0): {n_global_12} global, best f = −32.",
        f"20 starts (seed 1): {n_global_20}/20 global.",
        "More starts raise the odds, never to 100%.",
    ], spacing=0.08)
    save(fig, name="03_pitfalls.png")


# ================================================================ notebook 04
def sheet_04_random_vs_gradient():
    def naive_descent(start, step=0.3, n_dirs=4, max_iter=100, seed=0, record=None):
        rng = np.random.default_rng(seed)
        pos = np.array(start, dtype=float)
        path = [pos.copy()]
        for _ in range(max_iter):
            angles = rng.uniform(0, 2 * np.pi, n_dirs)
            if record is not None and not record:
                record.append(angles.copy())
            dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            candidates = pos + step * dirs
            temps = np.array([T_sauna(c[0], c[1]) for c in candidates])
            best = candidates[np.argmin(temps)]
            if T_sauna(*best) < T_sauna(*pos):
                pos = best
                path.append(pos.copy())
            else:
                break
        return np.array(path)

    first_angles = []
    naive = naive_descent((1.0, 1.0), record=first_angles)
    assert len(naive) - 1 == 19 and f"{T_sauna(*naive[-1]):.3f}" == "73.795"
    naive16 = naive_descent((1.0, 1.0), n_dirs=16)
    assert len(naive16) - 1 == 14 and f"{T_sauna(*naive16[-1]):.3f}" == "73.653"
    naive51 = naive_descent((5.0, 1.0))
    assert len(naive51) - 1 == 1 and f"{T_sauna(*naive51[-1]):.3f}" == "82.752"
    grad_path = gd_2d(grad_T_sauna, (1.0, 1.0), alpha=0.2, n_steps=200, tol=1e-6)
    assert len(grad_path) - 1 == 17 and f"{T_sauna(*grad_path[-1]):.3f}" == "73.622"

    fig = new_figure("Gradient descent in two variables: don't guess directions, use −∇T",
                     "Sauna T(x, y) = 85 − (1/90)·x²(x − 6)·y²(y − 6), coldest at (4, 4) with "
                     "T = 73.622. Random directions waste steps; −∇T is the steepest way down.")
    gs = grid(fig, top=0.84, wspace=0.3)

    g = np.linspace(0, 6, 200)
    X, Y = np.meshgrid(g, g)
    Z = T_sauna(X, Y)
    ax = fig.add_subplot(gs[0, 0])
    light_contours(ax, X, Y, Z, levels=np.arange(73, 86, 1.0))
    path_line(ax, naive, color=INK2, label=f"naive, 4 random dirs: {len(naive) - 1} steps")
    path_line(ax, grad_path, color=ORANGE, label=f"gradient, α = 0.2: {len(grad_path) - 1} steps")
    ax.plot(4, 4, "*", color=AQUA, markersize=18, markeredgecolor=INK, zorder=5)
    ax.plot(1, 1, "s", color=INK, markersize=6, zorder=5)
    ax.text(0.1, 0.35, "start (1, 1)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_title("Two paths to the cold spot")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(len(naive)), T_sauna(naive[:, 0], naive[:, 1]), "o-", color=INK2,
            markersize=4, label="naive (random)")
    ax.plot(np.arange(len(grad_path)), T_sauna(grad_path[:, 0], grad_path[:, 1]), "o-",
            color=ORANGE, markersize=4, label="gradient")
    ax.axhline(T_sauna(4, 4), color=AQUA, linestyle="--", linewidth=1.3, label="minimum 73.622")
    ax.set_xlabel("step")
    ax.set_ylabel("temperature T")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Temperature after each step")

    panel(fig.add_subplot(gs[0, 2]), "The naive 2D search", [
        "1. from (x, y) try a few random directions",
        "2. take a step of 0.3 along each",
        "3. move to the coldest, if it is colder",
        "4. stop when none of them is colder",
        "",
        f"From (1, 1): {len(naive) - 1} steps, ends at",
        f"({naive[-1, 0]:.3f}, {naive[-1, 1]:.3f}), T = {T_sauna(*naive[-1]):.3f}.",
        f"16 directions: {len(naive16) - 1} steps, T = {T_sauna(*naive16[-1]):.3f}.",
        f"From (5, 1): stuck after 1 step, T = {T_sauna(*naive51[-1]):.3f}.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    theta = np.linspace(0, 2 * np.pi, 400)
    change = T_sauna(1 + 0.3 * np.cos(theta), 1 + 0.3 * np.sin(theta)) - T_sauna(1, 1)
    ax.plot(np.degrees(theta), change, color=BLUE, linewidth=2.2)
    grad0 = grad_T_sauna(1.0, 1.0)
    best_angle = np.degrees(np.arctan2(-grad0[1], -grad0[0])) % 360
    ax.axvline(best_angle, color=ORANGE, linewidth=2)
    ax.text(best_angle + 6, change.max() * 0.85, f"−∇T direction\n{best_angle:.0f}°",
            fontsize=10, color=ORANGE, weight="bold")
    for a in first_angles[0]:
        d = T_sauna(1 + 0.3 * np.cos(a), 1 + 0.3 * np.sin(a)) - T_sauna(1, 1)
        ax.plot(np.degrees(a), d, "o", color=INK, markersize=8, zorder=3)
    ax.plot([], [], "o", color=INK, label="the 4 random tries of step 1")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("direction of a 0.3 step from (1, 1), degrees")
    ax.set_ylabel("change in T")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Every direction from (1, 1), one step")

    ax = fig.add_subplot(gs[1, 1])
    labels = ["naive 4 dirs\nfrom (1, 1)", "naive 16 dirs\nfrom (1, 1)", "naive 4 dirs\nfrom (5, 1)",
              "gradient\nfrom (1, 1)"]
    finals = [T_sauna(*p[-1]) - T_sauna(4, 4) for p in (naive, naive16, naive51, grad_path)]
    counts = [len(p) - 1 for p in (naive, naive16, naive51, grad_path)]
    pos = np.arange(4)
    ax.bar(pos, np.maximum(finals, 1e-9), width=0.6, color=[INK2, INK2, INK2, ORANGE], zorder=2)
    for p, v, c in zip(pos, finals, counts):
        txt = f"{v:.0e}" if v < 1e-3 else f"{v:.3f}"
        ax.text(p, max(v, 1e-9) * 1.6, f"{txt}\n{c} step{'s' if c != 1 else ''}", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e3)
    ax.set_xticks(pos, labels, fontsize=9.5)
    ax.set_ylabel("final T − 73.622 (log scale)")
    ax.grid(axis="y")
    ax.set_title("How close each run ends")

    panel(fig.add_subplot(gs[1, 2]), "The smart fix: the gradient", [
        r"$\nabla T = (T_x,\, T_y)$ points to steepest increase,",
        r"so $-\nabla T$ is the fastest way to get colder:",
        "",
        r"$(x_k, y_k) = (x_{k-1}, y_{k-1}) - \alpha\,\nabla T(x_{k-1}, y_{k-1})$",
        "",
        "The same rule as in one variable, with",
        "the derivative replaced by the gradient.",
        "No guessing: one evaluation of ∇T gives",
        "the best direction, and its length the step.",
    ], spacing=0.085)
    save(fig, name="04_random_vs_gradient.png")


# ================================================================ notebook 05
def sheet_05_gradient_descent_2d():
    start, alpha = np.array([0.5, 0.6]), 0.05
    g0 = grad_T_sauna(*start)
    x1 = start - alpha * g0
    assert f"{g0[0]:.4f}, {g0[1]:.4f}" == "-0.1134, -0.0935"
    assert f"{x1[0]:.4f}, {x1[1]:.4f}" == "0.5057, 0.6047"
    path = gd_2d(grad_T_sauna, start, alpha=0.05, n_steps=5000, tol=1e-8)
    assert len(path) - 1 == 143 and f"{T_sauna(*path[-1]):.4f}" == "73.6222"

    def f_multi(x, y):
        return 3 * x ** 4 + 4 * x ** 3 - 12 * x ** 2 + y ** 2

    def grad_multi(x, y):
        return np.array([12 * x ** 3 + 12 * x ** 2 - 24 * x, 2 * y])

    starts = [(-1.5, 2.0), (0.5, -2.0), (2.0, 1.5)]
    multi = [gd_2d(grad_multi, s, alpha=0.01, n_steps=2000, tol=1e-8) for s in starts]
    assert [round(p[-1, 0], 3) for p in multi] == [-2.0, 1.0, 1.0]
    rng = np.random.default_rng(0)                           # exercise 6
    n_global = 0
    for _ in range(10):
        s = (rng.uniform(-3, 2), rng.uniform(-2, 2))
        fin = gd_2d(grad_multi, s, alpha=0.01, n_steps=3000, tol=1e-8)[-1]
        n_global += int(np.isfinite(fin).all() and fin[0] < -1)
    assert n_global == 3
    close = [gd_2d(grad_multi, s, alpha=0.01, n_steps=3000, tol=1e-8)[-1, 0]
             for s in [(-0.1, 0.0), (0.1, 0.0)]]
    assert round(close[0], 3) == -2.0 and round(close[1], 3) == 1.0

    fig = new_figure("Gradient descent in two variables: xₖ = xₖ₋₁ − α·∇f(xₖ₋₁)",
                     "The gradient (∂f/∂x, ∂f/∂y) points uphill fastest, so step along −∇f. "
                     "Sauna from (0.5, 0.6) with α = 0.05: 143 steps to the coldest point (4, 4).")
    gs = grid(fig, top=0.84, wspace=0.3)

    g = np.linspace(0, 6, 200)
    X, Y = np.meshgrid(g, g)
    ax = fig.add_subplot(gs[0, 0])
    light_contours(ax, X, Y, T_sauna(X, Y), levels=np.arange(73, 86, 1.0))
    q = np.linspace(0.5, 5.5, 11)
    QX, QY = np.meshgrid(q, q)
    G = grad_T_sauna(QX, QY)
    norm = np.hypot(G[0], G[1]) + 1e-12
    ax.quiver(QX, QY, -G[0] / norm, -G[1] / norm, color=BLUE, scale=26, width=0.005,
              alpha=0.8, zorder=3)
    path_line(ax, path, markersize=2.5, label="143 steps from (0.5, 0.6)")
    ax.plot(4, 4, "*", color=AQUA, markersize=18, markeredgecolor=INK, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_title("Arrows: −∇T, the path follows them")

    ax = fig.add_subplot(gs[0, 1])
    p = np.array([2.0, 1.5])                                 # a point where the two parts differ
    gp = grad_T_sauna(*p)
    sc = 0.5                                                 # arrows drawn at half length
    light_contours(ax, X, Y, T_sauna(X, Y), levels=np.arange(73, 86, 0.5), filled=False)
    kw = dict(angles="xy", scale_units="xy", scale=1, width=0.012, zorder=4)
    ax.quiver(*p, sc * gp[0], 0, color=BLUE, **kw)
    ax.quiver(*p, 0, sc * gp[1], color=AQUA, **kw)
    ax.quiver(*p, sc * gp[0], sc * gp[1], color=INK2, **kw)
    ax.quiver(*p, -sc * gp[0], -sc * gp[1], color=ORANGE, **kw)
    ax.plot(*p, "o", color=INK, markersize=7, zorder=5)
    ax.text(p[0] + sc * gp[0] / 2, p[1] + 0.07, f"∂T/∂x = {gp[0]:.2f}", ha="center", fontsize=10,
            color=BLUE)
    ax.text(p[0] + 0.06, p[1] + sc * gp[1] / 2, f"∂T/∂y = {gp[1]:.2f}", ha="left", fontsize=10,
            color=AQUA)
    ax.text(p[0] + sc * gp[0], p[1] + sc * gp[1] - 0.08, "∇T (hotter)", ha="center", va="top",
            fontsize=10, color=INK2, weight="bold")
    ax.text(p[0] - sc * gp[0], p[1] - sc * gp[1] + 0.08, "−∇T (colder)", ha="center",
            fontsize=10, color=ORANGE, weight="bold")
    ax.set_xlim(0.9, 3.1)
    ax.set_ylim(0.15, 2.85)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(0.98, 0.02, "arrows at half length", transform=ax.transAxes, ha="right",
            fontsize=9, color=MUTED)
    ax.set_title("Two slopes add up to the gradient at (2, 1.5)")

    panel(fig.add_subplot(gs[0, 2]), "The worked step", [
        r"$\nabla T = -\frac{1}{90}\left(x(3x-12)\,y^2(y-6),\; x^2(x-6)\,y(3y-12)\right)$",
        "",
        "At (0.5, 0.6):  ∇T = (−0.1134, −0.0935)",
        "x₁ = (0.5, 0.6) − 0.05·∇T",
        "    = (0.5057, 0.6047)",
        "",
        "Exactly the 1D rule, with the derivative",
        "replaced by the gradient vector.",
        f"Ends at (4.0000, 4.0000), T = {T_sauna(*path[-1]):.4f}.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    temps = T_sauna(path[:, 0], path[:, 1])
    step_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    k_big = int(np.argmax(step_len))
    ax.plot(np.arange(len(path)), temps, color=ORANGE, linewidth=2.2, label="temperature T(xₖ)")
    ax.axhline(T_sauna(4, 4), color=AQUA, linestyle="--", linewidth=1.3, label="minimum 73.622")
    ax.annotate(f"first step: {step_len[0]:.4f}\n(almost flat floor)",
                (0, temps[0]), xytext=(3, 79.5), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.annotate(f"longest step {step_len[k_big]:.3f}\n(step {k_big + 1})", (k_big + 1, temps[k_big + 1]),
                textcoords="offset points", xytext=(18, 4), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlabel("iteration k")
    ax.set_ylabel("temperature")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Slow on the flat start, fast down the slope")

    ax = fig.add_subplot(gs[1, 1])
    gx, gy = np.linspace(-3, 2.2, 300), np.linspace(-2.5, 2.5, 300)
    MX, MY = np.meshgrid(gx, gy)
    light_contours(ax, MX, MY, f_multi(MX, MY), levels=[-30, -25, -20, -15, -10, -5, -2, 0, 5, 10,
                                                         20, 30, 45])
    for pth, s, color in zip(multi, starts, [AQUA, ORANGE, INK2]):
        path_line(ax, pth, color=color, markersize=2, label=f"start {s}")
    ax.plot(-2, 0, "*", color=AQUA, markersize=18, markeredgecolor=INK, zorder=5)
    ax.plot(1, 0, "P", color=ORANGE, markersize=12, markeredgecolor=INK, zorder=5)
    ax.text(-2, -0.55, "global (−2, 0)\nf = −32", ha="center", va="top", fontsize=9.5)
    ax.text(1.15, -0.25, "local (1, 0)\nf = −5", ha="left", va="top", fontsize=9.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_title("f = 3x⁴ + 4x³ − 12x² + y²: two valleys")

    panel(fig.add_subplot(gs[1, 2]), "Same drawbacks as in 1D", [
        "1. pick α and a start (x₀, y₀)",
        r"2. repeat  $\mathbf{x}_k = \mathbf{x}_{k-1} - \alpha\,\nabla f(\mathbf{x}_{k-1})$",
        "3. stop when the step is tiny",
        "",
        "Downhill only, so local minima trap it:",
        "(−0.1, 0) → global,  (0.1, 0) → local.",
        f"10 random starts (seed 0): {n_global} reach (−2, 0).",
        "Fix: many starts, keep the best;",
        "usually good, never guaranteed.",
    ], spacing=0.085)
    save(fig, name="05_gradient_descent_2d.png")


# ================================================================ notebook 06
def sheet_06_least_squares():
    pts = np.array([(1, 2), (2, 5), (3, 3)], dtype=float)

    def E(m, b):
        return 14 * m ** 2 + 3 * b ** 2 + 38 + 12 * m * b - 42 * m - 20 * b

    def grad_E(m, b):
        return np.array([28 * m + 12 * b - 42, 12 * m + 6 * b - 20])

    m_t, b_t = 0.7, 1.9
    assert np.isclose(E(m_t, b_t), sum((m_t * px + b_t - py) ** 2 for px, py in pts))
    hist = gd_2d(grad_E, (0.0, 0.0), alpha=0.02, n_steps=5000, tol=1e-9)
    assert len(hist) - 1 == 1182 and f"{E(*hist[-1]):.5f}" == "4.16667"
    assert len(gd_2d(grad_E, (2.0, 0.0), 0.02, 5000, 1e-9)) - 1 == 1205
    eig, vecs = np.linalg.eigh(np.array([[28.0, 12.0], [12.0, 6.0]]))
    assert np.round(eig, 3).tolist() == [0.721, 33.279]
    e_min = E(0.5, 7 / 3)

    fig = new_figure("Least squares by gradient descent: the power-line fit, without solving",
                     "E(m, b) = (m + b − 2)² + (2m + b − 5)² + (3m + b − 3)²; from (0, 0) with α = 0.02 "
                     "the iterates reach m = 1/2, b = 7/3, E = 4.167 in 1182 steps.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 4, 2)
    show = [0, 5, 20, 100, len(hist) - 1]
    shades = [BLUE_LIGHT, "#86b6ef", BLUE, "#1c5cab", ORANGE]
    for k, color in zip(show, shades):
        m, b = hist[k]
        ax.plot(xs, m * xs + b, color=color, linewidth=2.6 if k == show[-1] else 1.6,
                label=f"k = {k}: y = {m:.2f}x + {b:.2f}, E = {E(m, b):.2f}")
    ax.plot(pts[:, 0], pts[:, 1], "o", color=INK, markersize=9, zorder=4)
    m, b = hist[-1]
    for px, py in pts:
        ax.plot([px, px], [py, m * px + b], color=ORANGE, linestyle=":", linewidth=1.5)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.3, 7.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)
    ax.set_title("The line improves step by step")

    ax = fig.add_subplot(gs[0, 1])
    mm, bb = np.linspace(-1, 2, 200), np.linspace(0, 5, 200)
    MM, BB = np.meshgrid(mm, bb)
    light_contours(ax, MM, BB, E(MM, BB), levels=[4.5, 6, 10, 20, 40, 70, 110, 160], labels=True)
    path_line(ax, hist[::4], markersize=2.5, label="descent path (every 4th step)")
    ax.plot(0.5, 7 / 3, "*", color=AQUA, markersize=18, markeredgecolor=INK, zorder=5)
    ax.plot(0, 0, "s", color=INK, markersize=6, zorder=5)
    ax.set_xlabel("m (slope)")
    ax.set_ylabel("b (intercept)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_title("Down the cost surface E(m, b)")

    panel(fig.add_subplot(gs[0, 2]), "Cost and gradient", [
        r"$E = 14m^2 + 3b^2 + 38 + 12mb - 42m - 20b$",
        r"$\nabla E = (28m + 12b - 42,\; 12m + 6b - 20)$",
        "",
        "Analytical: solve ∇E = 0 → m = 1/2, b = 7/3.",
        "Iterative:",
        r"$(m, b)_{k+1} = (m, b)_k - \alpha\,\nabla E(m_k, b_k)$",
        "",
        "Both give E = 4.167; from (2, 0) it takes",
        "1205 steps, same answer.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    gap = np.array([E(m, b) for m, b in hist]) - e_min
    ax.plot(np.arange(len(hist)), gap, color=ORANGE, linewidth=2.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 200)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("E(mₖ, bₖ) − 4.167")
    ax.grid(True)
    ax.text(0.97, 0.93, f"E(0, 0) = {E(0, 0):.0f}\nE after 1182 steps = {E(*hist[-1]):.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.set_title("Cost gap shrinks by a constant factor")

    ax = fig.add_subplot(gs[1, 1])
    a_crit = 2 / eig.max()
    for a, color, label in [(0.02, AQUA, "α = 0.02 (notebook)"),
                            (0.06, BLUE, f"α = 0.06 (just under 2/λmax = {a_crit:.4f})"),
                            (0.061, ORANGE, "α = 0.061 (just over)")]:
        h = gd_2d(grad_E, (0.0, 0.0), a, n_steps=200, tol=1e-9)
        ax.plot(np.arange(len(h)), [E(m, b) - e_min for m, b in h], color=color, linewidth=1.8,
                label=label)
    h06 = gd_2d(grad_E, (0.0, 0.0), 0.06, n_steps=200, tol=1e-9)
    assert f"{E(*h06[-1]):.3f}" == "12.928"
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("E − 4.167")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)
    ax.set_title("α must stay below 2/λmax")

    panel(fig.add_subplot(gs[1, 2]), "Why α has to be small here", [
        "Hessian of E: [[28, 12], [12, 6]]",
        f"eigenvalues λ = {eig[0]:.3f} and {eig[1]:.3f}",
        "",
        f"Stable only for α < 2/λmax = {a_crit:.4f}.",
        f"Slowest direction shrinks by 1 − αλmin",
        f"= {1 - 0.02 * eig[0]:.4f} per step at α = 0.02,",
        "hence over a thousand steps: a long,",
        "narrow valley (ill-conditioned).",
        "α = 0.06: still E = 12.928 after 200 steps.",
    ], spacing=0.085)
    save(fig, name="06_least_squares.png")


# ================================================================ notebook 07
def sheet_07_linear_regression():
    df = pd.read_csv(REPO / "coursera" / "data" / "tvmarketing.csv")
    assert len(df) == 200
    x_raw, y_raw = df["TV"].to_numpy(), df["Sales"].to_numpy()
    x = (x_raw - x_raw.mean()) / x_raw.std()
    y = (y_raw - y_raw.mean()) / y_raw.std()

    def loss(m, b):
        return np.mean((m * x + b - y) ** 2) / 2

    def grad(m, b):
        err = m * x + b - y
        return np.array([np.mean(err * x), np.mean(err)])

    hist = gd_2d(grad, (0.0, 0.0), alpha=0.5, n_steps=1000, tol=1e-9)
    m_fit, b_fit = hist[-1]
    m_exact, b_exact = np.polyfit(x, y, 1)
    assert len(hist) - 1 == 30 and f"{m_fit:.5f}" == "0.78222"
    assert np.allclose([m_fit, b_fit], [m_exact, b_exact], atol=1e-6)
    h_slow = gd_2d(grad, (0.0, 0.0), alpha=0.05, n_steps=20000, tol=1e-9)
    assert len(h_slow) - 1 == 342
    # back to the original units: sales = m_raw * TV + b_raw
    m_raw = m_fit * y_raw.std() / x_raw.std()
    b_raw = y_raw.mean() + y_raw.std() * b_fit - m_raw * x_raw.mean()
    m_np, b_np = np.polyfit(x_raw, y_raw, 1)
    assert np.allclose([m_raw, b_raw], [m_np, b_np], atol=1e-6)
    # curvature of L in raw units vs standardized units
    h_raw = np.array([[np.mean(x_raw ** 2), np.mean(x_raw)], [np.mean(x_raw), 1.0]])
    eig_raw = np.linalg.eigvalsh(h_raw)
    h_std = np.array([[np.mean(x ** 2), np.mean(x)], [np.mean(x), 1.0]])
    assert np.allclose(h_std, np.eye(2))

    fig = new_figure("Linear regression is gradient descent on a cost surface",
                     "L(m, b) = (1/2n)·Σ(m·xᵢ + b − yᵢ)² over n = 200 TV-marketing rows; descending "
                     "it gives the same line as exact least squares.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x_raw, y_raw, s=14, color=BLUE, alpha=0.55, linewidths=0, label="200 markets")
    xs = np.linspace(0, 300, 2)
    ax.plot(xs, m_raw * xs + b_raw, color=ORANGE, linewidth=2.6,
            label=f"sales ≈ {m_raw:.4f}·TV + {b_raw:.2f}")
    ax.set_xlabel("TV budget")
    ax.set_ylabel("sales")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The fit, back in original units")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(x, y, s=12, color=MUTED, alpha=0.5, linewidths=0)
    xs = np.linspace(x.min(), x.max(), 2)
    show = [0, 1, 2, 4, len(hist) - 1]
    shades = [BLUE_LIGHT, "#86b6ef", BLUE, "#1c5cab", ORANGE]
    for k, color in zip(show, shades):
        m, b = hist[k]
        ax.plot(xs, m * xs + b, color=color, linewidth=2.6 if k == show[-1] else 1.7,
                label=f"k = {k}: m = {m:.3f}, L = {loss(m, b):.4f}")
    ax.set_xlabel("TV (standardized)")
    ax.set_ylabel("sales (standardized)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True)
    ax.set_title("Each step tilts the line (α = 0.5)")

    panel(fig.add_subplot(gs[0, 2]), "Cost, gradient, update", [
        r"$L(m,b) = \frac{1}{2n}\sum\,(m x_i + b - y_i)^2$",
        r"$\partial L/\partial m = \frac{1}{n}\sum\,(m x_i + b - y_i)\,x_i$",
        r"$\partial L/\partial b = \frac{1}{n}\sum\,(m x_i + b - y_i)$",
        r"$(m_k, b_k) = (m_{k-1}, b_{k-1}) - \alpha\,\nabla L$",
        "",
        "The ½ cancels the 2 of the square only.",
        f"GD: m = {m_fit:.5f}, b = 0.00000 in {len(hist) - 1} steps;",
        f"np.polyfit: m = {m_exact:.5f}, b = 0.00000.",
    ], spacing=0.105)

    ax = fig.add_subplot(gs[1, 0])
    mm, bb = np.linspace(-0.2, 1.0, 160), np.linspace(-0.6, 0.6, 160)
    MM, BB = np.meshgrid(mm, bb)
    LL = np.array([[loss(m, b) for m in mm] for b in bb])
    light_contours(ax, MM, BB, LL, levels=[0.2, 0.22, 0.25, 0.3, 0.4, 0.5], labels=True)
    path_line(ax, hist, markersize=4, label="α = 0.5 path")
    ax.plot(m_fit, b_fit, "*", color=AQUA, markersize=18, markeredgecolor=INK, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_ylabel("b")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_title("Round contours: straight to the bottom")

    ax = fig.add_subplot(gs[1, 1])
    l_min = loss(m_fit, b_fit)
    for h, color, label in [(hist, ORANGE, f"α = 0.5: {len(hist) - 1} steps"),
                            (h_slow, BLUE, f"α = 0.05: {len(h_slow) - 1} steps")]:
        gaps = np.array([loss(m, b) for m, b in h]) - l_min
        ax.plot(np.arange(len(h)), np.maximum(gaps, 1e-18), color=color, linewidth=2.2, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("L(mₖ, bₖ) − min L")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Learning rate sets the pace")

    panel(fig.add_subplot(gs[1, 2]), "Why standardize first", [
        "TV runs 0 to 300, sales 1 to 27.",
        "Curvature of L (its Hessian) in raw units:",
        f"eigenvalues {eig_raw[0]:.3f} and {eig_raw[1]:,.0f},",
        f"ratio ≈ {eig_raw[1] / eig_raw[0]:,.0f}: a long, narrow valley.",
        "",
        "Standardized: the Hessian is the identity,",
        "contours are circles, and α = 0.5 halves the",
        "distance to the minimum every step: 30 steps.",
    ], spacing=0.085)
    save(fig, name="07_linear_regression.png")


if __name__ == "__main__":
    sheet_00_naive_search()
    sheet_01_update_rule()
    sheet_03_pitfalls()
    sheet_04_random_vs_gradient()
    sheet_05_gradient_descent_2d()
    sheet_06_least_squares()
    sheet_07_linear_regression()
