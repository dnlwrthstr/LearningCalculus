#!/usr/bin/env python3
"""Concept sheets for coursera/05_newtons_method (Newton's method, second derivative, Hessian).

One page per notebook that summarizes its central idea. Every number is
computed here from the same functions, starting points and step rules the notebooks use,
with asserts against the values the notebooks print.
Images are written to coursera/05_newtons_method/img/.

Run from anywhere:  python scripts/make_coursera_05_newtons_method_concept_images.py
"""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "05_newtons_method"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
OMEGA = 0.5671432904          # the Omega constant, the minimum of g(x) = e^x - ln x


# ================================================================ the notebooks' functions
def newton_root(f, fprime, x0, n_iter=20, tol=1e-12):
    """Notebook 00: root finding, stops when the step is below tol."""
    x = float(x0)
    history = [x]
    for _ in range(n_iter):
        step = f(x) / fprime(x)
        x = x - step
        history.append(x)
        if abs(step) < tol:
            break
    return x, np.array(history)


def newton_optimize(gprime, gsecond, x0, n_iter=50, tol=1e-12):
    """Notebook 00: minimization = root finding on g'."""
    return newton_root(gprime, gsecond, x0, n_iter=n_iter, tol=tol)


def gradient_descent(gprime, x0, alpha=0.005, n_iter=5000, tol=1e-10):
    """Notebook 00: fixed-rate descent for comparison."""
    x = float(x0)
    hist = [x]
    for _ in range(n_iter):
        step = alpha * gprime(x)
        x -= step
        hist.append(x)
        if abs(step) < tol:
            break
    return x, np.array(hist)


def newton_2d(grad_n, H_n, start, n_iter=8):
    """Notebook 05: x <- x - d with H d = grad f."""
    p = np.array(start, dtype=float)
    history = [p.copy()]
    for _ in range(n_iter):
        d = np.linalg.solve(H_n(*p), grad_n(*p))
        p = p - d
        history.append(p.copy())
    return np.array(history)


def g(x):
    return np.exp(x) - np.log(x)


def gprime(x):
    return np.exp(x) - 1 / x


def gsecond(x):
    return np.exp(x) + 1 / x ** 2


# ================================================================ drawing helpers
def axis_lines(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)


def arrow(ax, start, end, color, lw=2.0, ms=14, z=4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, zorder=z))


def matrix_panel(ax, cells, title, colors, row_labels, col_labels):
    """A 2x2 matrix drawn as labelled tiles."""
    ax.set_xlim(-0.6, 2.05)
    ax.set_ylim(2.25, -0.45)
    ax.axis("off")
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=colors[i][j],
                                   edgecolor="none"))
            name, value = cells[i][j]
            ax.text(j + 0.5, i + 0.34, name, ha="center", va="center", fontsize=13, color=INK2)
            ax.text(j + 0.5, i + 0.64, value, ha="center", va="center", fontsize=20,
                    weight="bold", color=INK)
    for j, lab in enumerate(col_labels):
        ax.text(j + 0.5, -0.15, lab, ha="center", va="center", fontsize=11, color=INK2)
    for i, lab in enumerate(row_labels):
        ax.text(-0.12, i + 0.5, lab, ha="right", va="center", fontsize=11, color=INK2)
    ax.set_title(title)


# ================================================================ notebook 00
def sheet_00_newtons_method():
    f = lambda x: x ** 2 - 2
    fprime = lambda x: 2 * x
    root, hist = newton_root(f, fprime, x0=2.0)
    assert np.allclose(hist[:4], [2.0, 1.5, 1.41666667, 1.41421569], atol=1e-8)
    assert abs(root - np.sqrt(2)) < 1e-12

    x_min, hist_opt = newton_optimize(gprime, gsecond, x0=1.0)
    assert len(hist_opt) - 1 == 5 and abs(x_min - OMEGA) < 1e-9
    _, gd_hist = gradient_descent(gprime, x0=1.0, alpha=0.05)
    assert len(gd_hist) - 1 == 77

    cubic_roots = {x0: newton_root(lambda x: x ** 3 - x, lambda x: 3 * x ** 2 - 1, x0)[0]
                   for x0 in (0.6, 0.4)}
    assert abs(cubic_roots[0.6] - 1) < 1e-9 and abs(cubic_roots[0.4]) < 1e-9

    fig = new_figure("Newton's method: follow the tangent down to the x-axis",
                     r"Root of f:  $x_{k+1} = x_k - f(x_k)\,/\,f'(x_k)$          "
                     r"Minimum of g = root of g':  $x_{k+1} = x_k - g'(x_k)\,/\,g''(x_k)$")
    gs = grid(fig)

    # --- tangents marching to sqrt(2)
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0.8, 2.2, 300)
    ax.plot(xs, f(xs), color=BLUE, linewidth=2.2, label="f(x) = x² − 2")
    axis_lines(ax)
    shades = [ORANGE, "#f08f62", "#f4b08d", ORANGE_LIGHT]
    for k, xk in enumerate(hist[:3]):
        x_next = hist[k + 1]
        t = np.linspace(x_next - 0.05, xk + 0.1, 20)
        ax.plot(t, f(xk) + fprime(xk) * (t - xk), "--", color=shades[k], linewidth=1.6)
        ax.plot([xk, xk], [0, f(xk)], ":", color=MUTED, linewidth=1)
        ax.plot(xk, f(xk), "o", color=INK, markersize=6, zorder=4)
        ax.text(xk, -0.32, f"x{'₀₁₂'[k]}", ha="center", va="top", fontsize=11, color=INK2)
    ax.plot(np.sqrt(2), 0, "*", color=AQUA, markersize=16, zorder=5, label="root √2")
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(-1.0, 2.9)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Each tangent lands nearer √2")

    # --- one step as a triangle
    ax = fig.add_subplot(gs[0, 1])
    x0, x1 = hist[0], hist[1]
    xs = np.linspace(1.2, 2.2, 200)
    ax.plot(xs, f(xs), color=BLUE, linewidth=2.2)
    axis_lines(ax)
    ax.plot([x1 - 0.05, x0 + 0.1], [f(x0) + fprime(x0) * (x1 - 0.05 - x0),
                                   f(x0) + fprime(x0) * 0.1], "--", color=ORANGE, linewidth=1.8)
    ax.fill([x1, x0, x0], [0, 0, f(x0)], color=ORANGE_LIGHT, alpha=0.6, linewidth=0)
    ax.plot([x0, x0], [0, f(x0)], color=ORANGE, linewidth=2)
    ax.plot([x1, x0], [0, 0], color=ORANGE, linewidth=2)
    ax.plot(x0, f(x0), "o", color=INK, markersize=6, zorder=4)
    ax.plot(x1, 0, "o", color=ORANGE, markersize=7, zorder=4)
    ax.text(x0 + 0.03, f(x0) / 2, f"rise = f(x₀) = {f(x0):g}", fontsize=10.5, va="center")
    ax.text((x0 + x1) / 2, -0.15, f"run = x₀ − x₁ = {x0 - x1:g}", fontsize=10.5,
            ha="center", va="top")
    ax.text(1.22, 2.4, f"slope f'(x₀) = {fprime(x0):g} = rise / run\n"
                       f"⇒ x₁ = {x0:g} − {f(x0):g}/{fprime(x0):g} = {x1:g}",
            fontsize=10.5, va="top")
    ax.text(x1, 0.12, "x₁", ha="center", fontsize=11, color=ORANGE, weight="bold")
    ax.set_xlim(1.2, 2.2)
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.set_title("Where the update comes from")

    # --- card: the digits
    lines = [f"`k   x_k            |x_k − √2|`"]
    for k, xk in enumerate(hist[:5]):
        lines.append(f"`{k}   {xk:<13.10f}  {abs(xk - np.sqrt(2)):.1e}`")
    lines += ["",
              "Correct digits roughly double each",
              "step: quadratic convergence.",
              "",
              "Caveats: the start decides the root",
              f"(x³ − x: x₀ = 0.6 → {cubic_roots[0.6]:.0f},  x₀ = 0.4 → {abs(cubic_roots[0.4]):.0f}),",
              "and f'(x_k) = 0 breaks the division."]
    panel(fig.add_subplot(gs[0, 2]), "√2 from x₀ = 2", lines, spacing=0.068, fontsize=12)

    # --- optimization: root of g'
    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.25, 1.15, 300)
    ax.plot(xs, gprime(xs), color=BLUE, linewidth=2.2, label="g'(x) = eˣ − 1/x")
    ax.plot(xs, g(xs) - 2, color=MUTED, linewidth=1.6, label="g(x) − 2  (g = eˣ − ln x)")
    axis_lines(ax)
    for k, xk in enumerate(hist_opt[:3]):
        t = np.linspace(min(xk, hist_opt[k + 1]) - 0.05, max(xk, hist_opt[k + 1]) + 0.05, 20)
        ax.plot(t, gprime(xk) + gsecond(xk) * (t - xk), "--", color=ORANGE, linewidth=1.4)
        ax.plot(xk, gprime(xk), "o", color=INK, markersize=5, zorder=4)
    ax.axvline(OMEGA, color=AQUA, linestyle="--", linewidth=1.4)
    ax.text(OMEGA + 0.02, -2.1, "Ω = 0.5671\nmin of g, zero of g'", fontsize=10, color=AQUA)
    ax.set_xlim(0.25, 1.15)
    ax.set_ylim(-2.5, 2.4)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Minimize g: Newton on g', slope g''")

    # --- Newton vs gradient descent
    ax = fig.add_subplot(gs[1, 1])
    err_newton = np.abs(hist_opt - OMEGA)
    err_gd = np.abs(gd_hist - OMEGA)
    ax.semilogy(np.arange(len(err_gd)), np.maximum(err_gd, 1e-12), color=BLUE, linewidth=2,
                label=f"gradient descent, α = 0.05 ({len(gd_hist) - 1} steps)")
    ax.semilogy(np.arange(len(err_newton)), np.maximum(err_newton, 1e-11), "o-", color=ORANGE,
                linewidth=2, markersize=6, label=f"Newton ({len(hist_opt) - 1} steps)")
    ax.set_ylim(1e-12, 3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("|x − Ω|  (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Same g, same start x₀ = 1")

    panel(fig.add_subplot(gs[1, 2]), "Roots → optimization", [
        "A minimum of g is a zero of g'.",
        "Run Newton on f = g'; its slope is f' = g''.",
        "",
        r"$x_{k+1} = x_k - \dfrac{g'(x_k)}{g''(x_k)}$",
        "",
        "",
        "",
        f"g = eˣ − ln x from x₀ = 1: {len(hist_opt) - 1} Newton",
        f"steps vs {len(gd_hist) - 1} gradient-descent steps.",
        "",
        "The price: the second derivative g''",
        "(in many variables: the Hessian).",
    ], spacing=0.07)
    save(fig, name="00_newtons_method.png")


# ================================================================ notebook 01
def sheet_01_newton_example():
    x0 = 0.05
    hist = [x0]
    for _ in range(6):
        hist.append(hist[-1] - gprime(hist[-1]) / gsecond(hist[-1]))
    hist = np.array(hist)
    assert np.allclose(np.round(hist[1:], 3), [0.097, 0.183, 0.321, 0.477, 0.558, 0.567])
    assert abs(gprime(x0) + 18.9487) < 1e-4 and abs(gsecond(x0) - 401.0513) < 1e-4
    err = np.abs(hist - OMEGA)
    assert abs(err[-1] - 8.34e-05) < 1e-7

    h1 = [1.0]
    while abs(h1[-1] - OMEGA) > 1e-9:
        h1.append(h1[-1] - gprime(h1[-1]) / gsecond(h1[-1]))
    assert len(h1) - 1 == 4

    fig = new_figure("Newton minimizes eˣ − ln x in six steps",
                     r"$x_{k+1} = x_k - \dfrac{g'(x_k)}{g''(x_k)} = x_k - "
                     r"\dfrac{e^{x_k} - 1/x_k}{e^{x_k} + 1/x_k^{2}}$,   from $x_0 = 0.05$ "
                     r"to the Omega constant $\Omega \approx 0.5671$")
    gs = grid(fig)

    # --- zero of g'
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0.03, 0.8, 400)
    ax.plot(xs, gprime(xs), color=BLUE, linewidth=2.2, label="g'(x) = eˣ − 1/x")
    axis_lines(ax)
    for k in range(5):
        xk, xn = hist[k], hist[k + 1]
        t = np.linspace(xk, xn + 0.02, 20)
        ax.plot(t, gprime(xk) + gsecond(xk) * (t - xk), "--", color=ORANGE, linewidth=1.4)
        ax.plot([xn, xn], [0, gprime(xn)], ":", color=MUTED, linewidth=1)
        ax.plot(xk, gprime(xk), "o", color=INK, markersize=5, zorder=4)
    ax.axvline(OMEGA, color=AQUA, linestyle="--", linewidth=1.4, label="zero of g' = Ω")
    ax.set_ylim(-20, 3)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("g'(x)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Tangents of g' walk to its zero")

    # --- iterates on g
    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0.03, 1.2, 400)
    ax.plot(xs, g(xs), color=BLUE, linewidth=2.2, label="g(x) = eˣ − ln x")
    ax.plot(hist, g(hist), "o-", color=ORANGE, markersize=7, linewidth=1.6, zorder=4,
            label="Newton iterates")
    for k in (0, 1, 2, 3):
        ax.annotate(f"x{'₀₁₂₃'[k]}", (hist[k], g(hist[k])), textcoords="offset points",
                    xytext=(8, 2), fontsize=10, color=INK2)
    ax.axvline(OMEGA, color=AQUA, linestyle="--", linewidth=1.4, label="minimum at Ω")
    ax.set_ylim(2, 4.4)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("…and slide down g to its minimum")

    # --- card: the table
    lines = ["`k   x_k      g(x_k)    |x_k − Ω|`"]
    for k, xk in enumerate(hist):
        lines.append(f"`{k}   {xk:.4f}   {g(xk):.5f}   {err[k]:.2e}`")
    lines += ["", "Lecture: 0.097, 0.183, 0.320, 0.477,", "0.558, 0.567 ✓"]
    panel(fig.add_subplot(gs[0, 2]), "The six iterations", lines, spacing=0.07, fontsize=12)

    # --- step size vs curvature
    ax = fig.add_subplot(gs[1, 0])
    steps = np.diff(hist)
    curv = gsecond(hist[:-1])
    k_idx = np.arange(1, 7)
    ax.bar(k_idx, steps, width=0.6, color=ORANGE, zorder=2)
    for k, st, c in zip(k_idx, steps, curv):
        ax.text(k, st + 0.004, f"g'' = {c:.0f}", ha="center", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 0.19)
    ax.set_xticks(k_idx)
    ax.set_xlabel("step k  (g'' taken at the step's start)")
    ax.set_ylabel("step length  x_k − x_(k−1)")
    ax.grid(axis="y")
    ax.set_title("Big g'' near 0.05 → small first steps")

    # --- error, log scale
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(np.arange(len(err)), err, "o-", color=ORANGE, linewidth=2, markersize=7,
                label="from x₀ = 0.05 (lecture)")
    err1 = np.maximum(np.abs(np.array(h1) - OMEGA), 1e-12)
    ax.semilogy(np.arange(len(err1)), err1, "s-", color=BLUE, linewidth=2, markersize=6,
                label=f"from x₀ = 1.0: {len(h1) - 1} steps to 10⁻⁹")
    ax.set_ylim(1e-12, 2)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("|x_k − Ω|  (log scale)")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Slow far away, quadratic near Ω")

    panel(fig.add_subplot(gs[1, 2]), "Step 1 by hand", [
        f"g'(0.05) = e^0.05 − 1/0.05 = {gprime(x0):.4f}",
        f"g''(0.05) = e^0.05 + 1/0.05² = {gsecond(x0):.4f}",
        "",
        f"x₁ = 0.05 − ({gprime(x0):.4f}) / {gsecond(x0):.4f}",
        f"    = 0.05 + {-gprime(x0) / gsecond(x0):.4f} = {hist[1]:.4f}",
        "",
        "g'' (curvature) sizes every step:",
        "steep curvature → cautious step,",
        "gentle curvature → bold step.",
        "No learning rate to tune.",
    ], spacing=0.072)
    save(fig, name="01_newton_example.png")


# ================================================================ notebook 02
def car(t):
    t2, t5, t7 = 2 / 60, 5 / 60, 7 / 60
    x = np.where(t < t2, 750 * t ** 2,
                 np.where(t < t5, 50 * t - 5 / 6,
                          np.where(t < t7, -1500 * t ** 2 + 300 * t - 11.25, 55 / 6 - 50 * t)))
    v = np.where(t < t2, 1500 * t,
                 np.where(t < t5, 50.0, np.where(t < t7, 300 - 3000 * t, -50.0)))
    a = np.where(t < t2, 1500.0,
                 np.where(t < t5, 0.0, np.where(t < t7, -3000.0, 0.0)))
    return x, v, a


def sheet_02_second_derivative():
    t = np.linspace(0, 9 / 60, 901)
    x, v, a = car(t)
    tm = t * 60
    i_max = np.argmax(x)
    assert abs(tm[i_max] - 6) < 0.02 and abs(v[i_max]) < 1
    # the pieces join continuously, and v is the numerical slope of x
    assert np.max(np.abs(np.diff(x))) < 0.01
    assert np.allclose(np.gradient(x, t)[5:-5][np.abs(np.diff(v))[5:-4] < 1], v[5:-5][np.abs(np.diff(v))[5:-4] < 1], atol=2)

    fig = new_figure("The second derivative: how fast the slope itself changes",
                     r"$f''(x) = \dfrac{d^2 f}{dx^2}$:  distance → velocity → acceleration;   "
                     r"$f'' > 0$ concave up (minimum),  $f'' < 0$ concave down (maximum)")
    gs = grid(fig)

    phases = [(0, 2, "speed up"), (2, 5, "cruise"), (5, 7, "brake & reverse"), (7, 9, "return")]
    tints = [BLUE_LIGHT, SURFACE, ORANGE_LIGHT, SURFACE]

    def shade(ax, label_y=None):
        for (lo, hi, name), tint in zip(phases, tints):
            if tint != SURFACE:
                ax.axvspan(lo, hi, color=tint, alpha=0.45, linewidth=0, zorder=0)
            if label_y is not None:
                ax.text((lo + hi) / 2, label_y, name, ha="center", fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[0, 0])
    shade(ax, label_y=3.98)
    ax.plot(tm, x, color=BLUE, linewidth=2.4)
    ax.plot(6, x[i_max], "o", color=ORANGE, markersize=8, zorder=4)
    ax.annotate(f"farthest point, t = 6 min\nx = {x[i_max]:.2f}, v = 0", (6, x[i_max]),
                textcoords="offset points", xytext=(-150, -55), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_ylim(0, 4.35)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("distance x(t)")
    ax.grid(True)
    ax.set_title("Distance x(t)")

    ax = fig.add_subplot(gs[0, 1])
    shade(ax)
    ax.plot(tm, v, color=AQUA, linewidth=2.4)
    axis_lines(ax)
    ax.plot(6, 0, "o", color=ORANGE, markersize=8, zorder=4)
    ax.text(6.25, 8, "v = 0", fontsize=10, color=ORANGE)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("velocity v = x'")
    ax.grid(True)
    ax.set_title("Velocity = slope of distance")

    ax = fig.add_subplot(gs[0, 2])
    shade(ax)
    ax.step(tm, a, where="post", color=ORANGE, linewidth=2.4)
    axis_lines(ax)
    ax.text(1, 1650, "+1500: v rising", ha="center", fontsize=10)
    ax.text(7.25, -2900, "−3000:\nv falling", ha="left", fontsize=10)
    ax.text(3.5, 180, "0: v constant", ha="center", fontsize=10)
    ax.set_ylim(-3400, 2200)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("acceleration a = x''")
    ax.grid(True)
    ax.set_title("Acceleration = slope of velocity")

    # --- curvature: tangents rotating
    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2, 2, 200)
    ax.plot(xs, xs ** 2, color=AQUA, linewidth=2.4, label="x²:  f'' = 2 > 0")
    ax.plot(xs, -xs ** 2, color=ORANGE, linewidth=2.4, label="−x²:  f'' = −2 < 0")
    for x0 in (-1.2, 0, 1.2):
        t_ = np.linspace(x0 - 0.45, x0 + 0.45, 10)
        ax.plot(t_, x0 ** 2 + 2 * x0 * (t_ - x0), color=INK2, linewidth=1.2)
        ax.plot(t_, -x0 ** 2 - 2 * x0 * (t_ - x0), color=INK2, linewidth=1.2)
    ax.text(0, 2.6, "slopes −2.4, 0, 2.4: rising", ha="center", fontsize=10, color=AQUA)
    ax.text(0, -3.1, "slopes 2.4, 0, −2.4: falling", ha="center", fontsize=10, color=ORANGE)
    ax.set_ylim(-4.3, 4.3)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Curvature: is the slope rising or falling?")

    # --- second derivative test on x^3 - 3x
    sub = gs[1, 1].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    xs = np.linspace(-2.2, 2.2, 400)
    f = xs ** 3 - 3 * xs
    ax.plot(xs, f, color=BLUE, linewidth=2.2, label="f = x³ − 3x")
    ax.plot(-1, 2, "^", color=ORANGE, markersize=11, zorder=4)
    ax.plot(1, -2, "v", color=AQUA, markersize=11, zorder=4)
    ax.text(-1, 2.7, "max", ha="center", fontsize=10, color=ORANGE, weight="bold")
    ax.text(1, -3.6, "min", ha="center", fontsize=10, color=AQUA, weight="bold")
    axis_lines(ax)
    ax.set_ylim(-4.5, 4.5)
    ax.tick_params(labelbottom=False)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("f' = 3x² − 3 finds x = ±1; f'' decides")
    ax = fig.add_subplot(sub[1], sharex=ax)
    ax.plot(xs, 6 * xs, color=INK2, linewidth=2, label="f'' = 6x")
    ax.fill_between(xs, 6 * xs, where=xs > 0, color=AQUA, alpha=0.3, linewidth=0)
    ax.fill_between(xs, 6 * xs, where=xs < 0, color=ORANGE, alpha=0.3, linewidth=0)
    ax.plot([-1, 1], [-6, 6], "o", color=INK, markersize=6)
    ax.text(-1, -11, "−6", ha="center", fontsize=10)
    ax.text(1, 8, "+6", ha="center", fontsize=10)
    axis_lines(ax)
    ax.set_ylim(-14, 14)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)

    panel(fig.add_subplot(gs[1, 2]), "The second-derivative test", [
        "At a candidate with f'(x) = 0:",
        "  f'' > 0  →  concave up  →  local minimum",
        "  f'' < 0  →  concave down  →  local maximum",
        "  f'' = 0  →  inconclusive",
        "",
        "x⁴ − 2x²:  f''(±1) = 8 → min,  f''(0) = −4 → max",
        "x⁴:  f''(0) = 0, yet x = 0 is a minimum",
        "",
        "First derivative finds the candidates;",
        "the second classifies them. Newton's step",
        "and, in many variables, the Hessian use it.",
    ], spacing=0.072, fontsize=12)
    save(fig, name="02_second_derivative.png")


# ================================================================ notebook 03
def sheet_03_hessian():
    f = lambda x, y: 2 * x ** 2 + 3 * y ** 2 - x * y
    H = np.array([[4.0, -1.0], [-1.0, 6.0]])
    # finite-difference second partials agree with the notebook's Hessian
    h = 1e-3
    for (x0, y0) in [(0.0, 0.0), (1.3, -0.7)]:
        fxx = (f(x0 + h, y0) - 2 * f(x0, y0) + f(x0 - h, y0)) / h ** 2
        fyy = (f(x0, y0 + h) - 2 * f(x0, y0) + f(x0, y0 - h)) / h ** 2
        fxy = (f(x0 + h, y0 + h) - f(x0 + h, y0 - h) - f(x0 - h, y0 + h)
               + f(x0 - h, y0 - h)) / (4 * h ** 2)
        assert np.allclose([fxx, fxy, fyy], [4, -1, 6], atol=1e-5)
    eig, vec = np.linalg.eigh(H)
    assert np.allclose(np.round(eig, 4), [3.5858, 6.4142])

    fig = new_figure("The Hessian: all second partials in one symmetric matrix",
                     r"$f(x,y) = 2x^2 + 3y^2 - xy$:   $f' \to \nabla f = (f_x, f_y)$,   "
                     r"$f'' \to H$ with $f_{xx} = 4,\ f_{yy} = 6,\ f_{xy} = f_{yx} = -1$")
    gs = grid(fig)
    gx = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(gx, gx)
    Z = f(X, Y)
    levels = [1, 3, 6, 10, 15, 21, 28, 36, 45]

    # --- contour + gradient field
    ax = fig.add_subplot(gs[0, 0])
    ax.contour(X, Y, Z, levels=levels, cmap=SEQ, linewidths=1.4)
    q = np.linspace(-2.5, 2.5, 7)
    QX, QY = np.meshgrid(q, q)
    U, V = 4 * QX - QY, 6 * QY - QX
    ax.quiver(QX, QY, U, V, color=ORANGE, angles="xy", scale=140, width=0.006)
    ax.plot(0, 0, "*", color=INK, markersize=13)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Contours + gradient (4x − y, 6y − x)")

    # --- slices: f_xx and f_yy are the curvatures
    ax = fig.add_subplot(gs[0, 1])
    s = np.linspace(-2, 2, 200)
    ax.plot(s, f(s, 0), color=BLUE, linewidth=2.4, label="along x (y = 0): 2x²")
    ax.plot(s, f(0, s), color=AQUA, linewidth=2.4, label="along y (x = 0): 3y²")
    ax.text(1.55, 4.1, r"$f_{xx} = 4$", fontsize=13, color=BLUE, ha="left")
    ax.text(0.35, 9.3, r"$f_{yy} = 6$", fontsize=13, color=AQUA, ha="left")
    ax.set_ylim(0, 13)
    ax.set_xlabel("position along the slice")
    ax.set_ylabel("f")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title("Diagonal entries: curvature along each axis")

    # --- mixed partials
    ax = fig.add_subplot(gs[0, 2])
    s = np.linspace(-2, 2, 50)
    for x0, style in [(-1, ":"), (0, "-"), (1, "--")]:
        ax.plot(s, 4 * x0 - s, style, color=BLUE, linewidth=2)
        ax.text(1.2, 4 * x0 - 1.2 + 0.75, f"f_x at x = {x0}", fontsize=9.5, color=BLUE,
                rotation=-9)
    ax.plot(s, 6 * 1 - s, color=ORANGE, linewidth=2.4)
    ax.text(1.0, 5.75, "f_y at y = 1", fontsize=9.5, color=ORANGE, rotation=-9)
    ax.text(-1.95, 12.5, "f_x = 4x − y against y,  f_y = 6y − x against x:\n"
                         "every line has slope "
                         r"$\partial_y f_x = \partial_x f_y = -1$", fontsize=10.5, va="top")
    ax.set_ylim(-6.5, 13)
    ax.set_xlabel("the other variable (y for f_x, x for f_y)")
    ax.set_ylabel("first partial")
    ax.grid(True)
    ax.set_title("Off-diagonal: mixed partials agree")

    # --- the matrix
    ax = fig.add_subplot(gs[1, 0])
    matrix_panel(ax, [[(r"$f_{xx}$", "4"), (r"$f_{xy}$", "−1")],
                      [(r"$f_{yx}$", "−1"), (r"$f_{yy}$", "6")]],
                 "H for 2x² + 3y² − xy",
                 [[BLUE_LIGHT, ORANGE_LIGHT], [ORANGE_LIGHT, BLUE_LIGHT]],
                 ["∂/∂x", "∂/∂y"], ["then ∂/∂x", "then ∂/∂y"])
    ax.text(1.0, 2.15, "symmetric: mirror entries are equal (Clairaut)",
            ha="center", fontsize=10.5, color=INK2)

    # --- eigen-directions
    ax = fig.add_subplot(gs[1, 1])
    ax.contour(X, Y, Z, levels=levels, cmap=SEQ, linewidths=1.4)
    for lam, v_, color in zip(eig, vec.T, [AQUA, ORANGE]):
        d = v_ * 2.2 * np.sign(v_[0])
        arrow(ax, (0, 0), (d[0], d[1]), color)
        arrow(ax, (0, 0), (-d[0], -d[1]), color)
        ax.text(d[0] + 0.1, d[1] - 0.45, f"λ = {lam:.3f}", color=color, fontsize=11,
                weight="bold", ha="left" if d[1] < 0 else "center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none"))
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Eigenvalues 3.586, 6.414 > 0: a bowl")

    panel(fig.add_subplot(gs[1, 2]), "One variable → many", [
        "first derivative   f'  →  gradient ∇f (vector)",
        "second derivative  f''  →  Hessian H (matrix)",
        "",
        "Recipe: differentiate each first partial",
        "by each variable.",
        "  f_x = 4x − y   →  f_xx = 4,  f_xy = −1",
        "  f_y = 6y − x   →  f_yx = −1,  f_yy = 6",
        "",
        "Usually H depends on the point:",
        "  x²y  →  [[2y, 2x], [2x, 0]]",
        "Here H is constant because f is quadratic.",
    ], spacing=0.072, fontsize=12)
    save(fig, name="03_hessian.png")


# ================================================================ notebook 04
def sheet_04_hessian_concavity():
    cases = [
        ("Minimum: 2x² + 3y² − xy", lambda X, Y: 2 * X ** 2 + 3 * Y ** 2 - X * Y,
         np.array([[4.0, -1.0], [-1.0, 6.0]]), "positive definite"),
        ("Maximum: −2x² − 3y² − xy + 15", lambda X, Y: -2 * X ** 2 - 3 * Y ** 2 - X * Y + 15,
         np.array([[-4.0, -1.0], [-1.0, -6.0]]), "negative definite"),
        ("Saddle: 2x² − 2y²", lambda X, Y: 2 * X ** 2 - 2 * Y ** 2,
         np.array([[4.0, 0.0], [0.0, -4.0]]), "indefinite"),
    ]
    expected = [[3.5858, 6.4142], [-6.4142, -3.5858], [-4.0, 4.0]]
    for (_, _, H, _), e in zip(cases, expected):
        assert np.allclose(np.round(np.linalg.eigvalsh(H), 4), e)
    exercises = {"x² + 4y²": [2, 8], "−x² − y²": [-2, -2], "x² − 3y²": [-6, 2], "x² + y³": [0, 2]}
    ex_H = {"x² + 4y²": np.diag([2.0, 8.0]), "−x² − y²": np.diag([-2.0, -2.0]),
            "x² − 3y²": np.diag([2.0, -6.0]), "x² + y³": np.diag([2.0, 0.0])}
    for name, e in exercises.items():
        assert np.allclose(np.linalg.eigvalsh(ex_H[name]), e)

    fig = new_figure("Hessian eigenvalues sort critical points: bowl, dome, saddle",
                     r"At $\nabla f = 0$:  all $\lambda > 0$ → minimum,  all $\lambda < 0$ → "
                     r"maximum,  mixed signs → saddle,  any $\lambda = 0$ → inconclusive")
    gs = grid(fig)
    gx = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(gx, gx)

    for col, (title, fn, H, kind) in enumerate(cases):
        ax = fig.add_subplot(gs[0, col])
        Z = fn(X, Y)
        image = ax.contourf(X, Y, Z, levels=16, cmap=SEQ)
        ax.contour(X, Y, Z, levels=16, colors=SURFACE, linewidths=0.5)
        eig, vec = np.linalg.eigh(H)
        for lam, v_ in zip(eig, vec.T):
            color = AQUA if lam > 0 else ORANGE
            d = v_ * 1.45
            arrow(ax, (0, 0), (d[0], d[1]), color, lw=2.4)
            arrow(ax, (0, 0), (-d[0], -d[1]), color, lw=2.4)
        ax.plot(0, 0, "o", color=INK, markersize=6)
        cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        lam_text = ",  ".join(f"{lam:.2f}" for lam in eig)
        ax.set_title(f"{title}\n{kind}: λ = {lam_text}", fontsize=12.5)

    # --- eigenvalue plane
    ax = fig.add_subplot(gs[1, 0])
    lim = 9.5
    ax.add_patch(Rectangle((0, 0), lim, lim, color=AQUA, alpha=0.15, linewidth=0))
    ax.add_patch(Rectangle((-lim, -lim), lim, lim, color=ORANGE, alpha=0.15, linewidth=0))
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.axvline(0, color=INK2, linewidth=1.2)
    ax.text(6.5, 3.6, "MINIMUM", ha="center", fontsize=10.5, weight="bold", color=AQUA)
    ax.text(-5.5, -8.7, "MAXIMUM", ha="center", fontsize=10.5, weight="bold", color=ORANGE)
    ax.text(-5.5, 8.4, "saddle", ha="center", fontsize=10.5, color=INK2)
    ax.text(5.5, -8.7, "saddle", ha="center", fontsize=10.5, color=INK2)
    points = [("bowl", expected[0]), ("dome", expected[1]), ("saddle", expected[2])]
    points += list(exercises.items())
    offsets = {"bowl": (8, -12), "dome": (8, 4), "saddle": (8, 4), "x² + 4y²": (8, -4),
               "−x² − y²": (8, -4), "x² − 3y²": (8, -6), "x² + y³": (8, -16)}
    for name, (l1, l2) in points:
        marker = "o" if name in ("bowl", "dome", "saddle") else "s"
        ax.plot(l1, l2, marker, color=INK if marker == "o" else BLUE, markersize=8, zorder=4)
        label = name + ("  (λ = 0: inconclusive)" if l1 == 0 else "")
        ax.annotate(label, (l1, l2), textcoords="offset points", xytext=offsets[name],
                    fontsize=9.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("λ₁")
    ax.set_ylabel("λ₂")
    ax.set_title("Where the eigenvalues land")

    # --- saddle cross-sections
    ax = fig.add_subplot(gs[1, 1])
    s = np.linspace(-2, 2, 200)
    ax.plot(s, 2 * s ** 2, color=AQUA, linewidth=2.4, label="along x: 2x²  (λ = +4, curves up)")
    ax.plot(s, -2 * s ** 2, color=ORANGE, linewidth=2.4,
            label="along y: −2y²  (λ = −4, curves down)")
    axis_lines(ax)
    ax.plot(0, 0, "o", color=INK, markersize=7, zorder=4)
    ax.text(0, 9.5, "(0, 0): a minimum one way,\na maximum the other", fontsize=10.5,
            ha="center", va="center")
    ax.set_ylim(-9.5, 12)
    ax.set_xlabel("position along the slice")
    ax.set_ylabel("f")
    ax.grid(True)
    ax.legend(loc="lower center", fontsize=9.5)
    ax.set_title("Why mixed signs mean a saddle")

    panel(fig.add_subplot(gs[1, 2]), "The test, in 1, 2 and n variables", [
        "minimum:  f'' > 0  |  both λ > 0  |  all n λ > 0",
        "maximum:  f'' < 0  |  both λ < 0  |  all n λ < 0",
        "otherwise:  saddle (mixed) or inconclusive (a 0)",
        "",
        "Exercises, eigenvalues at (0, 0):",
        "  x² + 4y²: 2, 8 → minimum",
        "  −x² − y²: −2, −2 → maximum",
        "  x² − 3y²: −6, 2 → saddle",
        "  x² + y³: 0, 2 → inconclusive",
        "",
        "Always check ∇f = 0 first.",
    ], spacing=0.07, fontsize=12)
    save(fig, name="04_hessian_concavity.png")


# ================================================================ notebook 05
def sheet_05_newton_two_variables():
    grad_c = lambda x, y: np.array([np.sinh(x), np.sinh(y)])
    H_c = lambda x, y: np.array([[np.cosh(x), 0.0], [0.0, np.cosh(y)]])
    hist = newton_2d(grad_c, H_c, (4.0, 4.0), n_iter=8)
    assert np.allclose(hist[1:6, 0], [3.000671, 2.005609, 1.041188, 0.2628310, 5.889448e-03],
                       rtol=1e-6)
    assert abs(hist[6, 0] - 6.809208e-08) < 1e-13 and hist[8, 0] == 0.0

    grad_q = lambda x, y: np.array([4 * x - y, 6 * y - x])
    H_q = lambda x, y: np.array([[4.0, -1.0], [-1.0, 6.0]])
    hq = newton_2d(grad_q, H_q, (4.0, 4.0), n_iter=2)
    assert np.allclose(hq[1], 0, atol=1e-12)

    h3 = newton_2d(grad_c, H_c, (2.0, -3.0), n_iter=10)
    assert np.allclose(h3[-1], 0)

    grad_5 = lambda x, y: np.array([2 * x + x ** 3, 2 * y + y ** 3])
    H_5 = lambda x, y: np.array([[2 + 3 * x ** 2, 0.0], [0.0, 2 + 3 * y ** 2]])
    p, steps = np.array([3.0, 3.0]), 0
    while np.linalg.norm(grad_5(*p)) >= 1e-8:
        p = p - np.linalg.solve(H_5(*p), grad_5(*p))
        steps += 1
    assert steps == 6

    fig = new_figure("Newton in two variables: step by H⁻¹∇f",
                     r"$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}\,\nabla f(\mathbf{x}_k)$   "
                     r"(solve $H\mathbf{d} = \nabla f$):  gradient for the direction, "
                     r"Hessian for the shape")
    gs = grid(fig)

    # --- path on cosh + cosh
    ax = fig.add_subplot(gs[0, 0])
    gx = np.linspace(-4.5, 4.5, 300)
    GX, GY = np.meshgrid(gx, gx)
    ax.contour(GX, GY, np.cosh(GX) + np.cosh(GY), levels=np.logspace(0.4, 1.8, 18),
               cmap=SEQ, linewidths=1.2)
    ax.plot(hist[:, 0], hist[:, 1], "o-", color=ORANGE, markersize=6, linewidth=1.8, zorder=4)
    for k in range(5):
        ax.annotate(str(k), hist[k], textcoords="offset points", xytext=(-14, 4), fontsize=10,
                    color=INK2)
    ax.plot(0, 0, "*", color=INK, markersize=14, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("f = cosh x + cosh y from (4, 4)")

    # --- distance to the minimum
    ax = fig.add_subplot(gs[0, 1])
    dist = np.linalg.norm(hist, axis=1)
    k_idx = np.arange(len(dist))
    shown = dist > 0
    ax.semilogy(k_idx[shown], dist[shown], "o-", color=ORANGE, linewidth=2, markersize=7)
    for k in k_idx[shown]:
        if k >= 4:
            ax.annotate(f"{dist[k]:.1e}", (k, dist[k]), textcoords="offset points",
                        xytext=(8, 2), fontsize=9.5)
    ax.text(8, 1e-15, "k = 8:\nexactly 0\n(not on a\nlog axis)", fontsize=9.5, color=INK2,
            ha="center")
    ax.set_xlim(-0.4, 8.6)
    ax.set_ylim(1e-25, 30)
    ax.set_xticks(range(9))
    ax.set_xlabel("iteration k")
    ax.set_ylabel("distance to (0, 0)  (log)")
    ax.grid(True)
    ax.set_title("Slow at first, then digits double")

    # --- cobweb in one coordinate
    ax = fig.add_subplot(gs[0, 2])
    s = np.linspace(0, 4.3, 300)
    ax.plot(s, s - np.tanh(s), color=BLUE, linewidth=2.2, label="N(x) = x − sinh x / cosh x")
    ax.plot(s, s, color=MUTED, linewidth=1.2, linestyle="--", label="y = x")
    xs_ = hist[:, 0]
    for k in range(6):
        ax.plot([xs_[k], xs_[k]], [xs_[k], xs_[k + 1]], color=ORANGE, linewidth=1.5)
        ax.plot([xs_[k], xs_[k + 1]], [xs_[k + 1], xs_[k + 1]], color=ORANGE, linewidth=1.5)
    ax.plot(xs_[0], xs_[0], "o", color=ORANGE, markersize=6)
    ax.text(2.75, 0.35, "far out: steps of ≈ 1\n(tanh x ≈ 1)", fontsize=9.5)
    ax.set_xlim(0, 4.3)
    ax.set_ylim(0, 4.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x_k")
    ax.set_ylabel("x_(k+1)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Cobweb of one coordinate")

    # --- quadratic: one step, and why H matters
    ax = fig.add_subplot(gs[1, 0])
    gx = np.linspace(-1.5, 5, 300)
    QX, QY = np.meshgrid(gx, gx)
    ax.contour(QX, QY, 2 * QX ** 2 + 3 * QY ** 2 - QX * QY,
               levels=[1, 4, 9, 16, 25, 36, 49, 64, 81], cmap=SEQ, linewidths=1.2)
    arrow(ax, (4, 4), (0.08, 0.08), ORANGE, lw=2.6, ms=18)
    g0 = grad_q(4.0, 4.0)
    d_gd = -g0 / np.linalg.norm(g0) * 3.0
    arrow(ax, (4, 4), (4 + d_gd[0], 4 + d_gd[1]), BLUE, lw=2.2, ms=16)
    ax.plot(0, 0, "*", color=INK, markersize=14, zorder=5)
    ax.plot(4, 4, "s", color=INK, markersize=7, zorder=5)
    box = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9)
    ax.text(-1.35, 3.3, "Newton: −H⁻¹∇f\nlands on (0, 0)", fontsize=10.5, color=ORANGE,
            weight="bold", bbox=box)
    ax.text(3.2, 0.25, f"steepest descent −∇f\n= −({g0[0]:.0f}, {g0[1]:.0f}) direction",
            fontsize=10.5, color=BLUE, weight="bold", ha="center", va="top", bbox=box)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 5)
    ax.set_ylim(-1.5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Quadratic 2x² + 3y² − xy: one step")

    panel(fig.add_subplot(gs[1, 1]), "From 1D to 2D", [
        r"$x_{k+1} = x_k - [f''(x_k)]^{-1} f'(x_k)$",
        "",
        "  f'  →  gradient ∇f  (2 × 1)",
        "  f''  →  Hessian H  (2 × 2)",
        "  ÷ f''  →  multiply by H⁻¹ on the left",
        "",
        "H⁻¹ ∇f: (2×2)(2×1) = 2×1  ✓",
        "∇f H⁻¹: (2×1)(2×2)  ✗ shapes don't match",
        "",
        "cosh example:  ∇f = (sinh x, sinh y),",
        "H = diag(cosh x, cosh y)",
        "",
        "Cost: build and solve H every step,",
        "too heavy for millions of parameters.",
    ], spacing=0.064, fontsize=12)

    lines = ["`k   x_k = y_k`"]
    for k in range(9):
        lines.append(f"`{k}   {hist[k, 0]:.6e}`")
    lines += ["",
              "Other starts:  (2, −3) → (0, 0) too.",
              f"x² + y² + (x⁴ + y⁴)/4 from (3, 3):",
              f"{steps} steps to ‖∇f‖ < 10⁻⁸."]
    panel(fig.add_subplot(gs[1, 2]), "The iterates from (4, 4)", lines, spacing=0.058,
          fontsize=12)
    save(fig, name="05_newton_two_variables.png")


if __name__ == "__main__":
    sheet_00_newtons_method()
    sheet_01_newton_example()
    sheet_02_second_derivative()
    sheet_03_hessian()
    sheet_04_hessian_concavity()
    sheet_05_newton_two_variables()
