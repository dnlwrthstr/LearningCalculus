#!/usr/bin/env python3
"""Concept sheets for coursera/00_derivatives (Week 1: derivatives and their rules).

One page per notebook that summarizes its central idea. Every number is computed here from
the same example functions, points and step sizes the notebook uses, with asserts against the
values the notebook prints. Images are written to coursera/00_derivatives/img/.

Run from anywhere:  python scripts/make_coursera_00_derivatives_concept_images.py
"""
from functools import partial
from math import cos, e, exp, log, sin

import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid)
from viz_style import save as _save

LESSON = REPO / "coursera" / "00_derivatives"
save = partial(_save, folder=LESSON / "img")

RAMP = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab"]     # light -> dark blues for shrinking secants
NOTEBOOK_STEPS = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 1000]   # the Δx table used in notebooks 04-06, 10


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """viz_style.panel, but a line holding a fraction is set in display size (\\dfrac) and gets
    twice the line height, so stacked fractions stay readable and never touch the next line."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5, weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        tall = r"\frac" in text
        if tall:
            text = text.replace(r"\frac", r"\dfrac")
            y -= spacing * 0.15
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0), family="monospace" if is_code else None)
        y -= spacing * (1.95 if tall else 1.0)


def close(a, b, tol=5e-6):
    return abs(a - b) < tol


def secant_slope(f, x0, dx):
    return (f(x0 + dx) - f(x0)) / dx


def secant_picture(ax, f, x0, dxs, slope, xs, title, curve_label, xlabel="x", ylabel="f(x)",
                   ylim=None, legend_loc="upper left", fmt="{:.2f}"):
    """The curve, secants from x0 for each dx (light to dark), and the tangent in orange."""
    ax.plot(xs, f(xs), color=INK, linewidth=2.4, label=curve_label, zorder=3)
    colors = RAMP[-len(dxs):]
    for dx, color in zip(dxs, colors):
        s = secant_slope(f, x0, dx)
        ax.plot(xs, f(x0) + s * (xs - x0), color=color, linewidth=1.5,
                label=f"secant Δ = {dx:g}: slope {fmt.format(s)}")
        ax.plot(x0 + dx, f(x0 + dx), "o", color=color, markersize=6, zorder=4)
    ax.plot(xs, f(x0) + slope * (xs - x0), color=ORANGE, linestyle="--", linewidth=2.2,
            label=f"tangent: slope {fmt.format(slope)}")
    ax.plot(x0, f(x0), "o", color=INK, markersize=7, zorder=5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc=legend_loc, fontsize=9)
    ax.set_title(title)


def limit_picture(ax, dxs, slopes, target, title, target_label, ylabel="secant slope Δf/Δx",
                  xlabel="Δx (log scale)", fmt="{:.4f}", label_every=1, text_offset=(6, 4),
                  stagger=0):
    """Secant slope against the step size on a log axis, converging to the dotted target."""
    dxs, slopes = np.asarray(dxs, float), np.asarray(slopes, float)
    ax.axhline(target, color=ORANGE, linestyle=":", linewidth=1.6, label=target_label)
    ax.plot(dxs, slopes, "-o", color=BLUE, linewidth=2, markersize=6, zorder=3)
    for k, (d, s) in enumerate(zip(dxs, slopes)):
        last = k == len(dxs) - 1
        if k % label_every == 0 or last:
            offset = (-4, text_offset[1] + 6) if last else (text_offset[0], text_offset[1] + stagger * (k % 2))
            ax.annotate(fmt.format(s), (d, s), textcoords="offset points", xytext=offset,
                        ha="right" if last else "left", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(xlabel + "  ← shrinking")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(title)


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def f_and_derivative(ax, xs, f, fp, f_label, fp_label, title, ylim=None, legend_loc="upper left"):
    ax.plot(xs, f(xs), color=BLUE, linewidth=2.4, label=f_label)
    ax.plot(xs, fp(xs), color=ORANGE, linewidth=2.2, linestyle="--", label=fp_label)
    axes_cross(ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc=legend_loc, fontsize=10)
    ax.set_title(title)


def tangent_segment(ax, f, slope, x0, half=0.4, color=ORANGE, label=None, lw=2.2):
    t = np.array([x0 - half, x0 + half])
    ax.plot(t, f(x0) + slope * (t - x0), color=color, linewidth=lw, label=label, zorder=4)
    ax.plot(x0, f(x0), "o", color=INK, markersize=6, zorder=5)


# ================================================================ notebook 00
def sheet_00_instantaneous_rate():
    def x(t):
        return t ** 2

    t0 = 12.5
    dts = [4, 2, 1, 0.5, 0.1, 0.01, 0.001]
    slopes = [secant_slope(x, t0, dt) for dt in dts]
    printed = [29.0, 27.0, 26.0, 25.5, 25.1, 25.01, 25.001]
    assert all(close(s, p, 5e-6) for s, p in zip(slopes, printed))

    def numerical_derivative(f, t, h=1e-6):
        return (f(t + h) - f(t - h)) / (2 * h)

    checks = {t: numerical_derivative(x, t) for t in (5.0, 12.5, 20.0)}
    assert all(close(v, 2 * t, 5e-6) for t, v in checks.items())

    fig = new_figure("The derivative is the slope the secants settle on",
                     "Average velocity Δx/Δt over a shrinking interval → instantaneous velocity "
                     "dx/dt = lim Δx/Δt. For x(t) = t² at t = 12.5: 29, 27, 26, … → 25.")
    gs = grid(fig)

    ts = np.linspace(9, 16, 300)
    secant_picture(fig.add_subplot(gs[0, 0]), x, t0, [3, 2, 1, 0.5], 25.0, ts,
                   "Secants rotate into the tangent", "distance x(t) = t²", xlabel="time t",
                   ylabel="distance x", ylim=(60, 280), fmt="{:.1f}")

    limit_picture(fig.add_subplot(gs[0, 1]), dts, slopes, 25.0,
                  "Average velocity over [12.5, 12.5 + Δt]", "instantaneous velocity 25",
                  ylabel="average velocity Δx/Δt", xlabel="Δt (log scale)", fmt="{:g}")

    panel(fig.add_subplot(gs[0, 2]), "Average vs instantaneous", [
        "Average velocity = slope of a secant:",
        r"$\frac{\Delta x}{\Delta t} = \frac{x(t_0+\Delta t)-x(t_0)}{\Delta t}$",
        "For x = t² the algebra is exact:",
        r"$\frac{(t_0+\Delta t)^2 - t_0^2}{\Delta t} = 2t_0 + \Delta t$",
        "so at t₀ = 12.5 the error is exactly Δt.",
        "",
        "Let Δt → 0: the instantaneous velocity",
        r"$\frac{dx}{dt} = \lim_{\Delta t\to 0}\frac{\Delta x}{\Delta t} = 2t_0 = 25$",
    ], spacing=0.082)

    ax = fig.add_subplot(gs[1, 0])
    ts2 = np.linspace(0, 23, 300)
    ax.plot(ts2, x(ts2), color=INK, linewidth=2.4, label="x(t) = t²")
    for t, color in zip(checks, [AQUA, ORANGE, BLUE]):
        tangent_segment(ax, x, 2 * t, t, half=3.2, color=color, label=f"t = {t:g}: slope {2 * t:g}")
    ax.set_xlim(0, 23)
    ax.set_ylim(-20, 540)
    ax.set_xlabel("time t")
    ax.set_ylabel("distance x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("A different tangent slope at every instant")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts2, 2 * ts2, color=ORANGE, linewidth=2.2, label="exact x'(t) = 2t")
    for t, v in checks.items():
        ax.plot(t, v, "o", color=BLUE, markersize=8, zorder=3)
        ax.annotate(f"x'({t:g}) ≈ {v:.5f}", (t, v), textcoords="offset points",
                    xytext=(-14, 2) if t > 15 else (10, -14), ha="right" if t > 15 else "left",
                    fontsize=9.5, color=INK2)
    ax.plot([], [], "o", color=BLUE, label="numeric, h = 10⁻⁶")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 48)
    ax.set_xlabel("time t")
    ax.set_ylabel("velocity dx/dt")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Collect the slopes: the derivative x'(t)")

    panel(fig.add_subplot(gs[1, 2]), "The derivative", [
        "The instantaneous rate of change of a",
        "function = the slope of its tangent line.",
        "",
        r"$x'(t) = \lim_{\Delta t\to 0}\frac{x(t+\Delta t)-x(t)}{\Delta t}$",
        "",
        "Done at every t it is a new function:",
        "x(t) = t²  →  x'(t) = 2t.",
        "",
        "A tiny step h approximates it numerically:",
        "sin'(0) ≈ 1.000000 = cos(0).",
    ], spacing=0.088)
    save(fig, name="00_instantaneous_rate_of_change.png")


# ================================================================ notebook 01
def sheet_01_slopes_maxima_minima():
    t = sp.symbols("t")
    v = sp.Rational(1, 2) * (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5)
    x = sp.integrate(v, t)
    x_n, v_n = sp.lambdify(t, x, "numpy"), sp.lambdify(t, v, "numpy")
    stops = [1, 2, 3, 4, 5]
    assert sorted(sp.solve(sp.diff(x, t), t)) == stops
    heights = [float(x_n(s)) for s in stops]
    assert [round(h, 3) for h in heights] == [-19.792, -18.667, -19.125, -18.667, -19.792]

    xs_ = sp.symbols("x")
    cubic = xs_ ** 3 - 3 * xs_
    crit = [int(c) for c in sorted(sp.solve(sp.diff(cubic, xs_), xs_))]
    assert crit == [-1, 1]

    fig = new_figure("Maxima and minima sit where the slope is zero",
                     "A horizontal tangent means zero velocity (the car is stopped). Every peak "
                     "or valley of a smooth function has f'(x) = 0, so solve f' = 0 for the candidates.")
    gs = grid(fig)
    ts = np.linspace(0.6, 5.4, 500)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, x_n(ts), color=INK, linewidth=2.4, label="distance x(t)")
    for s, h in zip(stops, heights):
        ax.plot([s - 0.38, s + 0.38], [h, h], color=ORANGE, linewidth=2.6, zorder=4)
        ax.plot(s, h, "o", color=INK, markersize=6, zorder=5)
    ax.plot([], [], color=ORANGE, linewidth=2.6, label="horizontal tangent (v = 0)")
    ax.set_ylim(-20.6, -17.6)
    ax.set_xlabel("time t")
    ax.set_ylabel("distance x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    ax.set_title("Five stops: the tangent is flat")

    ax = fig.add_subplot(gs[0, 1])
    ts = np.linspace(0.85, 5.15, 500)
    vv = v_n(ts)
    ax.plot(ts, vv, color=BLUE, linewidth=2.4, label="v(t) = x'(t)")
    ax.fill_between(ts, vv, where=vv > 0, color=AQUA, alpha=0.25, linewidth=0, label="v > 0: x rising")
    ax.fill_between(ts, vv, where=vv < 0, color=ORANGE, alpha=0.25, linewidth=0, label="v < 0: x falling")
    ax.plot(stops, [0] * 5, "o", color=INK, markersize=6, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_ylim(-3.4, 2.6)
    ax.set_xlabel("time t")
    ax.set_ylabel("velocity")
    ax.grid(True)
    ax.legend(loc="lower center", fontsize=9, ncol=3)
    ax.set_title(r"$v(t)=\frac{1}{2}(t-1)(t-2)(t-3)(t-4)(t-5)$")

    panel(fig.add_subplot(gs[0, 2]), "The key property", [
        "A horizontal line has slope",
        r"$\frac{\Delta x}{\Delta t} = \frac{265-265}{20-19} = 0$  m/s: stopped.",
        "",
        "At a maximum or minimum of a smooth f:",
        r"$f'(x) = 0$  (horizontal tangent).",
        "",
        "If f' > 0 you could still go up by moving",
        "right; if f' < 0, by moving left. Only",
        "f' = 0 leaves no way further up (or down).",
    ], spacing=0.092)

    ax = fig.add_subplot(gs[1, 0])
    kinds = ["local min", "local max", "local min", "local max", "local min"]
    colors = [BLUE if k == "local min" else AQUA for k in kinds]
    for s, h, k, color in zip(stops, heights, kinds, colors):
        ax.plot([s, s], [-20.4, h], color=color, linewidth=3, zorder=2)
        ax.plot(s, h, "o", color=color, markersize=11, zorder=3)
        ax.text(s, h + 0.12, f"{h:.3f}\n{k}".replace("-", "−", 1), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(-20.4, -17.9)
    ax.set_xlim(0.4, 5.6)
    ax.set_xticks(stops, [f"t = {s}" for s in stops])
    ax.set_ylabel("distance x at the stop")
    ax.grid(axis="y")
    ax.set_title("Distance at each stop: v = 0 at every turn")

    ax = fig.add_subplot(gs[1, 1])
    fx = sp.lambdify(xs_, cubic, "numpy")
    fpx = sp.lambdify(xs_, sp.diff(cubic, xs_), "numpy")
    xx = np.linspace(-2.2, 2.2, 400)
    f_and_derivative(ax, xx, fx, fpx, "f(x) = x³ − 3x", "f'(x) = 3x² − 3",
                     "Find candidates by solving f'(x) = 0", ylim=(-4, 8))
    for c in crit:
        ax.plot(c, fx(c), "o", color=INK, markersize=7, zorder=5)
        ax.plot([c - 0.45, c + 0.45], [fx(c)] * 2, color=INK, linewidth=1.6)
        ax.plot(c, 0, "o", color=ORANGE, markersize=7, zorder=5)
        ax.annotate(f"x = {c}: f = {fx(c):g}", (c, fx(c)), textcoords="offset points",
                    xytext=(-40, 12 if c < 0 else -22), fontsize=10)

    panel(fig.add_subplot(gs[1, 2]), "How to use it", [
        "1. Differentiate:  x'(t) = v(t).",
        "2. Solve x'(t) = 0  →  t = 1, 2, 3, 4, 5.",
        "3. These are only candidates: compare",
        "    the values (or use f'' later).",
        "",
        "Here t = 2 and t = 4 are local maxima",
        "(x = −18.667), t = 1, 3, 5 local minima.",
        "",
        "x³ − 3x:  f' = 0 at x = −1 (peak, f = 2)",
        "and x = 1 (valley, f = −2).",
    ], spacing=0.085)
    save(fig, name="01_slopes_maxima_and_minima.png")


# ================================================================ notebook 02
def sheet_02_notation():
    x = sp.symbols("x")
    dictionary = [(x ** 2, "x²"), (x ** 3, "x³"), (sp.sin(x), "sin x"), (sp.exp(x), "eˣ"),
                  (sp.log(x), "ln x")]
    pretty = ["2x", "3x²", "cos x", "eˣ", "1/x"]
    derivs = [sp.diff(f, x) for f, _ in dictionary]
    assert derivs == [2 * x, 3 * x ** 2, sp.cos(x), sp.exp(x), 1 / x]
    f4 = x ** 4
    chain = [sp.diff(f4, x, k) for k in range(4)]
    assert chain == [x ** 4, 4 * x ** 3, 12 * x ** 2, 24 * x]

    def fnum(t):
        return t ** 2

    x0, slope, dx = 1.5, 3.0, 0.6

    fig = new_figure("Two notations, one slope",
                     "Lagrange f'(x) and Leibniz dy/dx = d/dx f(x) both name the slope of the "
                     "tangent: the limit of Δy/Δx as Δx → 0.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1, 3, 300)
    ax.plot(xs, fnum(xs), color=INK, linewidth=2.4, label="y = f(x) = x²")
    ax.plot(xs, fnum(x0) + slope * (xs - x0), color=ORANGE, linestyle="--", linewidth=2.2,
            label=f"tangent, slope f'({x0}) = {slope:g}")
    y0 = fnum(x0)
    ax.plot([x0, x0 + dx], [y0, y0], color=AQUA, linewidth=2.6)
    ax.plot([x0 + dx, x0 + dx], [y0, y0 + slope * dx], color=AQUA, linewidth=2.6)
    ax.text(x0 + dx / 2, y0 - 0.55, "dx = 0.6", ha="center", fontsize=10.5, color=AQUA, weight="bold")
    ax.text(x0 + dx + 0.07, y0 + slope * dx / 2, "dy = 1.8", va="center", fontsize=10.5,
            color=AQUA, weight="bold")
    ax.plot(x0, y0, "o", color=INK, markersize=7, zorder=5)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1.5, 9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("dy/dx = rise over run of the tangent")

    ax = fig.add_subplot(gs[0, 1])
    dxs = [1, 0.5, 0.25, 0.1, 0.01, 0.001]
    slopes = [secant_slope(fnum, x0, d) for d in dxs]
    assert all(close(s, 3 + d, 1e-9) for s, d in zip(slopes, dxs))
    limit_picture(ax, dxs, slopes, 3.0, "Δy/Δx becomes dy/dx at x = 1.5", "dy/dx = f'(1.5) = 3",
                  ylabel="Δy/Δx", fmt="{:g}")

    panel(fig.add_subplot(gs[0, 2]), "The two notations", [
        r"Lagrange:  $f'(x)$    ('f prime of x')",
        r"Leibniz:   $\frac{dy}{dx}\ =\ \frac{d}{dx}f(x)$",
        "d/dx is an operator: feed it a function,",
        "get its derivative back.",
        "",
        r"$dx,\ dy$: infinitesimal steps, the limits of",
        r"$\Delta x,\ \Delta y$ as the secant becomes the tangent.",
        "Same quantity, only the writing differs.",
    ], spacing=0.085)

    row = gs[1, 0:2].subgridspec(1, 4, wspace=0.42)
    xx = np.linspace(-1.5, 1.5, 300)
    names = [("f(x) = x⁴", r"$f$", r"$y$"), ("f'(x) = 4x³", r"$f'$", r"$\dfrac{dy}{dx}$"),
             ("f''(x) = 12x²", r"$f''$", r"$\dfrac{d^2y}{dx^2}$"),
             ("f'''(x) = 24x", r"$f'''$", r"$\dfrac{d^3y}{dx^3}$")]
    colors = [INK, BLUE, AQUA, ORANGE]
    for k, (expr, (label, lag, leib), color) in enumerate(zip(chain, names, colors)):
        ax = fig.add_subplot(row[k])
        fn = sp.lambdify(x, expr, "numpy")
        ax.plot(xx, fn(xx) * np.ones_like(xx), color=color, linewidth=2.4)
        axes_cross(ax)
        ax.grid(True)
        ax.set_xlabel("x")
        ax.set_title(label, fontsize=12)
        ax.text(0.5, 0.97, f"{lag}   =   {leib}", transform=ax.transAxes, ha="center", va="top",
                fontsize=13, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        if k < 3:
            ax.text(1.13, 1.065, "→", transform=ax.transAxes, ha="center", va="center",
                    fontsize=18, color=INK2)
    fig.text(0.35, 0.405, "Apply d/dx again and again: higher derivatives", ha="center",
             fontsize=13, weight="bold")

    panel(fig.add_subplot(gs[1, 2]), "A small dictionary", [
        "`f(x)       f'(x) = dy/dx`",
        *[f"`{name:<10} {p}`" for (_, name), p in zip(dictionary, pretty)],
        "",
        "Higher derivatives of y = x⁵:",
        r"$f''(x) = \frac{d^2y}{dx^2} = 20x^3$",
    ], spacing=0.088)
    save(fig, name="02_derivative_notation.png")


# ================================================================ notebook 03
def sheet_03_lines():
    x = sp.symbols("x")
    examples = [(sp.Integer(5), "5"), (3 * x + 2, "3x + 2"), (-2 * x + 7, "−2x + 7"),
                (sp.Rational(1, 2) * x - 4, "x/2 − 4")]
    slopes = [sp.diff(f, x) for f, _ in examples]
    assert slopes == [0, 3, -2, sp.Rational(1, 2)]
    a, b, dx = sp.symbols("a b Delta_x")
    assert sp.simplify(((a * (x + dx) + b) - (a * x + b)) / dx) == a

    fig = new_figure("A line's derivative is its slope; a constant's is zero",
                     "d/dx c = 0 and d/dx (ax + b) = a: every Δy/Δx on a line is the same number, "
                     "so the tangent is the line itself.")
    gs = grid(fig)
    xs = np.linspace(-3, 3, 200)

    ax = fig.add_subplot(gs[0, 0])
    for c, color in zip([7, 3.5, -1], [BLUE, AQUA, ORANGE]):
        name = f"{c:g}".replace("-", "−")
        ax.plot(xs, np.full_like(xs, c), color=color, linewidth=2.4)
        ax.plot([-1.5, 1.5], [c, c], "o", color=color, markersize=6)
        ax.text(-2.9, c + 0.35, f"f(x) = {name}", fontsize=10, color=color, weight="bold")
        ax.annotate(f"Δy = {name} − {name if c >= 0 else '(' + name + ')'} = 0", (0.4, c), textcoords="offset points", xytext=(0, 8),
                    ha="left", fontsize=9.5, color=INK2)
    axes_cross(ax)
    ax.set_ylim(-3, 9.5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.set_title("Constants: the height never changes")

    ax = fig.add_subplot(gs[0, 1])
    a_val, b_val = 2, 1
    ax.plot(xs, a_val * xs + b_val, color=INK, linewidth=2.4, label="f(x) = 2x + 1")
    for (x0, x1), color in zip([(-2.5, -1.5), (-1, 1), (1.2, 2.7)], [BLUE, AQUA, ORANGE]):
        y0, y1 = a_val * x0 + b_val, a_val * x1 + b_val
        ax.plot([x0, x1], [y0, y0], color=color, linewidth=2.4)
        ax.plot([x1, x1], [y0, y1], color=color, linewidth=2.4)
        ax.text(x1 + 0.1, (y0 + y1) / 2, f"{y1 - y0:g}/{x1 - x0:g} = {(y1 - y0) / (x1 - x0):g}",
                fontsize=10, color=color, weight="bold", va="center")
    axes_cross(ax)
    ax.set_xlim(-3, 4.3)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Any interval on 2x + 1 gives Δy/Δx = 2")

    panel(fig.add_subplot(gs[0, 2]), "The algebra", [
        "Constant f(x) = c:",
        r"$\frac{\Delta y}{\Delta x} = \frac{c-c}{x_1-x_0} = 0$",
        "",
        "Line f(x) = ax + b:",
        r"$\frac{[a(x+\Delta x)+b]-[ax+b]}{\Delta x} = \frac{a\,\Delta x}{\Delta x} = a$",
        "",
        "ax and b cancel, then Δx cancels.",
        "No Δx is left, so the limit is just a.",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    for (f, name), color in zip(examples, [MUTED, BLUE, ORANGE, AQUA]):
        fn = sp.lambdify(x, f, "numpy")
        ax.plot(xs, fn(xs) * np.ones_like(xs), color=color, linewidth=2.4, label=f"f(x) = {name}")
    axes_cross(ax)
    ax.set_ylim(-8, 17)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    ax.set_title("The notebook's four examples")

    ax = fig.add_subplot(gs[1, 1])
    for (f, name), s, color in zip(examples, slopes, [MUTED, BLUE, ORANGE, AQUA]):
        ax.plot(xs, np.full_like(xs, float(s)), color=color, linewidth=2.4)
        below = s == 0
        ax.text(-2.9, float(s) + (-0.12 if below else 0.12), f"({name})' = {s}".replace("-", "−"),
                va="top" if below else "bottom", fontsize=10, color=color, weight="bold")
    axes_cross(ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.grid(True)
    ax.set_title("...and their derivatives: flat lines")

    panel(fig.add_subplot(gs[1, 2]), "Read it off", [
        r"$\frac{d}{dx}\,c = 0$",
        r"$\frac{d}{dx}(ax+b) = a$",
        "",
        "f(x) = 7      →  f'(x) = 0",
        "g(x) = 4x − 9 →  g'(x) = 4",
        "h(x) = −x + 2 →  h'(x) = −1",
        "",
        "The intercept b never matters: shifting",
        "a graph up or down keeps every slope.",
    ], spacing=0.088)
    save(fig, name="03_derivatives_of_lines.png")


# ================================================================ notebook 04
def sheet_04_quadratics():
    def f(t):
        return t ** 2

    slopes = [secant_slope(f, 1.0, dx) for dx in NOTEBOOK_STEPS]
    printed = [3.0, 2.5, 2.25, 2.125, 2.0625, 2.001]
    assert all(close(s, p) for s, p in zip(slopes, printed))
    x, dx, a, b, c = sp.symbols("x Delta_x a b c")
    assert sp.expand(((x + dx) ** 2 - x ** 2) / dx) == 2 * x + dx
    assert sp.diff(a * x ** 2 + b * x + c, x) == 2 * a * x + b

    fig = new_figure("The derivative of x² is 2x",
                     "Δf/Δx = ((x+Δx)² − x²)/Δx = 2x + Δx → 2x. At x = 1 the secant slopes "
                     "3, 2.5, 2.25, … approach 2.")
    gs = grid(fig)

    secant_picture(fig.add_subplot(gs[0, 0]), f, 1.0, [1, 0.5, 0.25], 2.0,
                   np.linspace(-0.5, 2.5, 300), "Secants at x = 1 rotate into the tangent",
                   "f(x) = x²", ylim=(-1.5, 6.5))

    limit_picture(fig.add_subplot(gs[0, 1]), NOTEBOOK_STEPS, slopes, 2.0,
                  "Secant slope = 2 + Δx, exactly", "tangent slope 2", fmt="{:g}")

    ax = fig.add_subplot(gs[0, 2])
    xv, d = 1.0, 0.35
    ax.add_patch(Rectangle((0, 0), xv, xv, facecolor=BLUE_LIGHT, edgecolor=INK2, linewidth=1.2))
    ax.add_patch(Rectangle((xv, 0), d, xv, facecolor=ORANGE, alpha=0.75, edgecolor=INK2, linewidth=1.2))
    ax.add_patch(Rectangle((0, xv), xv, d, facecolor=ORANGE, alpha=0.75, edgecolor=INK2, linewidth=1.2))
    ax.add_patch(Rectangle((xv, xv), d, d, facecolor=AQUA, alpha=0.8, edgecolor=INK2, linewidth=1.2))
    ax.text(xv / 2, xv / 2, "x²", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(xv + d / 2, xv / 2, "x·Δx", ha="center", va="center", fontsize=11, rotation=90)
    ax.text(xv / 2, xv + d / 2, "x·Δx", ha="center", va="center", fontsize=11)
    ax.text(xv + d / 2, xv + d / 2, "Δx²", ha="center", va="center", fontsize=10)
    ax.text(xv / 2, -0.09, "x", ha="center", fontsize=12, color=INK2)
    ax.text(xv + d / 2, -0.09, "Δx", ha="center", fontsize=12, color=INK2)
    ax.text(-0.04, 1.47, "Δf = 2x·Δx + Δx²   ⇒   Δf/Δx = 2x + Δx", fontsize=11, color=INK2)
    ax.set_xlim(-0.05, 1.5)
    ax.set_ylim(-0.15, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Grow a square of side x by Δx")

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-3, 3, 300)
    ax.plot(xs, f(xs), color=INK, linewidth=2.4, label="f(x) = x²")
    for p, color in zip([-2, 0, 1], [ORANGE, AQUA, BLUE]):
        tangent_segment(ax, f, 2 * p, p, half=0.75, color=color, label=f"x = {p}: slope {2 * p}")
    ax.set_ylim(-1, 9)
    ax.set_xlim(-3, 3)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("Negative left, zero at the vertex, positive right")

    f_and_derivative(fig.add_subplot(gs[1, 1]), xs, f, lambda t: 2 * t, "f(x) = x²",
                     "f'(x) = 2x", "x² and its derivative 2x", ylim=(-6, 9))

    panel(fig.add_subplot(gs[1, 2]), "The general quadratic", [
        r"$\frac{d}{dx}\,(ax^2+bx+c) = 2ax + b$",
        "",
        "`3x²           →  6x`",
        "`x² + 2x + 1   →  2x + 2`",
        "`−2x² + 5x     →  5 − 4x`",
        "`3x² + 2x + 1  →  6x + 2`",
        "",
        "Slope zero where 2ax + b = 0:",
        "the vertex. For x² that is x = 0.",
    ], spacing=0.088)
    save(fig, name="04_derivative_of_quadratics.png")


# ================================================================ notebook 05
def sheet_05_polynomials():
    def f(t):
        return t ** 3

    x0 = 0.5
    slopes = [secant_slope(f, x0, dx) for dx in NOTEBOOK_STEPS]
    printed = [3.25, 1.75, 1.1875, 0.95312, 0.84766, 0.7515]
    assert all(close(s, p, 6e-6) for s, p in zip(slopes, printed))
    x, dx = sp.symbols("x Delta_x")
    assert sp.expand(((x + dx) ** 3 - x ** 3) / dx) == 3 * x ** 2 + 3 * x * dx + dx ** 2
    assert [sp.diff(x ** n, x) for n in range(6)] == [0, 1, 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4]
    assert sp.diff(2 * x ** 3 - x ** 2 + 5 * x - 7, x) == 6 * x ** 2 - 2 * x + 5

    fig = new_figure("The power rule: bring the exponent down, lower it by one",
                     "d/dx xⁿ = n·xⁿ⁻¹. For x³: Δf/Δx = 3x² + 3x·Δx + Δx² → 3x², "
                     "so the slope at x = 0.5 is 0.75.")
    gs = grid(fig)

    secant_picture(fig.add_subplot(gs[0, 0]), f, x0, [1, 0.5, 0.25], 0.75,
                   np.linspace(-0.4, 1.7, 300), "Secants at x = 0.5 on x³", "f(x) = x³",
                   ylim=(-0.5, 4))

    limit_picture(fig.add_subplot(gs[0, 1]), NOTEBOOK_STEPS, slopes, 0.75,
                  "Secant slopes approach 0.75", "tangent slope 3·0.5² = 0.75", fmt="{:.4g}")

    ax = fig.add_subplot(gs[0, 2])
    steps = NOTEBOOK_STEPS
    t1 = np.full(len(steps), 3 * x0 ** 2)
    t2 = np.array([3 * x0 * d for d in steps])
    t3 = np.array([d ** 2 for d in steps])
    pos = np.arange(len(steps))
    ax.bar(pos, t1, width=0.62, color=BLUE, label="3x² = 0.75 (stays)", zorder=2)
    ax.bar(pos, t2, width=0.62, bottom=t1, color=ORANGE, label="3x·Δx (vanishes)", zorder=2)
    ax.bar(pos, t3, width=0.62, bottom=t1 + t2, color=AQUA, label="Δx² (vanishes faster)", zorder=2)
    for p, total in zip(pos, t1 + t2 + t3):
        ax.text(p, total + 0.05, f"{total:.4g}", ha="center", fontsize=9.5)
    ax.set_xticks(pos, ["1", "1/2", "1/4", "1/8", "1/16", "1/1000"])
    ax.set_xlabel("Δx")
    ax.set_ylabel("Δf/Δx at x = 0.5")
    ax.set_ylim(0, 3.7)
    ax.grid(axis="y")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Each slope = 3x² + 3x·Δx + Δx²")

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-1.3, 1.3, 300)
    for n, color in zip(range(1, 6), [MUTED, BLUE, ORANGE, AQUA, INK]):
        rhs = f"{n}x^{n - 1}" if n > 2 else ("2x" if n == 2 else "1")
        ax.plot(xs, n * xs ** (n - 1), color=color, linewidth=2.2, label=f"$(x^{n})' = {rhs}$")
    axes_cross(ax)
    ax.set_ylim(-4, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("derivative")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("Derivatives of x¹ … x⁵")

    xs2 = np.linspace(-2, 2, 300)
    f_and_derivative(fig.add_subplot(gs[1, 1]), xs2, f, lambda t: 3 * t ** 2, "f(x) = x³",
                     "f'(x) = 3x²", "x³ never decreases: 3x² ≥ 0", ylim=(-8, 12))

    panel(fig.add_subplot(gs[1, 2]), "One rule for every polynomial", [
        r"$\frac{d}{dx}\,x^n = n\,x^{n-1}$",
        "",
        "x⁰ = 1 → 0,  x¹ → 1,  x² → 2x,  x³ → 3x²",
        "",
        "Term by term:",
        "`2x³ − x² + 5x − 7 → 6x² − 2x + 5`",
        "`x⁵ + 3x⁴         → 5x⁴ + 12x³`",
        "`x⁵ − 2x³ + 4x    → 5x⁴ − 6x² + 4`",
    ], spacing=0.09)
    save(fig, name="05_power_rule_for_polynomials.png")


# ================================================================ notebook 06
def sheet_06_other_powers():
    def f(t):
        return 1 / t

    slopes = [secant_slope(f, 1.0, dx) for dx in NOTEBOOK_STEPS]
    printed = [-0.5, -0.66667, -0.8, -0.88889, -0.94118, -0.999]
    assert all(close(s, p, 6e-6) for s, p in zip(slopes, printed))
    assert all(close(s, -1 / (1 + dx), 1e-12) for s, dx in zip(slopes, NOTEBOOK_STEPS))
    x = sp.symbols("x", positive=True)
    for n in [-1, 100, -100, sp.Rational(1, 2), -sp.Rational(1, 2)]:
        assert sp.simplify(sp.diff(x ** n, x) - n * x ** (n - 1)) == 0
    assert sp.simplify(sp.diff(1 / x ** 2 + 3 * sp.sqrt(x), x) - (-2 / x ** 3 + 3 / (2 * sp.sqrt(x)))) == 0

    fig = new_figure("The power rule works for any exponent",
                     "1/x = x⁻¹ has derivative −x⁻² = −1/x²: Δf/Δx = −1/((x+Δx)x) → −1/x². "
                     "Negative and fractional n follow the same n·xⁿ⁻¹.")
    gs = grid(fig)

    secant_picture(fig.add_subplot(gs[0, 0]), f, 1.0, [1, 0.5, 0.25], -1.0,
                   np.linspace(0.4, 2.5, 300), "Secants at x = 1 on the hyperbola",
                   "f(x) = 1/x", ylim=(0, 2.5), legend_loc="upper right")

    limit_picture(fig.add_subplot(gs[0, 1]), NOTEBOOK_STEPS, slopes, -1.0,
                  "Slope = −1/(1 + Δx) → −1", "tangent slope −1", fmt="{:.4g}")

    panel(fig.add_subplot(gs[0, 2]), "The derivation", [
        r"$\frac{\Delta f}{\Delta x} = \frac{\frac{1}{x+\Delta x}-\frac{1}{x}}{\Delta x}$",
        r"$= \frac{x-(x+\Delta x)}{(x+\Delta x)\,x\,\Delta x}$",
        r"$= \frac{-1}{(x+\Delta x)\,x}\ \ \to\ \ -\frac{1}{x^2}$",
        "",
        "At x = 2 the numeric slope is −0.25000",
        "= −1/2², as the formula says.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0.25, 3, 300)
    ax.plot(xs, f(xs), color=INK, linewidth=2.4, label="f(x) = 1/x")
    for p, color in zip([0.5, 1, 2], [ORANGE, BLUE, AQUA]):
        tangent_segment(ax, f, -1 / p ** 2, p, half=0.22 if p == 0.5 else 0.45, color=color,
                        label=f"x = {p:g}: slope −1/x² = {-1 / p ** 2:g}")
    ax.set_xlim(0.25, 3)
    ax.set_ylim(0, 4)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Steep near 0, flattening as x grows")

    xp = np.linspace(0.3, 3, 300)
    f_and_derivative(fig.add_subplot(gs[1, 1]), xp, f, lambda t: -1 / t ** 2, "f(x) = 1/x",
                     "f'(x) = −1/x²", "1/x and its derivative −1/x² (< 0)", ylim=(-6, 4),
                     legend_loc="upper right")

    panel(fig.add_subplot(gs[1, 2]), "Any exponent, same rule", [
        r"$x^{2} \to 2x \qquad x^{-1} \to -x^{-2} = -1/x^2$",
        r"$x^{100} \to 100\,x^{99} \qquad x^{-100} \to -100\,x^{-101}$",
        r"$x^{1/2} \to \frac{1}{2\sqrt{x}} \qquad x^{-1/2} \to -\frac{1}{2x^{3/2}}$",
        r"$x^{-2} + 3\sqrt{x} \to -\frac{2}{x^3} + \frac{3}{2\sqrt{x}}$",
        "",
        "Bring n down, subtract 1 from the",
        "exponent, whatever sign n has.",
    ], spacing=0.088)
    save(fig, name="06_power_rule_any_exponent.png")


# ================================================================ notebook 07
def sheet_07_inverse():
    x, y = sp.symbols("x y", positive=True)
    fprime, gprime = sp.diff(x ** 2, x), sp.diff(sp.sqrt(y), y)
    assert sp.simplify(gprime.subs(y, x ** 2) - 1 / fprime) == 0
    pairs = [(0.5, 1.0), (1, 2), (2, 4), (3, 6)]
    for xv, fp in pairs:
        assert fprime.subs(x, xv) == fp and sp.simplify(gprime.subs(y, xv ** 2) - 1 / sp.nsimplify(fp)) == 0

    fig = new_figure("An inverse's slope is the reciprocal",
                     "g = f⁻¹ undoes f, and its graph is f mirrored over y = x. Mirroring swaps "
                     "rise and run, so g'(y) = 1 / f'(x) where y = f(x).")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 2.5, 300)
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.4, label="f(x) = x²")
    ax.plot(xs, np.sqrt(xs), color=ORANGE, linewidth=2.4, label="g(x) = √x")
    ax.plot(xs, xs, color=MUTED, linestyle="--", linewidth=1.4, label="y = x (mirror)")
    for a, b_ in [(0.5, 0.25), (1, 1), (1.5, 2.25)]:
        ax.plot(a, b_, "o", color=BLUE, markersize=7, zorder=4)
        ax.plot(b_, a, "o", color=ORANGE, markersize=7, zorder=4)
        if a != 1:
            ax.plot([a, b_], [b_, a], color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(1.5, 2.25, "  (3/2, 9/4)", fontsize=9.5, va="center", color=BLUE)
    ax.text(2.42, 1.28, "(9/4, 3/2)", fontsize=9.5, va="top", ha="right", color=ORANGE)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("(a, b) on f  ↔  (b, a) on f⁻¹")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.4)
    ax.plot(xs, np.sqrt(xs), color=ORANGE, linewidth=2.4)
    ax.plot(xs, xs, color=MUTED, linestyle="--", linewidth=1.2)
    p0, p1 = (1, 1), (1.5, 2.25)
    # f's triangle goes up then across; its mirror image goes across then up
    ax.plot([p0[0], p0[0], p1[0]], [p0[1], p1[1], p1[1]], color=BLUE, linewidth=2.4)
    ax.plot([p0[1], p1[1], p1[1]], [p0[0], p0[0], p1[0]], color=ORANGE, linewidth=2.4)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "o", color=BLUE, markersize=6, zorder=5)
    ax.plot([p0[1], p1[1]], [p0[0], p1[0]], "o", color=ORANGE, markersize=6, zorder=5)
    ax.text(0.95, 1.62, "Δf = 1.25", ha="right", va="center", fontsize=10, color=BLUE)
    ax.text(1.25, 2.31, "Δx = 0.5", ha="center", fontsize=10, color=BLUE)
    ax.text(1.62, 0.92, "Δy = 1.25", ha="center", va="top", fontsize=10, color=ORANGE)
    ax.text(2.3, 1.25, "Δg\n= 0.5", ha="left", va="center", fontsize=10, color=ORANGE)
    ax.text(1.05, 0.45, "f secant: 1.25 / 0.5 = 2.5", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(1.05, 0.2, "g secant: 0.5 / 1.25 = 0.4", fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.set_title("The mirror swaps rise and run")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$g = f^{-1}$ means  $g(f(x)) = x$  (not $1/f$).",
        "",
        r"Secant on g:  $\frac{\Delta g}{\Delta y} = \frac{\Delta x}{\Delta f}$;  let Δ → 0:",
        r"$g'(y) = \frac{1}{f'(x)}$,   $y = f(x)$",
        r"For $x^2$:  $g'(y)=\frac{1}{2x}=\frac{1}{2\sqrt{y}}$, the power rule",
        "answer for √y.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    xs3 = np.linspace(0, 3, 300)
    ax.plot(xs3, xs3 ** 2, color=BLUE, linewidth=2.4)
    ax.plot(xs3, np.sqrt(xs3), color=ORANGE, linewidth=2.4)
    ax.plot(xs3, xs3, color=MUTED, linestyle="--", linewidth=1.2)
    ax.plot(xs3, 1 + 2 * (xs3 - 1), color=BLUE, linewidth=1.4, linestyle="--",
            label="f tangent at (1, 1): slope 2")
    ax.plot(xs3, 1 + 0.5 * (xs3 - 1), color=ORANGE, linewidth=1.4, linestyle="--",
            label="g tangent at (1, 1): slope 1/2")
    ax.plot(1, 1, "o", color=INK, markersize=7, zorder=5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Tangents at (1, 1): 2 and 1/2")

    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(len(pairs))
    fvals = [fp for _, fp in pairs]
    gvals = [1 / fp for _, fp in pairs]
    ax.bar(pos - 0.2, fvals, width=0.36, color=BLUE, label="f'(x) = 2x", zorder=2)
    ax.bar(pos + 0.2, gvals, width=0.36, color=ORANGE, label="g'(y) = 1/(2√y), y = x²", zorder=2)
    for p, fv, gv in zip(pos, fvals, gvals):
        ax.text(p - 0.2, fv + 0.1, f"{fv:g}", ha="center", fontsize=10)
        ax.text(p + 0.2, gv + 0.1, sp.nsimplify(gv).__str__(), ha="center", fontsize=10)
    ax.set_xticks(pos, [f"x = {xv:g}\ny = {xv ** 2:g}" for xv, _ in pairs])
    ax.set_ylim(0, 7.2)
    ax.set_ylabel("slope")
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Matching points: the slopes multiply to 1")

    panel(fig.add_subplot(gs[1, 2]), "Use it", [
        "At (2, 4) ↔ (4, 2):  f'(2) = 4,  g'(4) = 1/4.",
        "At (3, 9) ↔ (9, 3):  f'(3) = 6,  g'(9) = 1/6.",
        "",
        "Preview: ln is the inverse of eˣ.",
        r"$f'(x) = e^x = y$,  so  $\frac{d}{dy}\ln y = \frac{1}{y}$.",
        "",
        "Differentiate the easy direction,",
        "then flip the slope.",
    ], spacing=0.092)
    save(fig, name="07_derivative_of_the_inverse.png")


# ================================================================ notebook 08
def sheet_08_trig():
    points = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2])
    labels = ["−π", "−π/2", "0", "π/2"]
    assert np.allclose(np.cos(points), [-1, 0, 1, 0])
    x = sp.symbols("x")
    assert sp.diff(sp.sin(x), x) == sp.cos(x) and sp.diff(sp.cos(x), x) == -sp.sin(x)
    assert sp.diff(sp.sin(x), x, 100) == sp.sin(x)
    steps = [1, 0.5, 0.25, 0.1, 0.01, 0.001]
    sin_slopes = [secant_slope(np.sin, 1.0, d) for d in steps]
    cos_slopes = [secant_slope(np.cos, 1.0, d) for d in steps]
    assert close(cos(1.0), 0.54030, 5e-6) and close(-sin(1.0), -0.84147, 5e-6)

    fig = new_figure("Slopes of sin are the values of cos",
                     "(sin x)' = cos x  and  (cos x)' = −sin x; differentiating cycles "
                     "sin → cos → −sin → −cos → sin with period 4.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-np.pi - 0.5, np.pi / 2 + 0.6, 400)
    ax.plot(xs, np.sin(xs), color=INK, linewidth=2.4, label="sin x")
    ax.plot(xs, np.cos(xs), color=BLUE, linewidth=2, linestyle="--", label="cos x")
    offsets = [(-14, -20), (0, -20), (-14, 10), (0, 12)]
    for p, lab, off in zip(points, labels, offsets):
        tangent_segment(ax, np.sin, np.cos(p), p, half=0.55)
        ax.plot(p, np.cos(p), "s", color=BLUE, markersize=6, zorder=5)
        ax.annotate(f"slope {np.cos(p):+.0f}".replace("-", "−").replace("+0", "0"), (p, np.sin(p)),
                    textcoords="offset points", xytext=off, ha="right" if off[0] < 0 else "center",
                    fontsize=9.5, color=ORANGE, weight="bold")
    ax.plot([], [], color=ORANGE, linewidth=2.2, label="tangent of sin")
    ax.set_xticks(points, labels)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("At −π, −π/2, 0, π/2: slope = cos x")

    ax = fig.add_subplot(gs[0, 1])
    x0, dx = 0.9, 0.45
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED, linewidth=1.4)
    P = np.array([np.cos(x0), np.sin(x0)])
    Q = np.array([np.cos(x0 + dx), np.sin(x0 + dx)])
    corner = np.array([Q[0], P[1]])
    for pt, color in [(P, BLUE), (Q, AQUA)]:
        ax.plot([0, pt[0]], [0, pt[1]], color=color, linewidth=2)
        ax.plot(*pt, "o", color=color, markersize=6)
    ax.plot([P[0], P[0]], [P[1], 0], color=BLUE, linestyle=":", linewidth=1)
    ax.plot([0, P[0]], [0, 0], color=BLUE, linewidth=3)
    tri = np.array([P, corner, Q, P])
    ax.fill(tri[:, 0], tri[:, 1], color=ORANGE, alpha=0.35)
    ax.plot(tri[:, 0], tri[:, 1], color=ORANGE, linewidth=2)
    ax.text(corner[0] + 0.01, P[1] - 0.025, "−Δcos x", ha="left", va="top", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.text(corner[0] - 0.02, (P[1] + Q[1]) / 2, "Δsin x", ha="right", va="center", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.text((P[0] + Q[0]) / 2 + 0.03, (P[1] + Q[1]) / 2 + 0.03, "h ≈ Δx", fontsize=10.5, color=ORANGE,
            weight="bold")
    ax.text(P[0] / 2, -0.06, "cos x", ha="center", va="top", fontsize=10.5, color=BLUE)
    ax.text(P[0] + 0.02, P[1] / 2, "sin x", fontsize=10.5, color=BLUE)
    ax.text(0.16, 0.05, "x", fontsize=11, color=BLUE)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.12, 1.08)
    ax.set_aspect("equal")
    axes_cross(ax)
    ax.grid(True)
    ax.set_title(f"Unit circle: nudge the angle (x = {x0}, Δx = {dx})", fontsize=12.5)

    panel(fig.add_subplot(gs[0, 2]), "The geometric proof", [
        "In the small orange triangle (top angle φ):",
        r"$\Delta\sin x = h\cos\varphi$",
        r"$-\Delta\cos x = h\sin\varphi$",
        "",
        "As Δx → 0:  h → arc length Δx,  φ → x.",
        r"$\frac{\Delta \sin x}{\Delta x}\ \to\ \cos x$",
        r"$\frac{\Delta \cos x}{\Delta x}\ \to\ -\sin x$",
    ], spacing=0.105)

    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(cos(1.0), color=BLUE, linestyle=":", linewidth=1.6, label=f"cos 1 = {cos(1.0):.5f}")
    ax.axhline(-sin(1.0), color=ORANGE, linestyle=":", linewidth=1.6, label=f"−sin 1 = {-sin(1.0):.5f}")
    ax.plot(steps, sin_slopes, "-o", color=BLUE, linewidth=2, label="Δsin/Δx at x = 1")
    ax.plot(steps, cos_slopes, "-o", color=ORANGE, linewidth=2, label="Δcos/Δx at x = 1")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-1.15, 0.85)
    ax.set_xlabel("Δx (log scale)  ← shrinking")
    ax.set_ylabel("secant slope")
    ax.grid(True)
    ax.legend(loc="center left", fontsize=9, ncol=2, bbox_to_anchor=(0, 0.55))
    ax.set_title("Numerical check at x = 1")

    ax = fig.add_subplot(gs[1, 1])
    xx = np.linspace(-2 * np.pi, 2 * np.pi, 600)
    for fn, color, style, label in [(np.sin(xx), INK, "-", "sin x"),
                                    (np.cos(xx), BLUE, "-", "(sin x)' = cos x"),
                                    (-np.sin(xx), ORANGE, "--", "(sin x)'' = −sin x"),
                                    (-np.cos(xx), AQUA, "--", "(sin x)''' = −cos x")]:
        ax.plot(xx, fn, color=color, linestyle=style, linewidth=2, label=label)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xticks(np.arange(-2, 3) * np.pi, ["−2π", "−π", "0", "π", "2π"])
    ax.set_ylim(-1.3, 2.1)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.set_title("Each derivative shifts the wave a quarter turn")

    panel(fig.add_subplot(gs[1, 2]), "Use it", [
        r"$(\sin x)'' = -\sin x$:  $f'' = -f$,",
        "the signature of oscillation.",
        "",
        "100 = 4·25, so the 100th derivative",
        "of sin x is sin x again.",
        "",
        "With the other rules:",
        "`sin(3x)  → 3cos(3x)`",
        "`x·sin x  → sin x + x·cos x`",
        "`x²·cos x → 2x·cos x − x²·sin x`",
    ], spacing=0.085)
    save(fig, name="08_derivatives_of_sin_and_cos.png")


# ================================================================ notebook 09
def sheet_09_euler_number():
    def end_of_year(n, principal=1.0):
        return principal * (1 + 1 / n) ** n

    banks = [1, 2, 3, 12, 365]
    values = [end_of_year(n) for n in banks]
    assert [round(v, 4) for v in values[:3]] == [2.0, 2.25, 2.3704]
    assert close(values[3], 2.613035, 5e-7) and close(values[4], 2.714567, 5e-7)
    four_years = [end_of_year(n) ** 4 for n in (1, 2, 3)]
    assert [round(v, 2) for v in four_years] == [16.0, 25.63, 31.57]
    table = {n: (1 + 1 / n) ** n for n in [1, 10, 100, 1000, 10000, 1_000_000]}
    assert close(table[10], 2.593742, 5e-7) and close(table[1_000_000], 2.718280, 5e-7)

    fig = new_figure("e is what continuous compounding gives you",
                     "Pay 100%/n interest n times a year and $1 becomes (1 + 1/n)ⁿ, "
                     "which grows toward e = lim (1 + 1/n)ⁿ = 2.71828…")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    tt = np.linspace(0, 1, 2001)
    for n, color in zip([1, 2, 3, 12], [MUTED, BLUE_LIGHT, BLUE, AQUA]):
        money = (1 + 1 / n) ** np.floor(n * tt + 1e-12)
        ax.step(tt, money, where="post", color=color, linewidth=2, label=f"Bank {n}: \\${end_of_year(n):.4f}")
    ax.plot(tt, np.exp(tt), color=ORANGE, linewidth=2.4, label=f"Bank ∞ (continuous): \\${e:.4f}")
    ax.set_xlabel("time in years")
    ax.set_ylabel(r"balance (\$, start \$1)")
    ax.set_ylim(0.9, 2.85)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("One year at 100%, paid out n times")

    ax = fig.add_subplot(gs[0, 1])
    names = ["1", "2", "3", "12", "365", "∞"]
    vals = values + [e]
    colors = [BLUE] * 5 + [ORANGE]
    pos = np.arange(len(vals))
    ax.bar(pos, vals, width=0.62, color=colors, zorder=2)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.04, f"{v:.4f}", ha="center", fontsize=9.5)
    ax.axhline(e, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.set_xticks(pos, names)
    ax.set_ylim(1.5, 2.95)
    ax.set_xlabel("payouts per year n")
    ax.set_ylabel("year-end factor (1 + 1/n)ⁿ")
    ax.grid(axis="y")
    ax.set_title("More often is always better, but capped")

    panel(fig.add_subplot(gs[0, 2]), "Why it grows", [
        "Accrued interest: interest earns interest.",
        r"Bank 3:  \$1 → 4/3 → 16/9 → 64/27 = 2.37",
        "",
        "Over 4 years the gap widens:",
        r"Bank 1: \$16.00,  Bank 2: \$25.63,",
        r"Bank 3: \$31.57.",
        "",
        "Bank ∞ pays an infinitesimal amount at",
        r"every instant: $e = \lim_{n\to\infty}(1+\frac{1}{n})^n$",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    ns = np.unique(np.logspace(0, 5, 200).astype(int))
    vals_n = (1 + 1 / ns) ** ns
    ax.semilogx(ns, vals_n, color=BLUE, linewidth=2.4, label=r"$(1+1/n)^n$")
    ax.axhline(e, color=ORANGE, linestyle="--", linewidth=1.6, label=f"e = {e:.5f}")
    for n in [1, 10, 100, 1000]:
        ax.plot(n, table[n], "o", color=INK, markersize=6, zorder=4)
        ax.annotate(f"{table[n]:.6f}", (n, table[n]), textcoords="offset points", xytext=(8, -14),
                    fontsize=9.5, color=INK2)
    ax.set_ylim(1.9, 2.85)
    ax.set_xlabel("n (compounding periods per year)")
    ax.set_ylabel("year-end factor")
    ax.grid(True, which="major")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("The limit that defines e")

    ax = fig.add_subplot(gs[1, 1])
    gap = e - vals_n
    ax.loglog(ns, gap, color=BLUE, linewidth=2.4, label=r"$e - (1+1/n)^n$")
    ax.loglog(ns, e / (2 * ns), color=ORANGE, linestyle="--", linewidth=1.6, label=r"$\approx e/(2n)$")
    err50k = abs((1 + 1 / 50000) ** 50000 - e)
    assert close(err50k, 2.72e-5, 5e-8)
    ax.plot(50000, err50k, "o", color=INK, markersize=7, zorder=4)
    ax.annotate(f"n = 50,000: {err50k:.2e}", (50000, err50k), textcoords="offset points",
                xytext=(-140, -4), fontsize=9.5)
    ax.set_xlabel("n")
    ax.set_ylabel("distance to e")
    ax.grid(True, which="major")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Slow convergence: error shrinks like 1/n")

    panel(fig.add_subplot(gs[1, 2]), "Why calculus cares", [
        "e ≈ 2.718281828… is irrational.",
        "",
        "Its defining property:",
        r"$\frac{d}{dx}\,e^x = e^x$",
        "",
        "eˣ is its own derivative, so e appears",
        "wherever growth is proportional to the",
        "current amount.",
        r"$1000\,e^3 = \$20{,}085.54$ (3 years, continuous).",
    ], spacing=0.088)
    save(fig, name="09_meaning_of_e.png")


# ================================================================ notebook 10
def sheet_10_derivative_of_exp():
    x0 = 2.0
    slopes = [secant_slope(np.exp, x0, dx) for dx in NOTEBOOK_STEPS]
    printed = [12.69648, 9.58688, 8.39472, 7.87073, 7.62485, 7.39275]
    assert all(close(s, p, 6e-6) for s, p in zip(slopes, printed))
    small = [1, 0.1, 0.01, 0.001, 1e-6]
    ratios = [(exp(d) - 1) / d for d in small]
    assert [round(r, 6) for r in ratios] == [1.718282, 1.051709, 1.005017, 1.0005, 1.0]
    assert all(close(s, exp(x0) * r, 1e-9) for s, r in
               zip(slopes, [(exp(d) - 1) / d for d in NOTEBOOK_STEPS]))

    fig = new_figure("eˣ is its own derivative: slope equals height",
                     "(e^(x+Δx) − eˣ)/Δx = eˣ · (e^Δx − 1)/Δx, and (e^Δx − 1)/Δx → 1, "
                     "so d/dx eˣ = eˣ. At x = 2 the slope is e² = 7.389.")
    gs = grid(fig)

    secant_picture(fig.add_subplot(gs[0, 0]), np.exp, x0, [1, 0.5, 0.25], exp(2),
                   np.linspace(0.5, 3, 300), "Secants at x = 2 on eˣ", r"$f(x)=e^x$",
                   ylim=(0, 25))

    limit_picture(fig.add_subplot(gs[0, 1]), NOTEBOOK_STEPS, slopes, exp(2),
                  "Secant slopes approach the height e²", f"e² = {exp(2):.5f}", fmt="{:.5g}")

    panel(fig.add_subplot(gs[0, 2]), "Why", [
        "Factor the exponential out:",
        r"$\frac{e^{x+\Delta x}-e^x}{\Delta x} = e^x\cdot\frac{e^{\Delta x}-1}{\Delta x}$",
        "",
        "The second factor does not depend on x",
        "and tends to 1 (bottom-left panel).",
        "",
        r"$\frac{d}{dx}\,e^x = e^x\cdot 1 = e^x$",
    ], spacing=0.1)

    ax = fig.add_subplot(gs[1, 0])
    limit_picture(ax, small, ratios, 1.0, r"The key limit $(e^{\Delta x}-1)/\Delta x \to 1$",
                  "limit 1", ylabel=r"$(e^{\Delta x}-1)/\Delta x$", fmt="{:.6f}", stagger=14)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-2, 2.6, 300)
    ax.plot(xs, np.exp(xs), color=INK, linewidth=2.4, label=r"$e^x$ (and its derivative)")
    for a, color in zip([0, 1, 2], [AQUA, BLUE, ORANGE]):
        tt = np.linspace(a - 1, a + 0.4, 50)
        ax.plot(tt, exp(a) + exp(a) * (tt - a), color=color, linewidth=1.8,
                label=f"x = {a}: height = slope = {exp(a):.3f}")
        ax.plot([a, a], [0, exp(a)], color=color, linestyle=":", linewidth=1.3)
        ax.plot(a, exp(a), "o", color=INK, markersize=6, zorder=5)
        ax.plot(a - 1, 0, "o", color=color, markersize=6, zorder=5)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_ylim(-0.5, 11)
    ax.set_xlim(-2, 2.6)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Every tangent meets the axis 1 unit back")

    panel(fig.add_subplot(gs[1, 2]), "Use it", [
        "Slope of eˣ at x = 0:  e⁰ = 1.",
        "Slope at x = 2:  e² = 7.38906.",
        "",
        "Rise / run = eᵃ / 1 at every point a:",
        "that is 'slope equals height'.",
        "",
        "dy/dx = y: growth proportional to the",
        "current amount is exponential.",
        r"Chain rule preview: $\frac{d}{dx}e^{3x} = 3e^{3x}$",
    ], spacing=0.088)
    save(fig, name="10_derivative_of_e_to_the_x.png")


# ================================================================ notebook 11
def sheet_11_derivative_of_log():
    x, y = sp.symbols("x y", positive=True)
    assert sp.simplify(1 / sp.exp(x).subs(x, sp.ln(y))) == 1 / y
    assert sp.diff(sp.ln(y), y) == 1 / y
    assert close(exp(2), 7.38905609893065, 1e-12) and close(1 / exp(2), 0.1353352832366127, 1e-15)
    num2 = (log(2 + 1e-6) - log(2)) / 1e-6
    assert close(num2, 0.5, 5e-7)

    fig = new_figure("The derivative of ln x is 1/x",
                     "ln undoes eˣ, so its graph is eˣ mirrored over y = x. The inverse rule gives "
                     "d/dy ln y = 1 / e^(ln y) = 1/y.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 2.2, 300)
    xp = np.linspace(0.05, 8, 300)
    ax.plot(xs, np.exp(xs), color=BLUE, linewidth=2.4, label=r"$f(x)=e^x$")
    ax.plot(xp, np.log(xp), color=ORANGE, linewidth=2.4, label=r"$g(y)=\ln y$")
    ax.plot([-2, 8], [-2, 8], color=MUTED, linestyle="--", linewidth=1.2, label="y = x")
    for a, b_, lab in [(0, 1, "1"), (1, e, "e"), (2, e ** 2, "e²")]:
        ax.plot(a, b_, "o", color=BLUE, markersize=7, zorder=4)
        ax.plot(b_, a, "o", color=ORANGE, markersize=7, zorder=4)
        ax.plot([a, b_], [b_, a], color=MUTED, linestyle=":", linewidth=1.1)
        ax.annotate(f"({a}, {lab})", (a, b_), textcoords="offset points", xytext=(-48, 2), fontsize=9.5,
                    color=BLUE)
        ax.annotate(f"({lab}, {a})", (b_, a), textcoords="offset points", xytext=(4, -15), fontsize=9.5,
                    color=ORANGE)
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title(r"$e^{\ln x}=x$:  mirror images")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, np.exp(xs), color=BLUE, linewidth=2.4)
    ax.plot(xp, np.log(xp), color=ORANGE, linewidth=2.4)
    ax.plot([-2, 8], [-2, 8], color=MUTED, linestyle="--", linewidth=1.2)
    t1 = np.linspace(1.55, 2.25, 20)
    ax.plot(t1, exp(2) + exp(2) * (t1 - 2), color=BLUE, linewidth=2.2,
            label=f"eˣ at (2, e²): slope e² = {exp(2):.3f}")
    t2 = np.linspace(2.5, 8, 20)
    ax.plot(t2, 2 + (t2 - exp(2)) / exp(2), color=ORANGE, linewidth=2.2,
            label=f"ln at (e², 2): slope 1/e² = {1 / exp(2):.4f}")
    ax.plot(2, exp(2), "o", color=INK, markersize=7, zorder=5)
    ax.plot(exp(2), 2, "o", color=INK, markersize=7, zorder=5)
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Reflected point, reciprocal slope")

    panel(fig.add_subplot(gs[0, 2]), "The derivation", [
        r"Inverse rule: $\frac{d}{dy}f^{-1}(y) = \frac{1}{f'(f^{-1}(y))}$",
        "",
        r"With $f'(x)=e^x$ and $f^{-1}(y)=\ln y$:",
        r"$\frac{d}{dy}\ln y = \frac{1}{e^{\ln y}} = \frac{1}{y}$",
        "",
        "At y = e²: the slope of eˣ was e² = 7.389,",
        "so the slope of ln is 1/e² = 0.1353.",
    ], spacing=0.1)

    xx = np.linspace(0.1, 6, 300)
    f_and_derivative(fig.add_subplot(gs[1, 0]), xx, np.log, lambda t: 1 / t, r"$f(x)=\ln x$",
                     r"$f'(x)=1/x$", "ln x and its derivative 1/x", ylim=(-3, 6),
                     legend_loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    pts = [(1, "1"), (2, "2"), (e, "e"), (e ** 2, "e²")]
    pos = np.arange(len(pts))
    vals = [1 / p for p, _ in pts]
    ax.bar(pos, vals, width=0.6, color=[BLUE, AQUA, BLUE, BLUE], zorder=2)
    for p, v, (_, lab) in zip(pos, vals, pts):
        exact = {"1": "1", "2": "1/2", "e": "1/e", "e²": "1/e²"}[lab]
        ax.text(p, v + 0.03, f"{exact} = {v:.4f}", ha="center", fontsize=10)
    ax.text(1, 0.62, f"numeric:\n{num2:.6f}", ha="center", fontsize=9.5, color=AQUA)
    ax.set_xticks(pos, [f"x = {lab}" for _, lab in pts])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("slope of ln x")
    ax.grid(axis="y")
    ax.set_title("Slope of ln at a few points: 1/x")

    panel(fig.add_subplot(gs[1, 2]), "Read it", [
        r"$\ln x$ is the exponent with $e^{\ln x} = x$:",
        "ln 3 = 1.0986,  ln e = 1,  ln 10 = 2.3026.",
        "",
        "Its slope 1/x is huge near 0 and fades",
        "toward 0: ln keeps growing, ever slower.",
        "",
        "Chain rule preview:",
        r"$\frac{d}{dx}\ln(3x) = \frac{1}{x}$  (ln 3x = ln 3 + ln x)",
    ], spacing=0.09)
    save(fig, name="11_derivative_of_log_x.png")


# ================================================================ notebook 12
def sheet_12_existence():
    def one_sided_slopes(f, x0, h=1e-6):
        return (f(x0) - f(x0 - h)) / h, (f(x0 + h) - f(x0)) / h

    L, R = one_sided_slopes(np.abs, 0.0)
    assert round(L, 1) == -1.0 and round(R, 1) == 1.0
    cube_slopes = {h: np.cbrt(h) / h for h in [0.1, 0.01, 0.001, 1e-6]}
    assert [round(v, 2) for v in cube_slopes.values()] == [4.64, 21.54, 100.0, 10000.0]

    def step(t):
        return np.where(t < -1, 2.0, t + 1)

    fig = new_figure("No unique tangent, no derivative",
                     "A function fails to be differentiable at corners and cusps, at jumps, "
                     "and at vertical tangents. Continuity is needed but not enough.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 2, 401)
    ax.plot(xs, np.abs(xs), color=INK, linewidth=2.4, label="|x|")
    for s, color in [(-0.35, AQUA), (0.0, BLUE), (0.5, MUTED)]:
        ax.plot(xs, s * xs, color=color, linewidth=1.3, linestyle="--")
    ax.plot([], [], color=BLUE, linestyle="--", label="many lines touch at 0")
    ax.plot(0, 0, "o", color=ORANGE, markersize=9, zorder=5)
    ax.text(-0.85, 1.75, "left slope −1", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.text(0.85, 1.75, "right slope +1", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    axes_cross(ax)
    ax.set_ylim(-1, 2.75)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    ax.set_title("1. Corner: |x| at x = 0")

    ax = fig.add_subplot(gs[0, 1])
    xl, xr = np.linspace(-3, -1, 100), np.linspace(-1, 2, 100)
    ax.plot(xl, np.full_like(xl, 2.0), color=INK, linewidth=2.4)
    ax.plot(xr, xr + 1, color=INK, linewidth=2.4)
    ax.plot(-1, 2, "o", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=9, zorder=5)
    ax.plot(-1, float(step(np.array(-1.0))), "o", color=ORANGE, markersize=9, zorder=5)
    ax.annotate("", xy=(-1.0, 0.12), xytext=(-1.0, 1.88),
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.6))
    ax.text(-1.15, 1.0, "jump 2 → 0:\nlift the pencil", fontsize=10.5, color=ORANGE, va="center",
            ha="right", weight="bold")
    axes_cross(ax)
    ax.set_ylim(-0.6, 3.3)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.set_title("2. Jump: f = 2 for x < −1, x + 1 after")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(xs, np.cbrt(xs), color=INK, linewidth=2.4, label=r"$x^{1/3}$")
    ax.axvline(0, color=ORANGE, linestyle="--", linewidth=1.8, label="vertical tangent")
    ax.plot(0, 0, "o", color=ORANGE, markersize=9, zorder=5)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("3. Vertical tangent: x^(1/3) at x = 0")

    ax = fig.add_subplot(gs[1, 0])
    hs = np.logspace(-6, 0, 30)
    left = [one_sided_slopes(np.abs, 0.0, h)[0] for h in hs]
    right = [one_sided_slopes(np.abs, 0.0, h)[1] for h in hs]
    ax.plot(hs, right, "-o", color=BLUE, markersize=4, linewidth=2, label="from the right: +1")
    ax.plot(hs, left, "-o", color=ORANGE, markersize=4, linewidth=2, label="from the left: −1")
    ax.fill_between(hs, left, right, color=MUTED, alpha=0.12, linewidth=0)
    ax.text(1e-3, 0, "the gap never closes", ha="center", va="center", fontsize=11, color=INK2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("h (log scale)  ← shrinking")
    ax.set_ylabel("one-sided slope of |x| at 0")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Corner: the two limits disagree")

    ax = fig.add_subplot(gs[1, 1])
    hh = np.logspace(-6, -0.5, 60)
    ax.loglog(hh, np.cbrt(hh) / hh, color=BLUE, linewidth=2.2, label=r"slope over $[0, h]$ = $h^{-2/3}$")
    for h, v in cube_slopes.items():
        ax.plot(h, v, "o", color=ORANGE, markersize=7, zorder=4)
        ax.annotate(f"{v:,.2f}", (h, v), textcoords="offset points", xytext=(8, 2), fontsize=10)
    ax.invert_xaxis()
    ax.set_xlabel("h (log scale)  ← shrinking")
    ax.set_ylabel("secant slope (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Vertical tangent: the slope blows up")

    panel(fig.add_subplot(gs[1, 2]), "The three failures", [
        "Corner / cusp, e.g. |x| at 0:",
        "    one-sided slopes −1 ≠ +1.",
        "Jump discontinuity, e.g. the step:",
        "    not continuous, so no tangent.",
        "Vertical tangent, e.g. x^(1/3) at 0:",
        r"    $\frac{1}{3}x^{-2/3}\to\infty$, rise over zero run.",
        "",
        "|x| is continuous but not differentiable:",
        "continuity is necessary, not sufficient.",
    ], spacing=0.088)
    save(fig, name="12_existence_of_the_derivative.png")


# ================================================================ notebook 13
def sheet_13_scalar_multiple():
    x = sp.symbols("x")
    assert sp.diff(x ** 2, x).subs(x, 1) == 2 and sp.diff(2 * x ** 2, x).subs(x, 1) == 4
    cs = [2, 5, -3, sp.Rational(1, 2)]
    assert [sp.diff(c * x ** 2, x) for c in cs] == [4 * x, 10 * x, -6 * x, x]

    def g(t):
        return t ** 2

    def f(t):
        return 2 * t ** 2

    assert (g(2) - g(1)) / 1 == 3 and (f(2) - f(1)) / 1 == 6

    fig = new_figure("Scale a function by c, scale every slope by c",
                     "If f(x) = c·g(x) then f'(x) = c·g'(x). A vertical stretch keeps the run "
                     "and multiplies the rise: x² has slope 2 at x = 1, 2x² has slope 4.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-0.3, 2.03, 300)
    ax.plot(xs, g(xs), color=BLUE, linewidth=2.4, label="g(x) = x²")
    ax.plot(xs, f(xs), color=ORANGE, linewidth=2.4, label="f(x) = 2x²")
    for fn, color, dx_off in [(g, BLUE, 0.0), (f, ORANGE, 0.0)]:
        ax.plot([1, 2], [fn(1), fn(1)], color=color, linewidth=2, linestyle=":")
        ax.plot([2, 2], [fn(1), fn(2)], color=color, linewidth=2.4)
        ax.plot([1, 2], [fn(1), fn(2)], "o", color=color, markersize=6)
    ax.text(2.06, 2.5, "rise 3\nslope 3", fontsize=10, color=BLUE, va="center")
    ax.text(2.06, 5.0, "rise 6\nslope 6", fontsize=10, color=ORANGE, va="center")
    ax.text(1.5, 0.35, "run 1 (same)", ha="center", fontsize=10, color=INK2)
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.3, 9)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Secant from x = 1 to 2: slope doubles")

    ax = fig.add_subplot(gs[0, 1])
    xs2 = np.linspace(-2, 2, 300)
    ax.plot(xs2, g(xs2), color=BLUE, linewidth=2.4, label="g(x) = x²")
    ax.plot(xs2, f(xs2), color=ORANGE, linewidth=2.4, label="f(x) = 2x²")
    ax.plot(xs2, 1 + 2 * (xs2 - 1), color=BLUE, linestyle="--", linewidth=1.5,
            label="g tangent at x = 1: slope 2")
    ax.plot(xs2, 2 + 4 * (xs2 - 1), color=ORANGE, linestyle="--", linewidth=1.5,
            label="f tangent at x = 1: slope 4")
    ax.plot(1, 1, "o", color=BLUE, markersize=7)
    ax.plot(1, 2, "o", color=ORANGE, markersize=7)
    ax.set_ylim(-1, 8)
    ax.set_xlim(-2, 2)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Tangents at x = 1: 2 → 4")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$\frac{d}{dx}\,[c\,g(x)] = c\,g'(x)$",
        "",
        "From the definition:",
        r"$\frac{c\,g(x+h)-c\,g(x)}{h} = c\cdot\frac{g(x+h)-g(x)}{h}$",
        "and c simply rides along into the limit.",
        "",
        "Stretching by c maps (x, y) to (x, cy):",
        "same run, c times the rise.",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    colors = [BLUE, AQUA, ORANGE, MUTED]
    for c, color in zip(cs, colors):
        ax.plot(xs2, float(c) * xs2 ** 2, color=color, linewidth=2.2, label=f"{c}·x²".replace("-", "−"))
    ax.plot(xs2, xs2 ** 2, color=INK, linewidth=1.4, linestyle=":", label="x²")
    axes_cross(ax)
    ax.set_ylim(-9, 18)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, ncol=3)
    ax.set_title("c·x² for c = 2, 5, −3, 1/2")

    ax = fig.add_subplot(gs[1, 1])
    xs3 = np.linspace(-1.5, 1.5, 200)
    for c, color in zip(cs, colors):
        ax.plot(xs3, 2 * float(c) * xs3, color=color, linewidth=2.2,
                label=f"({c}·x²)' = {sp.diff(c * x ** 2, x)}".replace("*", "").replace("-", "−"))
    ax.plot(xs3, 2 * xs3, color=INK, linewidth=1.4, linestyle=":", label="(x²)' = 2x")
    axes_cross(ax)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-10, 25)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.set_title("Their derivatives: c·2x")

    panel(fig.add_subplot(gs[1, 2]), "Use it", [
        "`7x³      → 7·3x² = 21x²`",
        "`−2 sin x → −2 cos x`",
        "`5 g(x)   → 5 g'(x)`",
        "",
        "If g'(1) = 3, then (10g)'(1) = 30.",
        "",
        "A negative c flips the graph and every",
        "slope's sign; c = 1/2 halves them.",
    ], spacing=0.09)
    save(fig, name="13_scalar_multiple_rule.png")


# ================================================================ notebook 14
def sheet_14_sum_rule():
    x = sp.symbols("x")
    f1, f2 = sp.sin(x), x ** 2
    assert sp.diff(f1 + f2, x) == sp.cos(x) + 2 * x
    s1, s2 = cos(1.0), 2.0
    assert close(s1, 0.5403, 5e-5) and close(s1 + s2, 2.5403, 5e-5)
    v_boat, v_child = 0.6, 0.05
    assert close(v_boat + v_child, 0.65, 1e-12)

    fig = new_figure("The slope of a sum is the sum of the slopes",
                     "(g + h)' = g' + h'. A child running in a moving boat: distances add, so "
                     "velocities add (0.6 + 0.05 = 0.65 m/s), so derivatives add.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 20, 200)
    ax.plot(t, v_boat * t + v_child * t, color=INK, linewidth=2.4,
            label=f"child vs Earth: slope {v_boat + v_child:g}")
    ax.plot(t, v_boat * t, color=BLUE, linewidth=2.2, label=f"boat: slope {v_boat:g}")
    ax.plot(t, v_child * t, color=ORANGE, linewidth=2.2, label=f"child in boat: slope {v_child:g}")
    t0, dt = 12, 4
    ax.plot([t0 + dt, t0 + dt], [v_boat * t0, v_boat * (t0 + dt)], color=BLUE, linewidth=4)
    ax.plot([t0 + dt, t0 + dt], [v_boat * (t0 + dt), (v_boat + v_child) * (t0 + dt) - v_child * t0],
            color=ORANGE, linewidth=4)
    ax.plot([t0, t0 + dt], [v_boat * t0] * 2, color=MUTED, linestyle=":")
    ax.text(t0 + dt + 0.4, 8.1, r"$\Delta x_b$", fontsize=11, color=BLUE, va="center")
    ax.text(t0 + dt + 0.4, 9.1, r"$+\,\Delta x_c$", fontsize=11, color=ORANGE, va="center")
    ax.set_xlabel("time t (s)")
    ax.set_ylabel("distance (m)")
    ax.set_xlim(0, 20)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Distances add, over the same Δt")

    ax = fig.add_subplot(gs[0, 1])
    ax.barh([1], [v_boat], color=BLUE, height=0.45, zorder=2)
    ax.barh([1], [v_child], left=[v_boat], color=ORANGE, height=0.45, zorder=2)
    ax.barh([0], [v_boat + v_child], color=INK2, height=0.45, zorder=2)
    ax.text(v_boat / 2, 1, "boat 0.6", ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(v_boat + v_child + 0.01, 1, "child 0.05", va="center", fontsize=11, color=ORANGE)
    ax.text((v_boat + v_child) / 2, 0, "vs Earth 0.65 m/s", ha="center", va="center", color="white",
            fontsize=11, weight="bold")
    ax.set_yticks([0, 1], ["total", "parts"])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("velocity (m/s)")
    ax.grid(axis="x")
    ax.set_title("Velocities add: 0.6 + 0.05 = 0.65")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$f = g + h\ \Rightarrow\ f' = g' + h'$",
        "",
        "Over the same Δt:",
        r"$\Delta x_{total} = \Delta x_b + \Delta x_c$",
        "Divide by Δt, let Δt → 0:",
        r"$v_{total} = v_b + v_c$",
        "",
        "Running against the boat: the signed",
        "velocities still add (0.6 − 0.05).",
    ], spacing=0.092)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2, 2, 300)
    ax.plot(xs, np.sin(xs), color=BLUE, linewidth=2.2, label="g(x) = sin x")
    ax.plot(xs, xs ** 2, color=ORANGE, linewidth=2.2, label="h(x) = x²")
    ax.plot(xs, np.sin(xs) + xs ** 2, color=INK, linewidth=2.6, label="f(x) = sin x + x²")
    for fn, slope, color in [(np.sin, s1, BLUE), (lambda u: u ** 2, s2, ORANGE),
                             (lambda u: np.sin(u) + u ** 2, s1 + s2, INK)]:
        tangent_segment(ax, fn, slope, 1.0, half=0.55, color=color, lw=1.6)
    ax.annotate(f"slopes {s1:.4f} + {s2:g} = {s1 + s2:.4f}", (1, sin(1) + 1), textcoords="offset points",
                xytext=(-170, 30), fontsize=10)
    axes_cross(ax)
    ax.set_ylim(-1.5, 5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Tangents at x = 1")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xs, np.cos(xs), color=BLUE, linewidth=2.2, label="g'(x) = cos x")
    ax.plot(xs, 2 * xs, color=ORANGE, linewidth=2.2, label="h'(x) = 2x")
    ax.plot(xs, np.cos(xs) + 2 * xs, color=INK, linewidth=2.6, label="f'(x) = cos x + 2x")
    ax.plot(1, s1 + s2, "o", color=INK, markersize=7, zorder=5)
    ax.plot([1, 1], [s2, s2 + s1], color=BLUE, linewidth=4, zorder=4)
    ax.text(2.0, -3.3, f"f'(1) = {s1:.4f} + {s2:g} = {s1 + s2:.4f}", ha="right", fontsize=10.5,
            weight="bold")
    axes_cross(ax)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Derivative curves stack, point by point")

    panel(fig.add_subplot(gs[1, 2]), "With the scalar rule: term by term", [
        "`3x² + 5x         → 6x + 5`",
        "`x³ − 2x² + 7     → 3x² − 4x`",
        "`4x² − 3x + eˣ    → 8x − 3 + eˣ`",
        "`sin x + eˣ − ln x → cos x + eˣ − 1/x`",
        "",
        "Any polynomial, any sum of known",
        "functions: differentiate each piece.",
    ], spacing=0.095)
    save(fig, name="14_sum_rule.png")


# ================================================================ notebook 15
def sheet_15_product_rule():
    x = sp.symbols("x")
    gg, hh = x ** 2, sp.sin(x)
    assert sp.expand(sp.diff(gg * hh, x)) == x ** 2 * sp.cos(x) + 2 * x * sp.sin(x)
    assert sp.diff(gg, x) * sp.diff(hh, x) == 2 * x * sp.cos(x)
    x0 = 1.3
    gp, hp = 2 * x0, cos(x0)
    g0, h0 = x0 ** 2, sin(x0)
    rule = gp * h0 + g0 * hp
    assert close(rule, 2.957324, 5e-7)

    fig = new_figure("Product rule: (gh)' = g'h + gh', not g'h'",
                     "Grow both sides of a rectangle of area g·h: the area gains two strips "
                     "(Δg·h and g·Δh) plus a tiny corner Δg·Δh that vanishes in the limit.")
    gs = grid(fig)

    ax = fig.add_subplot(gs[0, 0])
    G, H, dG, dH = 3.0, 4.0, 0.8, 0.6
    ax.add_patch(Rectangle((0, 0), H, G, facecolor=BLUE_LIGHT, edgecolor=INK2, linewidth=1.4))
    ax.add_patch(Rectangle((H, 0), dH, G, facecolor=AQUA, alpha=0.6, edgecolor=INK2, linewidth=1.2))
    ax.add_patch(Rectangle((0, G), H, dG, facecolor=ORANGE, alpha=0.6, edgecolor=INK2, linewidth=1.2))
    ax.add_patch(Rectangle((H, G), dH, dG, facecolor=INK2, alpha=0.8, edgecolor=INK2, linewidth=1.2))
    ax.text(H / 2, G / 2, "g·h", ha="center", va="center", fontsize=18, weight="bold")
    ax.text(H + dH / 2, G / 2, "g·Δh", ha="center", va="center", fontsize=11, rotation=90)
    ax.text(H / 2, G + dG / 2, "Δg·h", ha="center", va="center", fontsize=11)
    ax.annotate("Δg·Δh\n(tiny)", (H + dH / 2, G + dG / 2), xytext=(H + dH + 0.25, G + dG + 0.4),
                fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlim(-0.3, H + dH + 1.4)
    ax.set_ylim(-0.3, G + dG + 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("front wall h(t)")
    ax.set_ylabel("side wall g(t)")
    ax.set_xticks([0, H, H + dH], ["0", "h", "h+Δh"])
    ax.set_yticks([0, G, G + dG], ["0", "g", "g+Δg"])
    ax.set_title("The house: two strips + a corner")

    ax = fig.add_subplot(gs[0, 1])
    dts = np.logspace(-4, 0, 60)

    def g_(u):
        return u ** 2

    def h_(u):
        return np.sin(u)

    dg = g_(x0 + dts) - g_(x0)
    dh = h_(x0 + dts) - h_(x0)
    ax.plot(dts, dg * h0 / dts, color=ORANGE, linewidth=2.2, label="Δg·h / Δx  → g'h")
    ax.plot(dts, g0 * dh / dts, color=AQUA, linewidth=2.2, label="g·Δh / Δx  → gh'")
    ax.plot(dts, dg * dh / dts, color=INK2, linewidth=2.2, label="Δg·Δh / Δx  → 0")
    ax.axhline(gp * h0, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.axhline(g0 * hp, color=AQUA, linestyle=":", linewidth=1.3)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Δx (log scale)  ← shrinking")
    ax.set_ylabel("contribution to Δ(gh)/Δx")
    ax.grid(True)
    ax.legend(loc="center right", fontsize=9.5)
    ax.set_title("g = x², h = sin x at x = 1.3")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$\Delta f = \Delta g\,h + g\,\Delta h + \Delta g\,\Delta h$",
        "Divide by Δt:",
        r"$\frac{\Delta f}{\Delta t} = \frac{\Delta g}{\Delta t}h + g\frac{\Delta h}{\Delta t} + \frac{\Delta g\,\Delta h}{\Delta t}$",
        "The last term → 0 (two vanishing changes).",
        r"$(gh)' = g'h + gh'$",
        "",
        "Hammer mnemonic: hit one factor at a",
        "time, leave the other, add the results.",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-3, 3, 400)
    ax.plot(xs, xs ** 2 * np.cos(xs) + 2 * xs * np.sin(xs), color=BLUE, linewidth=2.4,
            label="(gh)' = 2x sin x + x² cos x")
    ax.plot(xs, 2 * xs * np.cos(xs), color=ORANGE, linewidth=2.2, linestyle="--",
            label="g'h' = 2x cos x  (wrong)")
    axes_cross(ax)
    ax.set_xlabel("x")
    ax.set_ylim(-8, 6)
    ax.grid(True)
    ax.legend(loc="lower center", fontsize=9.5)
    ax.set_title("For x² · sin x the shortcut fails")

    ax = fig.add_subplot(gs[1, 1])
    names = ["g'h", "gh'", "g'h + gh'", "g'h'"]
    vals = [gp * h0, g0 * hp, rule, gp * hp]
    colors = [ORANGE, AQUA, BLUE, MUTED]
    pos = np.arange(4)
    ax.bar(pos[:2], vals[:2], width=0.6, color=colors[:2], zorder=2)
    ax.bar(pos[2], vals[0], width=0.6, color=ORANGE, zorder=2)
    ax.bar(pos[2], vals[1], bottom=vals[0], width=0.6, color=AQUA, zorder=2)
    ax.bar(pos[3], vals[3], width=0.6, color=MUTED, zorder=2)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.06, f"{v:.6f}" if p == 2 else f"{v:.4f}", ha="center", fontsize=10)
    ax.text(2, rule + 0.35, "numeric (gh)' = 2.957324", ha="center", fontsize=9.5, color=BLUE)
    ax.set_xticks(pos, names)
    ax.set_ylim(0, 3.8)
    ax.grid(axis="y")
    ax.set_title("At x = 1.3: the two strips sum to the slope")

    panel(fig.add_subplot(gs[1, 2]), "Use it", [
        "`x² sin x → 2x sin x + x² cos x`",
        "`x eˣ     → eˣ + x eˣ`",
        "`x ln x   → ln x + 1`",
        "`(x²+1)(x³−2x) → 5x⁴ − 3x² − 2`",
        "",
        "Three factors, one at a time:",
        "(uvw)' = u'vw + uv'w + uvw'",
    ], spacing=0.095)
    save(fig, name="15_product_rule.png")


# ================================================================ notebook 16
def sheet_16_chain_rule():
    t, hh = sp.symbols("t hh")
    h_of_t = 1000 * t
    T_of_h = 30 - sp.Rational(6, 1000) * hh
    dT_dt = sp.diff(T_of_h, hh) * sp.diff(h_of_t, t)
    assert dT_dt == -6 and sp.diff(T_of_h.subs(hh, h_of_t), t) == -6
    x = sp.symbols("x")
    assert sp.simplify(sp.diff(sp.sin(x ** 2), x) - 2 * x * sp.cos(x ** 2)) == 0

    def numder(fn, u, eps=1e-6):
        return (fn(u + eps) - fn(u - eps)) / (2 * eps)

    checks = {u: numder(lambda v: np.sin(v ** 2), u) for u in [0.5, 1.0, 1.5]}
    assert [round(v, 6) for v in checks.values()] == [0.968912, 1.080605, -1.884521]

    fig = new_figure("Chain rule: rates of change multiply",
                     "dT/dt = dT/dh · dh/dt, or in Lagrange form (g∘h)'(t) = g'(h(t)) · h'(t): "
                     "evaluate the outer derivative at the inner value.")
    gs = grid(fig, wspace=0.34)

    tt = np.linspace(0, 3, 100)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tt, 1000 * tt, color=BLUE, linewidth=2.4, label="h(t) = 1000t")
    ax.plot([1, 2, 2], [1000, 1000, 2000], color=INK2, linewidth=1.4, linestyle=":")
    ax.text(2.06, 1500, "dh/dt = 1000 m\nper unit time", fontsize=10, va="center", color=BLUE)
    ax.set_xlabel("time t")
    ax.set_ylabel("height h (m)")
    ax.set_xlim(0, 3)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Climbing: height vs time")

    hs = np.linspace(0, 3000, 100)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hs, 30 - 0.006 * hs, color=AQUA, linewidth=2.4, label="T(h) = 30 − 0.006h")
    ax.plot([1000, 2000, 2000], [24, 24, 18], color=INK2, linewidth=1.4, linestyle=":")
    ax.text(2060, 21.5, "dT/dh = −6 °C\nper 1000 m", fontsize=10, va="center", color=AQUA)
    ax.set_xlabel("height h (m)")
    ax.set_ylabel("temperature T (°C)")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("Mountain: temperature vs height")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(tt, 30 - 6 * tt, color=ORANGE, linewidth=2.4, label="T(h(t)) = 30 − 6t")
    ax.plot([1, 2, 2], [24, 24, 18], color=INK2, linewidth=1.4, linestyle=":")
    ax.text(2.06, 21.5, "dT/dt = (−0.006)(1000)\n= −6 °C per unit time", fontsize=10, va="center",
            color=ORANGE)
    ax.set_xlabel("time t")
    ax.set_ylabel("temperature T (°C)")
    ax.set_xlim(0, 3.6)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("Composed: temperature vs time")

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2.2, 2.2, 500)
    ax.plot(xs, np.sin(xs ** 2), color=INK, linewidth=2.4, label="f(x) = sin(x²)")
    ax.plot(xs, 2 * xs * np.cos(xs ** 2), color=ORANGE, linewidth=2.2, linestyle="--",
            label="f'(x) = cos(x²) · 2x")
    label_offsets = {0.5: (-8, 10), 1.0: (10, 4), 1.5: (10, -6)}
    for u, v in checks.items():
        ax.plot(u, v, "o", color=BLUE, markersize=7, zorder=5)
        ax.annotate(f"{v:.6f}", (u, v), textcoords="offset points", xytext=label_offsets[u],
                    ha="right" if u == 0.5 else "left", fontsize=9.5, color=BLUE, weight="bold")
    ax.plot([], [], "o", color=BLUE, label="numeric derivative")
    axes_cross(ax)
    ax.set_ylim(-5, 5.2)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("sin(x²): outer sin, inner x²")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xs, 2 * xs * np.cos(xs ** 2), color=ORANGE, linewidth=2.4, label="right: cos(x²) · 2x")
    ax.plot(xs, 2 * xs * np.cos(xs), color=MUTED, linewidth=2.0, linestyle="--",
            label="wrong: cos(x) · 2x")
    ax.fill_between(xs, 2 * xs * np.cos(xs ** 2), 2 * xs * np.cos(xs), color=MUTED, alpha=0.12,
                    linewidth=0)
    axes_cross(ax)
    ax.set_ylim(-5, 5.2)
    ax.set_xlabel("x")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Mind the input: g' at h(x), not at x")

    panel(fig.add_subplot(gs[1, 2]), "The rule, and longer chains", [
        r"$\frac{dg}{dt} = \frac{dg}{dh}\cdot\frac{dh}{dt}$",
        r"$(f\circ g\circ h)' = f'(g(h))\cdot g'(h)\cdot h'$",
        "",
        "`e^(3x)      → 3e^(3x)`",
        "`(x²+1)⁵     → 10x(x²+1)⁴`",
        "`ln(cos x)   → −tan x`",
        "`sin(e^(x²)) → 2x e^(x²) cos(e^(x²))`",
        "",
        "Backpropagation is this rule, repeated.",
    ], spacing=0.088)
    save(fig, name="16_chain_rule.png")


if __name__ == "__main__":
    sheet_00_instantaneous_rate()
    sheet_01_slopes_maxima_minima()
    sheet_02_notation()
    sheet_03_lines()
    sheet_04_quadratics()
    sheet_05_polynomials()
    sheet_06_other_powers()
    sheet_07_inverse()
    sheet_08_trig()
    sheet_09_euler_number()
    sheet_10_derivative_of_exp()
    sheet_11_derivative_of_log()
    sheet_12_existence()
    sheet_13_scalar_multiple()
    sheet_14_sum_rule()
    sheet_15_product_rule()
    sheet_16_chain_rule()
