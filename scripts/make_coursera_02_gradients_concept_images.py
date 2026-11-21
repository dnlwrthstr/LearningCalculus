#!/usr/bin/env python3
"""Concept sheets for coursera/02_gradients (tangent planes, partial derivatives, gradients, optimization).

One page per notebook that summarizes its central idea. Every number is
computed here from the same functions, points and data the notebook uses,
and checked against the values the notebook prints.
Images are written to coursera/02_gradients/img/.

Run from anywhere:  python scripts/make_coursera_02_gradients_concept_images.py
"""
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "02_gradients"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
X_SYM, Y_SYM = sp.symbols("x y")


# ================================================================ helpers
def bowl(x, y):
    """f(x, y) = x² + y², the running example of notebooks 00, 01, 03 and 04."""
    return x ** 2 + y ** 2


def partial_x(f, x, y, h=1e-6):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def partial_y(f, x, y, h=1e-6):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def pretty(expr):
    """SymPy expression as compact text: x**2 → x², 3*x → 3x."""
    text = sp.sstr(expr).replace("**2", "²").replace("**3", "³")
    return text.replace("*", "·")


def tangent(ax, t, x0, z0, slope, color, label, half_width):
    """Dashed tangent line through (x0, z0) drawn over x0 ± half_width."""
    seg = t[np.abs(t - x0) <= half_width]
    ax.plot(seg, z0 + slope * (seg - x0), "--", color=color, linewidth=2, label=label)


def style_3d(ax, xlabel="x", ylabel="y", zlabel="z"):
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_zlabel(zlabel, labelpad=2)
    ax.tick_params(labelsize=8, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.97, 0.97, 0.96, 1.0))


def plain_axes(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


# ================================================================ notebook 00
def sheet_00_tangent_planes():
    x0, y0 = 2.0, 4.0
    f_sym = X_SYM ** 2 + Y_SYM ** 2
    fx0 = sp.diff(f_sym, X_SYM).subs({X_SYM: 2, Y_SYM: 4})
    fy0 = sp.diff(f_sym, Y_SYM).subs({X_SYM: 2, Y_SYM: 4})
    f0 = f_sym.subs({X_SYM: 2, Y_SYM: 4})
    assert (f0, fx0, fy0) == (20, 4, 8)
    plane_sym = sp.expand(f0 + fx0 * (X_SYM - 2) + fy0 * (Y_SYM - 4))
    assert plane_sym == 4 * X_SYM + 8 * Y_SYM - 20
    assert round(partial_x(bowl, x0, y0), 5) == 4 and round(partial_y(bowl, x0, y0), 5) == 8
    f0, fx0, fy0 = float(f0), float(fx0), float(fy0)

    def plane(x, y):
        return f0 + fx0 * (x - x0) + fy0 * (y - y0)

    fig = new_figure("Tangent plane: two tangent lines that cross, and the plane they span",
                     "z = f(x₀, y₀) + $f_x$(x₀, y₀)(x − x₀) + $f_y$(x₀, y₀)(y − y₀).   "
                     "For f = x² + y² at (2, 4):  z = 20 + 4(x − 2) + 8(y − 4).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(-1, 4, 400)
    ax.plot(t, t ** 2, color=BLUE, linewidth=2.2, label="f(x) = x²")
    ax.plot(t, 4 + 4 * (t - 2), "--", color=ORANGE, linewidth=2, label="tangent, slope f′(2) = 4")
    ax.plot(2, 4, "o", color=INK, zorder=3)
    ax.annotate("(2, 4)", (2, 4), textcoords="offset points", xytext=(10, -14))
    ax.set_ylim(-2, 16)
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "1D recap: a slope gives a tangent line", "x", "f(x)")

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    g = np.linspace(-4, 5, 50)
    X, Y = np.meshgrid(g, g)
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.55, linewidth=0, rstride=2, cstride=2)
    p = np.linspace(-1.5, 1.5, 12)
    PX, PY = np.meshgrid(x0 + p, y0 + p)
    ax.plot_surface(PX, PY, plane(PX, PY), color=ORANGE, alpha=0.6, linewidth=0)
    s = np.linspace(-2, 2, 30)
    ax.plot(x0 + s, np.full_like(s, y0), f0 + fx0 * s, color=INK, linewidth=2)
    ax.plot(np.full_like(s, x0), y0 + s, f0 + fy0 * s, color=AQUA, linewidth=2)
    ax.scatter([x0], [y0], [f0], color=INK, s=40, depthshade=False)
    ax.view_init(elev=24, azim=-125)
    style_3d(ax)
    ax.set_title("The plane touches the bowl at (2, 4, 20)", pad=0)

    panel(fig.add_subplot(gs[0, 2]), "Slice, differentiate, combine", [
        "Fix y = 4:   f(x, 4) = x² + 16",
        r"$\partial f/\partial x = 2x \;\rightarrow\; 4$ at x = 2",
        "Fix x = 2:   f(2, y) = 4 + y²",
        r"$\partial f/\partial y = 2y \;\rightarrow\; 8$ at y = 4",
        "",
        "Both tangent lines pass through (2, 4, 20)",
        "in different directions, so they pin down",
        "exactly one plane:",
        "`z = 20 + 4(x − 2) + 8(y − 4) = 4x + 8y − 20",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-4, 6, 400)
    ax.plot(t, t ** 2 + y0 ** 2, color=BLUE, linewidth=2.2, label="slice y = 4:  x² + 16  (vs x)")
    tangent(ax, t, x0, f0, fx0, BLUE, "tangent, slope $f_x$ = 4", 3)
    ax.plot(t, x0 ** 2 + t ** 2, color=AQUA, linewidth=2.2, label="slice x = 2:  4 + y²  (vs y)")
    tangent(ax, t, y0, f0, fy0, AQUA, "tangent, slope $f_y$ = 8", 1.8)
    ax.plot([x0, y0], [f0, f0], "o", color=INK, zorder=3)
    ax.set_ylim(0, 45)
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "The two slices through (2, 4)", "x (blue slice)  or  y (green slice)", "z")

    ax = fig.add_subplot(gs[1, 1])
    dist = np.logspace(-3, 0.5, 60)
    direction = np.array([1.0, 1.0]) / np.sqrt(2)
    gap = [bowl(x0 + d * direction[0], y0 + d * direction[1])
           - plane(x0 + d * direction[0], y0 + d * direction[1]) for d in dist]
    assert np.allclose(gap, dist ** 2)
    ax.loglog(dist, gap, color=ORANGE, linewidth=2.4, label="surface − plane")
    ax.loglog(dist, dist, ":", color=MUTED, linewidth=1.5, label="distance (for scale)")
    for d in (0.01, 0.1, 1.0):
        ax.plot(d, d ** 2, "o", color=INK, markersize=5, zorder=3)
        ax.annotate(f"{d:g} away: gap {d ** 2:g}", (d, d ** 2), textcoords="offset points",
                    xytext=(8, -12), fontsize=9.5)
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "Near the point the plane is almost the surface",
               "distance from (2, 4) in the floor", "height gap")
    ax.grid(True, which="major")

    panel(fig.add_subplot(gs[1, 2]), "Checked two ways", [
        "SymPy:",
        "`∂f/∂x = 2*x,  ∂f/∂y = 2*y",
        "`tangent plane: z = 4*x + 8*y - 20",
        "",
        "Central differences, h = 10⁻⁶:",
        "`f_x ≈ 4.00000,  f_y ≈ 8.00000",
        "",
        "The gap f − plane = (x − 2)² + (y − 4)²",
        "shrinks like distance²: a 10× closer look",
        "makes the plane 100× more accurate.",
    ], spacing=0.085)
    save(fig, name="00_tangent_planes.png")


# ================================================================ notebook 01
def sheet_01_partial_derivatives():
    x0, y0 = 2.0, 4.0
    f_sym = X_SYM ** 2 + Y_SYM ** 2
    assert sp.diff(f_sym, X_SYM) == 2 * X_SYM and sp.diff(f_sym, Y_SYM) == 2 * Y_SYM
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    g = x1 ** 2 + 3 * x1 * x2 + x3 ** 2
    g_partials = [sp.diff(g, v) for v in (x1, x2, x3)]
    assert g_partials == [2 * x1 + 3 * x2, 3 * x1, 2 * x3]
    checks = [(2.0, 4.0), (1.0, 3.0), (-2.0, 0.5)]
    numeric = [(partial_x(bowl, a, b), partial_y(bowl, a, b)) for a, b in checks]
    assert all(abs(nx - 2 * a) < 1e-5 and abs(ny - 2 * b) < 1e-5
               for (a, b), (nx, ny) in zip(checks, numeric))
    f0 = bowl(x0, y0)

    fig = new_figure("A partial derivative is the slope of one slice",
                     "∂f/∂x: hold every other variable constant, then differentiate as usual.   "
                     "f = x² + y²  →  $f_x$ = 2x,  $f_y$ = 2y.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    gv = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(gv, gv)
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.45, linewidth=0, rstride=2, cstride=2)
    t = np.linspace(-4, 4, 100)
    ax.plot(t, np.full_like(t, y0), bowl(t, y0), color=ORANGE, linewidth=3)
    seg = np.linspace(0, 4, 20)
    ax.plot(seg, np.full_like(seg, y0), f0 + 2 * x0 * (seg - x0), color=INK, linewidth=2)
    ax.scatter([x0], [y0], [f0], color=INK, s=40, depthshade=False)
    ax.view_init(elev=22, azim=-60)
    style_3d(ax)
    ax.set_title("Cut the bowl with the plane y = 4", pad=0)

    for col, (label, curve, at, slope, varname, color) in enumerate([
            ("Fix y = 4  →  f(x, 4) = x² + 16", lambda s: s ** 2 + y0 ** 2, x0, 2 * x0, "x", ORANGE),
            ("Fix x = 2  →  f(2, y) = 4 + y²", lambda s: x0 ** 2 + s ** 2, y0, 2 * y0, "y", AQUA)], 1):
        ax = fig.add_subplot(gs[0, col])
        t = np.linspace(-4, 6, 300)
        ax.plot(t, curve(t), color=color, linewidth=2.4, label=label.split("→")[1].strip())
        tangent(ax, t, at, f0, slope, INK, f"tangent, slope {slope:.0f}", 2.5)
        ax.plot(at, f0, "o", color=INK, zorder=3)
        ax.set_ylim(0, 50)
        ax.legend(loc="upper left", fontsize=9.5)
        sym = "$f_x$" if varname == "x" else "$f_y$"
        plain_axes(ax, f"{label.split('  →')[0]}:  {sym}(2, 4) = {slope:.0f}", varname, "z")

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-1, 4, 300)
    for y_fixed, color in [(0, BLUE_LIGHT), (2, BLUE), (4, ORANGE)]:
        z_at = bowl(x0, y_fixed)
        slope = partial_x(bowl, x0, y_fixed)
        assert abs(slope - 4) < 1e-5
        ax.plot(t, bowl(t, y_fixed), color=color, linewidth=2.2, label=f"y = {y_fixed}:  x² + {y_fixed ** 2}")
        tangent(ax, t, x0, z_at, slope, color, None, 1.2)
        ax.plot(x0, z_at, "o", color=INK, markersize=5, zorder=3)
    ax.legend(loc="upper left", fontsize=9.5, title="slice at", title_fontsize=9.5)
    ax.set_ylim(-2, 35)
    plain_axes(ax, "Any fixed y: slope at x = 2 is still 4", "x", "z")

    panel(fig.add_subplot(gs[1, 1]), "The two-step recipe", [
        "1. Treat all other variables as constants.",
        "2. Differentiate with the ordinary rules.",
        "",
        r"$\frac{\partial}{\partial x}(x^2 + y^2) = 2x + 0$   (y² is a constant)",
        r"$\frac{\partial}{\partial y}(x^2 + y^2) = 0 + 2y$   (x² is a constant)",
        "",
        r"Notation: $\partial f/\partial x \equiv f_x$,  $\partial f/\partial y \equiv f_y$",
        "$f_x$ = 2x has no y in it, which is why every",
        "slice on the left has the same slope at x = 2.",
    ], spacing=0.09)

    lines = ["Central differences (h = 10⁻⁶) vs exact:",
             "`   point        f_x num   f_y num"]
    for (a, b), (nx, ny) in zip(checks, numeric):
        lines.append(f"`({a:>4}, {b:>4})   {nx:8.5f}  {ny:8.5f}")
    lines += ["",
              "n variables → n partial derivatives:",
              r"$g = x_1^2 + 3x_1x_2 + x_3^2$",
              r"$g_{x_1} = 2x_1 + 3x_2,\;\; g_{x_2} = 3x_1,\;\; g_{x_3} = 2x_3$"]
    panel(fig.add_subplot(gs[1, 2]), "Checked, and it scales", lines, spacing=0.088)
    save(fig, name="01_partial_derivatives.png")


# ================================================================ notebook 02
def sheet_02_box_the_constant():
    def f(x, y):
        return 3 * x ** 2 * y ** 3

    def fx_exact(x, y):
        return 6 * x * y ** 3

    def fy_exact(x, y):
        return 9 * x ** 2 * y ** 2

    f_sym = 3 * X_SYM ** 2 * Y_SYM ** 3
    assert sp.diff(f_sym, X_SYM) == 6 * X_SYM * Y_SYM ** 3
    assert sp.diff(f_sym, Y_SYM) == 9 * X_SYM ** 2 * Y_SYM ** 2
    checks = [(1.0, 1.0), (2.0, -1.0), (0.5, 2.0)]
    expected = [(6, 9), (-12, 36), (24, 9)]
    for (a, b), (ex, ey) in zip(checks, expected):
        assert abs(partial_x(f, a, b) - ex) < 1e-4 and abs(partial_y(f, a, b) - ey) < 1e-4
        assert (fx_exact(a, b), fy_exact(a, b)) == (ex, ey)
    practice = [5 * X_SYM ** 3 * Y_SYM ** 2, X_SYM ** 2 * Y_SYM + X_SYM * Y_SYM ** 2,
                sp.sin(X_SYM) * Y_SYM ** 2]
    practice_d = [(sp.diff(e, X_SYM), sp.diff(e, Y_SYM)) for e in practice]
    assert practice_d[0] == (15 * X_SYM ** 2 * Y_SYM ** 2, 10 * X_SYM ** 3 * Y_SYM)
    x0, y0 = 1.0, 1.0
    f0, sx, sy = f(x0, y0), fx_exact(x0, y0), fy_exact(x0, y0)

    fig = new_figure("Box the constant, differentiate the rest",
                     "f(x, y) = 3x²y³:   ∂f/∂x = 3·2x·[y³] = 6xy³,   ∂f/∂y = 3·[x²]·3y² = 9x²y².   "
                     "At (1, 1): f = 3, $f_x$ = 6, $f_y$ = 9.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    gx = np.linspace(-2, 2, 300)
    gy = np.linspace(-1.6, 1.6, 300)
    X, Y = np.meshgrid(gx, gy)
    Z = f(X, Y)
    levels = np.linspace(-12, 12, 13)
    cf = ax.contourf(X, Y, np.clip(Z, -12, 12), levels=levels, cmap="RdBu_r", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("f(x, y)  (clipped at ±12)")
    cbar.outline.set_visible(False)
    ax.axhline(y0, color=ORANGE, linewidth=2.2)
    ax.axvline(x0, color=AQUA, linewidth=2.2)
    ax.plot(x0, y0, "o", color=INK, zorder=3)
    ax.text(-1.9, y0 + 0.08, "slice y = 1", color=INK, fontsize=10, weight="bold")
    ax.text(x0 + 0.06, -1.5, "slice x = 1", color=INK, fontsize=10, weight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("z = 3x²y³ from above, two cuts at (1, 1)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(-1.5, 1.6, 300)
    ax.plot(t, f(t, y0), color=ORANGE, linewidth=2.4, label="slice y = 1:  3x²")
    tangent(ax, t, x0, f0, sx, ORANGE, f"tangent, $f_x$ = {sx:.0f}", 0.6)
    ax.plot(t, f(x0, t), color=AQUA, linewidth=2.4, label="slice x = 1:  3y³")
    tangent(ax, t, y0, f0, sy, AQUA, f"tangent, $f_y$ = {sy:.0f}", 0.45)
    ax.plot(x0, f0, "o", color=INK, zorder=3)
    ax.set_ylim(-6, 10)
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "Two slices through (1, 1), two slopes", "x (orange)  or  y (green)", "z")

    panel(fig.add_subplot(gs[0, 2]), "The recipe, with boxes", [
        "Differentiating in x?  Then y is a constant:",
        r"$f = 3\cdot x^2\cdot[\,y^3\,]$  →  $f_x = 3\cdot 2x\cdot y^3 = 6xy^3$",
        "",
        "Differentiating in y?  Then x is a constant:",
        r"$f = 3\cdot[\,x^2\,]\cdot y^3$  →  $f_y = 3\cdot x^2\cdot 3y^2 = 9x^2y^2$",
        "",
        "Boxed factors [ ] ride along unchanged:",
        r"$\frac{d}{dx}[c\cdot u(x)] = c\cdot u'(x)$",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-2.4, 2.4, 300)
    for (a, b), color in zip(checks, [ORANGE, BLUE, AQUA]):
        c = 3 * b ** 3
        ax.plot(t, f(t, b), color=color, linewidth=2.2,
                label=f"y = {b:g}:  {c:g}x²,  slope {fx_exact(a, b):g} at x = {a:g}")
        tangent(ax, t, a, f(a, b), fx_exact(a, b), color, None, 0.7)
        ax.plot(a, f(a, b), "o", color=INK, markersize=5, zorder=3)
    ax.set_ylim(-20, 30)
    ax.set_xlim(-2.4, 2.4)
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "The frozen y³ scales (and flips) the slice", "x", "z")

    ax = fig.add_subplot(gs[1, 1])
    labels = [f"({a:g}, {b:g})" for a, b in checks]
    pos = np.arange(len(checks))
    vals_x = [partial_x(f, a, b) for a, b in checks]
    vals_y = [partial_y(f, a, b) for a, b in checks]
    ax.bar(pos - 0.19, vals_x, width=0.36, color=ORANGE, label="$f_x$ = 6xy³", zorder=2)
    ax.bar(pos + 0.19, vals_y, width=0.36, color=AQUA, label="$f_y$ = 9x²y²", zorder=2)
    for p, vx, vy in zip(pos, vals_x, vals_y):
        ax.text(p - 0.19, vx + (1.2 if vx >= 0 else -3.2), f"{vx:.0f}", ha="center", fontsize=10)
        ax.text(p + 0.19, vy + 1.2, f"{vy:.0f}", ha="center", fontsize=10)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(pos, labels)
    ax.set_ylim(-20, 44)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Finite differences agree with the formulas")
    ax.set_xlabel("point (x, y)")
    ax.set_ylabel("partial derivative")

    lines = []
    for expr, (dx, dy) in zip(practice, practice_d):
        lines += [f"f = {pretty(expr)}", f"`  f_x = {pretty(dx)}", f"`  f_y = {pretty(dy)}", ""]
    panel(fig.add_subplot(gs[1, 2]), "Same recipe, more functions", lines[:-1], spacing=0.085)
    save(fig, name="02_partial_derivatives_part_2.png")


# ================================================================ notebook 03
def sheet_03_gradients():
    grad_sym = [sp.diff(X_SYM ** 2 + Y_SYM ** 2, v) for v in (X_SYM, Y_SYM)]
    assert grad_sym == [2 * X_SYM, 2 * Y_SYM]
    p0 = np.array([2.0, 3.0])
    grad0 = np.array([float(gc.subs({X_SYM: 2, Y_SYM: 3})) for gc in grad_sym])
    assert grad0.tolist() == [4.0, 6.0]
    num = np.array([partial_x(bowl, *p0), partial_y(bowl, *p0)])
    assert np.allclose(num, grad0, atol=1e-5)
    vars4 = sp.symbols("x1 x2 x3 x4")
    a1, a2, a3, a4 = vars4
    g4 = a1 ** 2 + 3 * a2 * a3 + sp.sin(a4)
    grad4 = [sp.diff(g4, v) for v in vars4]
    assert grad4 == [2 * a1, 3 * a3, 3 * a2, sp.cos(a4)]
    norm = np.linalg.norm(grad0)

    fig = new_figure("The gradient: all partial derivatives in one vector",
                     "∇f = (∂f/∂x, ∂f/∂y).  For f = x² + y²:  ∇f = (2x, 2y),  ∇f(2, 3) = (4, 6), "
                     "pointing uphill, the way f grows fastest.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(-4, 5, 300)
    X, Y = np.meshgrid(g, g)
    ax.contour(X, Y, bowl(X, Y), levels=[1, 4, 9, 13, 16, 25], colors=BLUE_LIGHT, linewidths=1.3)
    q = np.linspace(-3, 3, 7)
    QX, QY = np.meshgrid(q, q)
    ax.quiver(QX, QY, 2 * QX, 2 * QY, color=BLUE, alpha=0.85, angles="xy", scale_units="xy",
              scale=10, width=0.006)
    ax.quiver(*p0, *grad0, color=ORANGE, angles="xy", scale_units="xy", scale=4, width=0.012,
              zorder=4)
    ax.plot(*p0, "o", color=INK, zorder=5)
    ax.text(-3.4, 4.5, "∇f(2, 3) = (4, 6)", fontsize=10.5, color=ORANGE, weight="bold",
            va="center")
    ax.set_xlim(-3.6, 4.2)
    ax.set_ylim(-3.6, 5.0)
    ax.set_aspect("equal")
    plain_axes(ax, "Arrows point away from the minimum", "x", "y")

    ax = fig.add_subplot(gs[0, 1])
    theta = np.linspace(0, 360, 721)
    rad = np.radians(theta)
    slope = grad0[0] * np.cos(rad) + grad0[1] * np.sin(rad)
    best = np.degrees(np.arctan2(grad0[1], grad0[0]))
    ax.plot(theta, slope, color=BLUE, linewidth=2.4)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(best, norm, "o", color=ORANGE, zorder=3)
    ax.annotate(f"along ∇f ({best:.1f}°):\nslope |∇f| = √52 = {norm:.2f}", (best, norm),
                textcoords="offset points", xytext=(12, -32), fontsize=10)
    ax.plot(best + 180, -norm, "o", color=AQUA, zorder=3)
    ax.annotate(f"along −∇f: {-norm:.2f}\n(steepest descent)", (best + 180, -norm),
                textcoords="offset points", xytext=(-150, 6), fontsize=10)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlim(0, 360)
    ax.set_ylim(-9.5, 9.5)
    plain_axes(ax, "Slope of f at (2, 3) in each direction", "direction angle θ (degrees)",
               r"slope  $\nabla f\cdot(\cos\theta, \sin\theta)$")

    panel(fig.add_subplot(gs[0, 2]), "Definition and notation", [
        r"$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2},"
        r" \dots, \frac{\partial f}{\partial x_n}\right)$",
        "",
        "One entry per variable; at a point it is",
        "a plain vector of numbers.",
        "",
        r"$f = x^2 + y^2:\;\; \nabla f = (2x, 2y)$",
        r"$\nabla f(2, 3) = (2\cdot 2,\; 2\cdot 3) = (4, 6)$",
        "",
        "Central differences at (2, 3):",
        f"`∇f ≈ ({num[0]:.4f}, {num[1]:.4f})",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-1, 5, 300)
    f0 = bowl(*p0)
    ax.plot(t, bowl(t, p0[1]), color=ORANGE, linewidth=2.2, label="slice y = 3:  x² + 9  (vs x)")
    tangent(ax, t, p0[0], f0, grad0[0], ORANGE, "slope 4 = 1st entry of ∇f", 1.8)
    ax.plot(t, bowl(p0[0], t), color=AQUA, linewidth=2.2, label="slice x = 2:  4 + y²  (vs y)")
    tangent(ax, t, p0[1], f0, grad0[1], AQUA, "slope 6 = 2nd entry of ∇f", 1.3)
    ax.plot(p0, [f0, f0], "o", color=INK, zorder=3)
    ax.set_ylim(0, 32)
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "The entries are the tangent-plane slopes", "x (orange)  or  y (green)", "z")

    ax = fig.add_subplot(gs[1, 1])
    step = np.linspace(0, 1, 200)
    u = grad0 / norm
    perp = np.array([-u[1], u[0]])
    for vec, color, label in [(u, ORANGE, "along ∇f"), (np.array([1.0, 0.0]), BLUE, "along +x"),
                              (perp, MUTED, "along the contour"), (-u, AQUA, "along −∇f")]:
        ax.plot(step, bowl(p0[0] + step * vec[0], p0[1] + step * vec[1]), color=color,
                linewidth=2.2, label=label)
    ax.axhline(f0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_ylim(5, 22.5)
    plain_axes(ax, "Walk 1 unit from (2, 3): ∇f climbs fastest", "distance walked", "f")

    panel(fig.add_subplot(gs[1, 2]), "Many variables, same idea", [
        r"$g = x_1^2 + 3x_2x_3 + \sin x_4$",
        r"$\nabla g = (2x_1,\; 3x_3,\; 3x_2,\; \cos x_4)$",
        "",
        "17 variables → a vector of 17 entries.",
        "",
        "∇f points uphill, perpendicular to the",
        "contour through the point.",
        "Gradient descent steps the other way: −∇f.",
    ], spacing=0.09)
    save(fig, name="03_gradients.png")


# ================================================================ notebook 04
def sheet_04_gradient_zero():
    grad = [sp.diff(X_SYM ** 2 + Y_SYM ** 2, v) for v in (X_SYM, Y_SYM)]
    assert sp.solve(grad, [X_SYM, Y_SYM], dict=True) == [{X_SYM: 0, Y_SYM: 0}]
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    g = (x1 - 1) ** 2 + (x2 + 2) ** 2 + (x3 - 3) ** 2
    assert sp.solve([sp.diff(g, v) for v in (x1, x2, x3)], [x1, x2, x3],
                    dict=True) == [{x1: 1, x2: -2, x3: 3}]
    kinds = [(X_SYM ** 2 + Y_SYM ** 2, "minimum", lambda x, y: x ** 2 + y ** 2),
             (-X_SYM ** 2 - Y_SYM ** 2, "maximum", lambda x, y: -x ** 2 - y ** 2),
             (X_SYM ** 2 - Y_SYM ** 2, "saddle", lambda x, y: x ** 2 - y ** 2)]
    for expr, _, _ in kinds:
        assert sp.solve([sp.diff(expr, v) for v in (X_SYM, Y_SYM)], [X_SYM, Y_SYM],
                        dict=True) == [{X_SYM: 0, Y_SYM: 0}]

    fig = new_figure("Minima and maxima: where the gradient is the zero vector",
                     "1D: f′(x) = 0.   nD: every partial derivative is 0 at once, ∇f = 0, "
                     "and the tangent plane lies flat. For f = x² + y²: 2x = 0, 2y = 0 → (0, 0).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(-3, 3, 400)
    ax.plot(t, t ** 2, color=BLUE, linewidth=2.2, label="f(x) = x²")
    for a, color in [(-1.5, MUTED), (0.0, ORANGE), (1.0, MUTED)]:
        seg = np.linspace(a - 1, a + 1, 20)
        ax.plot(seg, a ** 2 + 2 * a * (seg - a), "--", color=color, linewidth=2)
        ax.plot(a, a ** 2, "o", color=INK if a else ORANGE, markersize=5, zorder=3)
        ax.annotate(f"slope {2 * a:g}", (a, a ** 2), textcoords="offset points",
                    xytext=(10, -4) if a > 0 else ((-62, -4) if a else (-22, 10)), fontsize=9.5,
                    color=ORANGE if a == 0 else INK2)
    ax.set_ylim(-1.5, 9)
    ax.legend(loc="upper center", fontsize=9.5)
    plain_axes(ax, "1D recap: the minimum has slope 0", "x", "f(x)")

    ax = fig.add_subplot(gs[0, 1], projection="3d")
    gv = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(gv, gv)
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.5, linewidth=0, rstride=2, cstride=2)
    p = np.linspace(-1.2, 1.2, 10)
    for (px, py), color in [((0.0, 0.0), ORANGE), ((1.5, 1.5), MUTED)]:
        PX, PY = np.meshgrid(px + p, py + p)
        ax.plot_surface(PX, PY, bowl(px, py) + 2 * px * (PX - px) + 2 * py * (PY - py),
                        color=color, alpha=0.75, linewidth=0)
        ax.scatter([px], [py], [bowl(px, py)], color=INK, s=30, depthshade=False)
    ax.view_init(elev=18, azim=-55)
    style_3d(ax)
    ax.set_title("Flat plane at the bottom, tilted elsewhere", pad=0)

    panel(fig.add_subplot(gs[0, 2]), "The condition", [
        "Tangent plane parallel to the floor",
        "⇔ both tangent-line slopes are 0:",
        r"$\frac{\partial f}{\partial x} = 0 \;\;\mathrm{and}\;\; \frac{\partial f}{\partial y} = 0"
        r"\;\;\Leftrightarrow\;\; \nabla f = \mathbf{0}$",
        "",
        "12 variables: all 12 partials must be 0,",
        "a system of 12 equations.",
        r"$g = (x_1-1)^2 + (x_2+2)^2 + (x_3-3)^2$",
        r"$\nabla g = 0 \;\Rightarrow\; (1, -2, 3)$",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    g2 = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(g2, g2)
    ax.contour(X, Y, bowl(X, Y), levels=[0.5, 1.5, 3, 5, 7.5, 10.5, 14], colors=BLUE_LIGHT,
               linewidths=1.3)
    q = np.linspace(-3, 3, 9)
    QX, QY = np.meshgrid(q, q)
    ax.quiver(QX, QY, 2 * QX, 2 * QY, color=BLUE, alpha=0.8, angles="xy", scale_units="xy",
              scale=10, width=0.006)
    ax.plot(0, 0, "o", color=ORANGE, markersize=9, zorder=4)
    ax.annotate("∇f = 0 here", (0, 0), textcoords="offset points", xytext=(10, 10), fontsize=10,
                weight="bold", color=ORANGE)
    ax.set_aspect("equal")
    plain_axes(ax, "Arrows ∇f = (2x, 2y) shrink to nothing", "x", "y")

    sub = gs[1, 1:].subgridspec(1, 3, wspace=0.25)
    g3 = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(g3, g3)
    for k, (expr, kind, fn) in enumerate(kinds):
        ax = fig.add_subplot(sub[0, k])
        Z = fn(X, Y)
        ax.contourf(X, Y, Z, levels=14, cmap="RdBu_r", alpha=0.8)
        q = np.linspace(-1.6, 1.6, 5)
        QX, QY = np.meshgrid(q, q)
        dfx = sp.lambdify((X_SYM, Y_SYM), sp.diff(expr, X_SYM))
        dfy = sp.lambdify((X_SYM, Y_SYM), sp.diff(expr, Y_SYM))
        ax.quiver(QX, QY, dfx(QX, QY), dfy(QX, QY), color=INK, alpha=0.7, angles="xy",
                  scale_units="xy", scale=6, width=0.014)
        ax.plot(0, 0, "o", color=ORANGE, markersize=8, markeredgecolor=INK, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel("x")
        if k == 0:
            ax.set_ylabel("y")
        pretty = sp.sstr(expr).replace("**2", "²").replace("-", "−")
        ax.set_title(f"{kind}\nf = {pretty}", fontsize=11.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.68, 0.455, "∇f = 0 at the origin in all three: a candidate, not yet a verdict "
             "(the Hessian decides later)", ha="center", fontsize=11, color=INK2)
    save(fig, name="04_gradients_and_maxima_minima.png")


# ================================================================ notebook 05
def sheet_05_sauna():
    T = 85 - sp.Rational(1, 90) * X_SYM ** 2 * (X_SYM - 6) * Y_SYM ** 2 * (Y_SYM - 6)
    Tx, Ty = sp.diff(T, X_SYM), sp.diff(T, Y_SYM)
    assert sp.simplify(Tx - (-X_SYM * Y_SYM ** 2 * (X_SYM - 4) * (Y_SYM - 6) / 30)) == 0
    assert sp.simplify(Ty - (-X_SYM ** 2 * Y_SYM * (X_SYM - 6) * (Y_SYM - 4) / 30)) == 0
    crit = sp.solve([Tx, Ty], [X_SYM, Y_SYM], dict=True)
    assert {X_SYM: 4, Y_SYM: 4} in crit and {X_SYM: 6, Y_SYM: 6} in crit

    def Tnum(xv, yv):
        return 85 - (1 / 90) * xv ** 2 * (xv - 6) * yv ** 2 * (yv - 6)

    t_min = Tnum(4, 4)
    assert f"{t_min:.3f}" == "73.622"

    fig = new_figure("The coldest spot in a 2D sauna: solve ∇T = 0, then check the candidates",
                     "T(x, y) = 85 − (1/90)·x²(x − 6)·y²(y − 6).   A product is zero only when a factor is; "
                     "the coldest point is (4, 4) with T ≈ 73.6.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(0, 6, 300)
    X, Y = np.meshgrid(g, g)
    cf = ax.contourf(X, Y, Tnum(X, Y), levels=24, cmap="coolwarm")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("temperature T")
    cbar.outline.set_visible(False)
    ax.plot(4, 4, "o", color=AQUA, markersize=11, markeredgecolor=INK, zorder=4)
    ax.annotate(f"coldest (4, 4)\nT = {t_min:.1f}", (4, 4), textcoords="offset points",
                xytext=(-110, -42), fontsize=10, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Temperature over the room")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    # zero sets of each partial derivative (lines where one factor vanishes)
    tx_lines_v, tx_lines_h = [0, 4], [0, 6]
    ty_lines_v, ty_lines_h = [0, 6], [0, 4]
    for xv in tx_lines_v:
        ax.axvline(xv, color=ORANGE, linewidth=2.5, alpha=0.85)
    for yv in tx_lines_h:
        ax.axhline(yv, color=ORANGE, linewidth=2.5, alpha=0.85)
    for xv in ty_lines_v:
        ax.axvline(xv, color=BLUE, linewidth=2.5, linestyle=(0, (4, 3)))
    for yv in ty_lines_h:
        ax.axhline(yv, color=BLUE, linewidth=2.5, linestyle=(0, (4, 3)))
    ax.plot([], [], color=ORANGE, linewidth=2.5, label="∂T/∂x = 0:  x = 0, 4  or  y = 0, 6")
    ax.plot([], [], color=BLUE, linewidth=2.5, linestyle=(0, (4, 3)),
            label="∂T/∂y = 0:  x = 0, 6  or  y = 0, 4")
    ax.plot(4, 4, "o", color=AQUA, markersize=11, markeredgecolor=INK, zorder=4)
    ax.plot(6, 6, "o", color=SURFACE, markersize=9, markeredgecolor=INK, zorder=4)
    ax.annotate("(4, 4)", (4, 4), textcoords="offset points", xytext=(-42, 8), fontsize=10)
    ax.annotate("(6, 6)", (6, 6), textcoords="offset points", xytext=(-46, -18), fontsize=10)
    ax.text(0.15, 2.2, "every point of x = 0\nand y = 0 solves both", fontsize=9.5, color=INK2)
    ax.set_xlim(-0.4, 7.4)
    ax.set_ylim(-0.4, 8.6)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, borderaxespad=0.1, frameon=True, facecolor=SURFACE,
              edgecolor="none", framealpha=1)
    plain_axes(ax, "Critical points: where both sets cross", "x", "y")

    panel(fig.add_subplot(gs[0, 2]), "The partials factor", [
        r"$\frac{\partial T}{\partial x} = -\frac{1}{90}\,x(3x-12)\,y^2(y-6)$",
        r"$\frac{\partial T}{\partial y} = -\frac{1}{90}\,x^2(x-6)\,y(3y-12)$",
        "",
        "SymPy, factored:",
        "`∂T/∂x = -x*y^2*(x - 4)*(y - 6)/30",
        "`∂T/∂y = -x^2*y*(x - 6)*(y - 4)/30",
        "",
        "Each is zero when any one factor is zero;",
        "a critical point needs both zero at once.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(0, 6, 300)
    assert np.allclose(Tnum(t, 4), Tnum(4, t))
    ax.plot(t, Tnum(t, 4), color=ORANGE, linewidth=2.4,
            label="T(x, 4) = T(4, x)  (T is symmetric in x, y)")
    ax.axhline(t_min, color=MUTED, linewidth=1, linestyle=":")
    ax.plot(4, t_min, "o", color=AQUA, markersize=9, markeredgecolor=INK, zorder=4)
    ax.annotate(f"flat at 4:  ∂T/∂x = ∂T/∂y = 0,  T = {t_min:.3f}", (4, t_min),
                textcoords="offset points", xytext=(-150, 60), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(72, 86.5)
    ax.legend(loc="upper center", fontsize=9.5)
    plain_axes(ax, "Through (4, 4) both slices bottom out", "x  (or y, same curve)", "T")

    ax = fig.add_subplot(gs[1, 1])
    cands = ["x = 0\n(any y)", "y = 0\n(any x)", "(4, 4)", "(6, 6)"]
    vals = [Tnum(0, 2.5), Tnum(2.5, 0), t_min, Tnum(6, 6)]
    assert vals[0] == vals[1] == vals[3] == 85
    colors = [BLUE_LIGHT, BLUE_LIGHT, AQUA, BLUE_LIGHT]
    pos = np.arange(len(cands))
    ax.bar(pos, vals, width=0.6, color=colors, zorder=2)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.25, f"{v:.3f}" if v != 85 else "85", ha="center", fontsize=10.5)
    ax.set_xticks(pos, cands)
    ax.set_ylim(70, 87.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Check every candidate: only (4, 4) is cold")
    ax.set_ylabel("temperature T")

    panel(fig.add_subplot(gs[1, 2]), "The general recipe", [
        "1. Compute every partial derivative.",
        r"2. Solve the system $\nabla T = \mathbf{0}$.",
        "3. That gives a list of candidates.",
        "4. Evaluate each and pick the best.",
        "",
        "On x = 0 or y = 0 the product vanishes,",
        "so T = 85, the hot baseline.",
        "",
        "Exact, but solving the system gets hard;",
        "next up: gradient descent.",
    ], spacing=0.085)
    save(fig, name="05_sauna_optimization.png")


# ================================================================ notebook 06
def sheet_06_least_squares():
    M, B = sp.symbols("M B")
    points = [(1, 2), (2, 5), (3, 3)]
    E = sum((M * px + B - py) ** 2 for px, py in points)
    assert sp.expand(E) == 14 * M ** 2 + 3 * B ** 2 + 38 + 12 * M * B - 42 * M - 20 * B
    EM, EB = sp.expand(sp.diff(E, M)), sp.expand(sp.diff(E, B))
    assert EM == 28 * M + 12 * B - 42 and EB == 12 * M + 6 * B - 20
    sol = sp.solve([EM, EB], [M, B], dict=True)[0]
    assert sol == {M: sp.Rational(1, 2), B: sp.Rational(7, 3)}
    e_min = E.subs(sol)
    assert e_min == sp.Rational(25, 6)
    m_opt, b_opt = float(sol[M]), float(sol[B])
    pts = np.array(points, float)
    m_fit, b_fit = np.polyfit(pts[:, 0], pts[:, 1], 1)
    assert np.isclose(m_fit, m_opt) and np.isclose(b_fit, b_opt)

    def E_num(m, b):
        return 14 * m ** 2 + 3 * b ** 2 + 38 + 12 * m * b - 42 * m - 20 * b

    residuals = m_opt * pts[:, 0] + b_opt - pts[:, 1]
    sq = residuals ** 2
    assert np.isclose(sq.sum(), float(e_min))

    fig = new_figure("Linear regression is an optimization problem: minimize E(M, B)",
                     "E(M, B) = Σ (M·xᵢ + B − yᵢ)² = 14M² + 3B² + 38 + 12MB − 42M − 20B.   "
                     "∇E = 0  →  M = 1/2,  B = 7/3,  E = 25/6 ≈ 4.167.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(0, 4.2, 100)
    ax.plot(xx, m_opt * xx + b_opt, color=BLUE, linewidth=2.4, label="y = x/2 + 7/3")
    for (px, py), r in zip(pts, residuals):
        ax.plot([px, px], [py, py + r], color=ORANGE, linewidth=2.5)
        ax.add_patch(Rectangle(
            (px, min(py, py + r)), abs(r), abs(r), facecolor=ORANGE_LIGHT, edgecolor="none",
            alpha=0.8, zorder=1))
        ax.text(px + abs(r) / 2, max(py, py + r) - 0.1, f"{r * r:.3f}", fontsize=9.5,
                ha="center", va="top", color=INK)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=INK, markersize=8, zorder=3, label="power lines")
    ax.set_xlim(0, 4.6)
    ax.set_ylim(1, 6)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "Vertical wires; cost = area of the squares", "x", "y")

    ax = fig.add_subplot(gs[0, 1])
    mm = np.linspace(-1, 2, 300)
    bb = np.linspace(0, 5, 300)
    MM, BB = np.meshgrid(mm, bb)
    cs = ax.contour(MM, BB, E_num(MM, BB), levels=[4.5, 6, 9, 14, 21, 30, 42], colors=BLUE_LIGHT,
                    linewidths=1.3)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    ax.plot(mm, (42 - 28 * mm) / 12, color=ORANGE, linewidth=2, label="∂E/∂M = 28M + 12B − 42 = 0")
    ax.plot(mm, (20 - 12 * mm) / 6, color=AQUA, linewidth=2, label="∂E/∂B = 12M + 6B − 20 = 0")
    ax.plot(m_opt, b_opt, "o", color=INK, markersize=8, zorder=4)
    ax.annotate("(1/2, 7/3)", (m_opt, b_opt), textcoords="offset points", xytext=(10, 6),
                fontsize=10, weight="bold")
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", fontsize=9)
    plain_axes(ax, "Contours of E: two lines meet at the bottom", "slope M", "intercept B")

    panel(fig.add_subplot(gs[0, 2]), "Solve ∇E = 0 by elimination", [
        "28M + 12B − 42 = 0",
        "12M +  6B − 20 = 0",
        "Double the second:  24M + 12B − 40 = 0",
        "",
        "Subtract:  4M − 2 = 0   →   M = 1/2",
        "Back in:   6 + 6B − 20 = 0   →   B = 7/3",
        "",
        "`SymPy: {B: 7/3, M: 1/2},  E = 25/6",
        f"`np.polyfit: M = {m_fit:.6f}, B = {b_fit:.6f}",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    m_line = np.linspace(-0.5, 1.5, 200)
    b_line = np.linspace(0.8, 3.9, 200)
    ax.plot(m_line, E_num(m_line, b_opt), color=ORANGE, linewidth=2.4, label="E(M, 7/3)  vs M")
    ax.plot(b_line, E_num(m_opt, b_line), color=AQUA, linewidth=2.4, linestyle="--",
            label="E(1/2, B)  vs B")
    ax.plot([m_opt, b_opt], [float(e_min)] * 2, "o", color=INK, zorder=3)
    ax.axhline(float(e_min), color=MUTED, linewidth=1, linestyle=":")
    ax.text(3.9, float(e_min) + 0.4, "E = 4.167", ha="right", fontsize=10, color=INK2)
    ax.set_ylim(0, 20)
    ax.legend(loc="upper right", fontsize=9.5)
    plain_axes(ax, "Both slices of E bottom out at the solution", "M (orange)  or  B (green)", "E")

    ax = fig.add_subplot(gs[1, 1])
    labels = [f"({px:g}, {py:g})" for px, py in points]
    pos = np.arange(3)
    ax.bar(pos, sq, width=0.6, color=[BLUE, ORANGE, BLUE], zorder=2)
    for p, r, s in zip(pos, residuals, sq):
        ax.text(p, s + 0.06, f"residual {r:+.3f}\nsquared {s:.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(pos, labels)
    ax.set_ylim(0, 3.6)
    ax.grid(axis="y", zorder=0)
    ax.set_xlabel("power line")
    ax.set_ylabel("squared wire length")
    ax.set_title(f"The minimum cost splits {sq[0]:.3f} + {sq[1]:.3f} + {sq[2]:.3f} = {sq.sum():.3f}")
    ax.title.set_fontsize(12)

    panel(fig.add_subplot(gs[1, 2]), "Why it matters", [
        "This is ordinary least squares:",
        "the line closest to the points in the",
        "sum-of-squared-vertical-distances sense.",
        "",
        "A sum of squares is an upward bowl in",
        "(M, B), so ∇E = 0 has one solution,",
        "and it is the minimum.",
        "",
        "Many parameters → a big linear system;",
        "gradient descent gets there iteratively.",
    ], spacing=0.085)
    save(fig, name="06_least_squares_line.png")


if __name__ == "__main__":
    sheet_00_tangent_planes()
    sheet_01_partial_derivatives()
    sheet_02_box_the_constant()
    sheet_03_gradients()
    sheet_04_gradient_zero()
    sheet_05_sauna()
    sheet_06_least_squares()
