#!/usr/bin/env python3
"""Concept sheets for coursera/gradine_and_gradient_descent (tangent planes, partial derivatives,
gradient, Newton's method and the Hessian) and the two stand-alone rule notebooks
coursera/chain_rule.ipynb and coursera/product_rule.ipynb.

One page per notebook that summarizes its central idea. Every number is computed here from
the same functions, points, step sizes and starting values the notebook uses, with asserts
against the values the notebook prints (or states in its text where nothing is printed).
Images are written to coursera/gradine_and_gradient_descent/img/ and coursera/img/.

Run from anywhere:  python scripts/make_coursera_gradient_and_rules_concept_images.py
"""
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid)
from viz_style import save as _save

GRAD_LESSON = REPO / "coursera" / "gradine_and_gradient_descent"
save_grad = partial(_save, folder=GRAD_LESSON / "img")
save_rules = partial(_save, folder=REPO / "coursera" / "img")

ORANGE_LIGHT = "#f8c9b3"


# ================================================================ helpers
def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """viz_style.panel, but a line holding a fraction is set in display size (\\dfrac) and gets
    extra line height, so stacked fractions stay readable and never touch the next line."""
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
        y -= spacing * (1.8 if tall else 1.0)


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


def tangent(ax, t, x0, z0, slope, color, label, half_width, lw=2):
    seg = t[np.abs(t - x0) <= half_width]
    ax.plot(seg, z0 + slope * (seg - x0), "--", color=color, linewidth=lw, label=label)


def arrow(ax, start, end, color, lw=2.0, ms=14, z=4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, zorder=z))


def axes_cross(ax):
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)


def bowl(x, y):
    """f(x, y) = x² + y², the running surface of notebooks 00-02."""
    return x ** 2 + y ** 2


# ================================================================ 00_introduction_to_tangent_planes
def sheet_00_tangent_planes():
    x, y = sp.symbols("x y")
    f = x ** 2 + y ** 2
    a, b = 2, 4
    fx, fy = sp.diff(f, x), sp.diff(f, y)
    f_ab, fx_ab, fy_ab = (e.subs({x: a, y: b}) for e in (f, fx, fy))
    plane_sym = sp.expand(f_ab + fx_ab * (x - a) + fy_ab * (y - b))
    assert (str(fx), str(fy), f_ab, fx_ab, fy_ab) == ("2*x", "2*y", 20, 4, 8)
    assert str(plane_sym) == "4*x + 8*y - 20"

    def plane(px, py):
        return 20 + 4 * (px - 2) + 8 * (py - 4)

    test_points = [(2.1, 4.1), (1.9, 3.8), (2.5, 4.2), (3.0, 5.0)]
    rows = [(px, py, bowl(px, py), plane(px, py)) for px, py in test_points]
    printed = [f"(x,y)=({px:.1f},{py:.1f}) | exact={ex:.4f} | plane={ap:.4f} | error={ex - ap:.4f}"
               for px, py, ex, ap in rows]
    assert printed == [
        "(x,y)=(2.1,4.1) | exact=21.2200 | plane=21.2000 | error=0.0200",
        "(x,y)=(1.9,3.8) | exact=18.0500 | plane=18.0000 | error=0.0500",
        "(x,y)=(2.5,4.2) | exact=23.8900 | plane=23.6000 | error=0.2900",
        "(x,y)=(3.0,5.0) | exact=34.0000 | plane=32.0000 | error=2.0000"]

    g2 = x * y
    g2_plane = sp.expand(g2.subs({x: 1, y: 3}) + sp.diff(g2, x).subs({x: 1, y: 3}) * (x - 1)
                         + sp.diff(g2, y).subs({x: 1, y: 3}) * (y - 3))
    assert g2_plane == 3 * x + y - 3

    fig = new_figure("Tangent plane: the flat surface that matches both slice slopes",
                     r"$z = f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b)$.     "
                     r"For $f = x^2 + y^2$ at (2, 4):  z = 20 + 4(x − 2) + 8(y − 4) = 4x + 8y − 20")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    X, Y = np.meshgrid(np.linspace(-2, 6, 50), np.linspace(0, 8, 50))
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.5, linewidth=0, rstride=2, cstride=2)
    PX, PY = np.meshgrid(np.linspace(0, 4, 10), np.linspace(2, 6, 10))
    ax.plot_surface(PX, PY, plane(PX, PY), color=ORANGE, alpha=0.55, linewidth=0)
    s = np.linspace(-2, 2, 30)
    ax.plot(2 + s, np.full_like(s, 4), 20 + 4 * s, color=INK, linewidth=2.2)
    ax.plot(np.full_like(s, 2), 4 + s, 20 + 8 * s, color=AQUA, linewidth=2.2)
    ax.scatter([2], [4], [20], color=INK, s=40, depthshade=False)
    ax.view_init(elev=22, azim=-120)
    style_3d(ax)
    ax.set_title("Plane through (2, 4, 20) holds both tangent lines", pad=0)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(-2, 6.5, 400)
    ax.plot(t, t ** 2 + 16, color=BLUE, linewidth=2.3, label="fix y = 4:  x² + 16  (vs x)")
    tangent(ax, t, 2, 20, 4, BLUE, "z = 20 + 4(x − 2)", 2.5)
    ax.plot(t, 4 + t ** 2, color=AQUA, linewidth=2.3, label="fix x = 2:  4 + y²  (vs y)")
    tangent(ax, t, 4, 20, 8, AQUA, "z = 20 + 8(y − 4)", 1.6)
    ax.plot([2, 4], [20, 20], "o", color=INK, zorder=3)
    ax.set_ylim(0, 48)
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "Two slices, two slopes: 4 and 8", "x (blue slice)  or  y (green slice)", "z")

    panel(fig.add_subplot(gs[0, 2]), "The four-step workflow", [
        "1. Height:   f(2, 4) = 2² + 4² = 20",
        r"2. Partials:  $f_x = 2x$,   $f_y = 2y$",
        r"3. At (2, 4):  $f_x(2,4) = 4$,   $f_y(2,4) = 8$",
        "4. Substitute:",
        "`   z = 20 + 4(x − 2) + 8(y − 4)",
        "`     = 4x + 8y − 20",
        "",
        "SymPy prints the same:",
        "`Tangent plane = 4*x + 8*y - 20",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    labels = [f"({px:g}, {py:g})" for px, py in test_points]
    pos = np.arange(len(rows))
    exact = [r[2] for r in rows]
    approx = [r[3] for r in rows]
    ax.bar(pos - 0.19, exact, width=0.36, color=BLUE, label="exact f(x, y)", zorder=2)
    ax.bar(pos + 0.19, approx, width=0.36, color=ORANGE, label="plane L(x, y)", zorder=2)
    for p, ex, ap in zip(pos, exact, approx):
        ax.text(p, max(ex, ap) + 0.6, f"error {ex - ap:.2f}", ha="center", fontsize=10)
    ax.set_xticks(pos, labels)
    ax.set_ylim(15, 37)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "f ≈ L near (2, 4): the notebook's test points", "test point (x, y)", "value")
    ax.grid(False, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    g = np.linspace(0, 4, 300)
    GX, GY = np.meshgrid(g, g + 2)
    gap = bowl(GX, GY) - plane(GX, GY)
    assert np.allclose(gap, (GX - 2) ** 2 + (GY - 4) ** 2)
    cf = ax.contourf(GX, GY, gap, levels=[0, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 4, 6, 8], cmap=SEQ)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("error f − L")
    cbar.outline.set_visible(False)
    ax.plot(2, 4, "*", color=ORANGE, markersize=14, zorder=4)
    offsets = [(-34, 6), (-36, -12), (7, -5), (7, 4)]
    for (px, py, ex, ap), off in zip(rows, offsets):
        ax.plot(px, py, "o", color=INK, markersize=5, zorder=4)
        ax.annotate(f"{ex - ap:.2f}", (px, py), textcoords="offset points", xytext=off,
                    fontsize=10, weight="bold", color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Error = (x − 2)² + (y − 4)²: tiny nearby")
    for spine in ax.spines.values():
        spine.set_visible(False)

    panel(fig.add_subplot(gs[1, 2]), "Worked example 2:  f = xy at (1, 3)", [
        "Height:   f(1, 3) = 1·3 = 3",
        r"Partials:  $f_x = y$,   $f_y = x$",
        r"At (1, 3):  $f_x = 3$,   $f_y = 1$",
        "`z = 3 + 3(x − 1) + 1(y − 3) = 3x + y − 3",
        "",
        "Why it works: each slice is a curve on the",
        "surface; its tangent line gives the slope in",
        "one direction. The plane is the only one that",
        "matches both slopes at the point.",
    ], spacing=0.09)
    save_grad(fig, name="00_introduction_to_tangent_planes.png")


# ================================================================ 01_partial_derivatives
def sheet_01_partial_derivatives():
    x, y = sp.symbols("x y")
    f = x ** 2 + y ** 2
    assert (str(sp.diff(f, x)), str(sp.diff(f, y))) == ("2*x", "2*y")
    f2 = x ** 2 * y + sp.sin(y)
    assert (str(sp.diff(f2, x)), str(sp.diff(f2, y))) == ("2*x*y", "x**2 + cos(y)")
    f3 = sp.exp(x * y)
    assert (str(sp.diff(f3, x)), str(sp.diff(f3, y))) == ("y*exp(x*y)", "x*exp(x*y)")

    def f_numeric(u, v):
        return u ** 2 + v ** 2

    def partial_x(func, u, v, h=1e-6):
        return (func(u + h, v) - func(u, v)) / h

    def partial_y(func, u, v, h=1e-6):
        return (func(u, v + h) - func(u, v)) / h

    x0, y0 = 2, 1
    num_x, num_y = partial_x(f_numeric, x0, y0), partial_y(f_numeric, x0, y0)
    assert (num_x, num_y) == (4.0000010006480125, 2.0000010003684565)
    z0 = x0 ** 2 + y0 ** 2
    assert (x0, y0, z0, 2 * x0, 2 * y0) == (2, 1, 5, 4, 2)

    fig = new_figure("A partial derivative: freeze the other variable, differentiate the slice",
                     r"$\partial f/\partial x = \lim_{h\to 0}\,[f(x+h,\,y) - f(x,\,y)]\,/\,h$"
                     r" with y held fixed.    $f = x^2 + y^2$:  $f_x = 2x$,  $f_y = 2y$;"
                     "  at (2, 1): 4 and 2")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    gv = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(gv, gv)
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.42, linewidth=0, rstride=2, cstride=2)
    t = np.linspace(-4, 4, 100)
    ax.plot(t, np.full_like(t, y0), bowl(t, y0), color=ORANGE, linewidth=3, label="slice y = 1")
    ax.plot(np.full_like(t, x0), t, bowl(x0, t), color=AQUA, linewidth=3, label="slice x = 2")
    ax.scatter([x0], [y0], [z0], color=INK, s=40, depthshade=False)
    ax.view_init(elev=24, azim=-58)
    style_3d(ax)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Two cuts through (2, 1, 5)", pad=0)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(-4, 4, 400)
    ax.plot(t, t ** 2 + 1, color=ORANGE, linewidth=2.3, label="f(x, 1) = x² + 1  (vs x)")
    tangent(ax, t, x0, z0, 4, ORANGE, "slope $f_x$(2, 1) = 4", 1.6)
    ax.plot(t, 4 + t ** 2, color=AQUA, linewidth=2.3, label="f(2, y) = 4 + y²  (vs y)")
    tangent(ax, t, y0, z0, 2, AQUA, "slope $f_y$(2, 1) = 2", 2.2)
    ax.plot([x0, y0], [z0, z0], "o", color=INK, zorder=3)
    ax.set_ylim(-1, 22)
    ax.legend(loc="upper center", fontsize=9)
    plain_axes(ax, "Each slice is an ordinary 1D function", "x (orange)  or  y (green)", "z")

    panel(fig.add_subplot(gs[0, 2]), "Treat the others as constants", [
        r"$\frac{\partial}{\partial x}(x^2 + y^2) = 2x + 0$   ($y^2$ is constant)",
        r"$\frac{\partial}{\partial y}(x^2 + y^2) = 0 + 2y$   ($x^2$ is constant)",
        "",
        r"$f = x^2y + \sin y$:   $f_x = 2xy$,   $f_y = x^2 + \cos y$",
        r"$f = e^{xy}$:   $f_x = y\,e^{xy}$,   $f_y = x\,e^{xy}$   (chain rule)",
        "",
        "Common slip: a frozen variable's term",
        "is a constant, so its derivative is 0.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    hs = np.logspace(-13, -1, 121)
    err_x = np.abs([partial_x(f_numeric, float(x0), float(y0), h) - 4 for h in hs])
    err_y = np.abs([partial_y(f_numeric, float(x0), float(y0), h) - 2 for h in hs])
    ax.loglog(hs, np.maximum(err_x, 1e-17), color=ORANGE, linewidth=2, label="|numeric $f_x$ − 4|")
    ax.loglog(hs, np.maximum(err_y, 1e-17), color=AQUA, linewidth=2, label="|numeric $f_y$ − 2|")
    ax.loglog(hs, hs, ":", color=MUTED, linewidth=1.5, label="h  (truncation error)")
    ax.plot(1e-6, num_x - 4, "o", color=INK, zorder=4)
    ax.annotate(f"h = 10⁻⁶ (notebook):\n$f_x$ ≈ {num_x:.10f}", (1e-6, num_x - 4),
                textcoords="axes fraction", xytext=(0.5, 0.1), fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(1e-12, 1)
    ax.legend(loc="upper center", fontsize=9)
    plain_axes(ax, "Forward difference at (2, 1): error vs step h", "step h", "absolute error")

    ax = fig.add_subplot(gs[1, 1])
    g = np.linspace(-3, 3.5, 300)
    GX, GY = np.meshgrid(g, g)
    ax.contour(GX, GY, bowl(GX, GY), levels=[1, 2, 5, 9, 13], colors=BLUE_LIGHT, linewidths=1.3)
    arrow(ax, (x0, y0), (x0 + 4 * 0.35, y0), ORANGE, lw=2.2)
    arrow(ax, (x0, y0), (x0, y0 + 2 * 0.35), AQUA, lw=2.2)
    arrow(ax, (x0, y0), (x0 + 4 * 0.35, y0 + 2 * 0.35), INK, lw=2.6, ms=16)
    ax.plot(x0, y0, "o", color=INK, zorder=5)
    ax.text(x0 + 1.5, y0 + 0.8, "∇f(2, 1) = (4, 2)", fontsize=10.5, weight="bold")
    ax.text(x0 + 0.6, y0 - 0.35, "$f_x$ = 4", fontsize=10, color=ORANGE, weight="bold")
    ax.text(x0 - 1.05, y0 + 0.45, "$f_y$ = 2", fontsize=10, color=AQUA, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 4.2)
    ax.set_ylim(-3, 3.5)
    plain_axes(ax, "Preview: stack the partials into ∇f", "x", "y")

    panel(fig.add_subplot(gs[1, 2]), "Checked in code", [
        "SymPy:",
        "`f_x = 2*x    f_y = 2*y",
        "",
        "Forward difference, h = 10⁻⁶, at (2, 1):",
        f"`Numerical f_x: {num_x!r}",
        "`Exact f_x    : 4",
        f"`Numerical f_y: {num_y!r}",
        "`Exact f_y    : 2",
        "",
        "For x² the forward difference is exactly",
        "2x + h: the error equals h until rounding",
        "(at h < 10⁻⁸) takes over.",
    ], spacing=0.076)
    save_grad(fig, name="01_partial_derivatives.png")


# ================================================================ 02_gradient
def sheet_02_gradient():
    x, y = sp.symbols("x y")
    f = x ** 2 + y ** 2
    grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
    assert grad == sp.Matrix([2 * x, 2 * y])
    g2 = x * y
    grad2 = sp.Matrix([sp.diff(g2, x), sp.diff(g2, y)])
    assert grad2 == sp.Matrix([y, x]) and grad2.subs({x: 2, y: 3}) == sp.Matrix([3, 2])
    assert sp.diff(3 * x ** 2 + 5 * y ** 2, x) == 6 * x and sp.diff(3 * x ** 2 + 5 * y ** 2, y) == 10 * y

    def f_numeric(u, v):
        return u ** 2 + v ** 2

    def partial_x(func, u, v, h=1e-5):
        return (func(u + h, v) - func(u, v)) / h

    def partial_y(func, u, v, h=1e-5):
        return (func(u, v + h) - func(u, v)) / h

    x0, y0 = 2, 3
    num = [partial_x(f_numeric, x0, y0), partial_y(f_numeric, x0, y0)]
    assert num == [4.000010000027032, 6.000009999951316]
    g0 = np.array([4, 6])
    magnitude = np.linalg.norm(g0)
    assert repr(magnitude) == "7.211102550927978" and np.isclose(magnitude, np.sqrt(52))
    f0 = bowl(x0, y0)

    fig = new_figure("The gradient: all partial derivatives in one vector, pointing uphill",
                     r"$\nabla f = (f_x,\; f_y)$.    $f = x^2 + y^2$:  $\nabla f = (2x,\; 2y)$,  "
                     r"$\nabla f(2,3) = (4,\; 6)$,  $\|\nabla f\| = \sqrt{52} \approx 7.211$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(-4, 4, 25)
    QX, QY = np.meshgrid(g, g)
    fine = np.linspace(-4, 4, 300)
    FX, FY = np.meshgrid(fine, fine)
    ax.contour(FX, FY, bowl(FX, FY), levels=[2, 5, 9, 20], colors=BLUE_LIGHT, linewidths=1.1)
    ax.quiver(QX[::4, ::4], QY[::4, ::4], 2 * QX[::4, ::4], 2 * QY[::4, ::4], color=BLUE,
              alpha=0.7, angles="xy", scale_units="xy", scale=12, width=0.005)
    th = np.linspace(0, 2 * np.pi, 200)
    r = np.sqrt(13)
    ax.plot(r * np.cos(th), r * np.sin(th), color=INK2, linewidth=1.6)
    tdir = np.array([-6, 4]) / magnitude
    ax.plot([x0 - 1.4 * tdir[0], x0 + 1.4 * tdir[0]], [y0 - 1.4 * tdir[1], y0 + 1.4 * tdir[1]],
            ":", color=INK, linewidth=1.6)
    arrow(ax, (x0, y0), (x0 + g0[0] / 5, y0 + g0[1] / 5), ORANGE, lw=2.6, ms=16, z=6)
    ax.plot(x0, y0, "o", color=INK, zorder=7)
    box = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9)
    ax.text(-4.15, 4.75, "∇f(2, 3) = (4, 6),  ⊥ circle x² + y² = 13", fontsize=10, color=ORANGE,
            weight="bold", va="center", bbox=box, zorder=8)
    ax.set_xlim(-4.3, 4.6)
    ax.set_ylim(-4.3, 5.1)
    ax.set_aspect("equal")
    plain_axes(ax, "Arrows point radially out, across the contours", "x", "y")

    ax = fig.add_subplot(gs[0, 1])
    theta = np.linspace(0, 360, 721)
    rad = np.radians(theta)
    slope = g0[0] * np.cos(rad) + g0[1] * np.sin(rad)
    best = np.degrees(np.arctan2(g0[1], g0[0]))
    ax.plot(theta, slope, color=BLUE, linewidth=2.4)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.plot(best, magnitude, "o", color=ORANGE, zorder=3)
    ax.annotate(f"along ∇f ({best:.1f}°): slope √52 = {magnitude:.3f}", (best, magnitude),
                textcoords="offset points", xytext=(10, 6), fontsize=10)
    for zero in (best + 90, best + 270):
        ax.plot(zero, 0, "o", color=INK2, zorder=3)
    ax.annotate("along the contour: 0", (best + 90, 0), textcoords="offset points",
                xytext=(8, 8), fontsize=10)
    ax.plot(best + 180, -magnitude, "o", color=AQUA, zorder=3)
    ax.annotate(f"along −∇f: −{magnitude:.3f}", (best + 180, -magnitude),
                textcoords="offset points", xytext=(10, 0), fontsize=10, va="center")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlim(0, 360)
    ax.set_ylim(-9.5, 9.5)
    plain_axes(ax, "Slope of f at (2, 3) in every direction", "direction angle θ (degrees)",
               r"slope  $\nabla f\cdot(\cos\theta,\ \sin\theta)$")

    panel(fig.add_subplot(gs[0, 2]), "Build it, evaluate it", [
        r"$\nabla f = \left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right) = (2x,\ 2y)$",
        r"$\nabla f(2,3) = (2\cdot 2,\ 2\cdot 3) = (4,\ 6)$",
        r"$\|\nabla f\| = \sqrt{4^2 + 6^2} = \sqrt{52}$",
        "",
        "Forward differences, h = 10⁻⁵:",
        "`Numerical gradient: [4.000010000027032,",
        "`                     6.000009999951316]",
        "`Magnitude: 7.211102550927978",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0], projection="3d")
    gv = np.linspace(-1, 4.5, 40)
    X, Y = np.meshgrid(gv, gv)
    ax.plot_surface(X, Y, bowl(X, Y), cmap=SEQ, alpha=0.45, linewidth=0, rstride=2, cstride=2)
    p = np.linspace(-1.2, 1.2, 10)
    PX, PY = np.meshgrid(x0 + p, y0 + p)
    ax.plot_surface(PX, PY, f0 + 4 * (PX - x0) + 6 * (PY - y0), color=ORANGE, alpha=0.65,
                    linewidth=0)
    ax.scatter([x0], [y0], [f0], color=INK, s=40, depthshade=False)
    ax.view_init(elev=24, azim=-130)
    style_3d(ax)
    ax.set_title("Tangent plane z = 13 + 4(x − 2) + 6(y − 3)", pad=0)

    ax = fig.add_subplot(gs[1, 1])
    g = np.linspace(-4, 4, 300)
    FX, FY = np.meshgrid(g, g)
    ax.contour(FX, FY, FX * FY, levels=[-9, -6, -3, -1, 1, 3, 6, 9], cmap="RdBu_r",
               linewidths=1.3)
    q = np.linspace(-3.5, 3.5, 8)
    QX, QY = np.meshgrid(q, q)
    ax.quiver(QX, QY, QY, QX, color=INK2, alpha=0.7, angles="xy", scale_units="xy", scale=6,
              width=0.005)
    arrow(ax, (x0, y0), (x0 + 3 / 2.2, y0 + 2 / 2.2), ORANGE, lw=2.6, ms=16, z=6)
    ax.plot(x0, y0, "o", color=INK, zorder=7)
    ax.text(x0 + 0.1, y0 - 0.75, "∇f(2, 3) = (3, 2)", fontsize=10, color=ORANGE, weight="bold",
            ha="center", bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none",
                                   alpha=0.9), zorder=8)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    plain_axes(ax, "Another example: f = xy,  ∇f = (y, x)", "x", "y")

    panel(fig.add_subplot(gs[1, 2]), "What the vector tells you", [
        "Direction: steepest ascent; −∇f is",
        "steepest descent (gradient descent).",
        "Length: the slope in that direction.",
        "",
        "Perpendicular to contours: along a level",
        "curve f does not change, so the slope there",
        "is 0, i.e. ∇f · direction = 0.",
        "",
        "Its entries are the tangent-plane slopes.",
        r"$3x^2 + 5y^2 \;\rightarrow\; \nabla f = (6x,\ 10y)$",
        r"$\sin(xy) \;\rightarrow\; \nabla f = (y\cos xy,\ x\cos xy)$",
    ], spacing=0.077)
    save_grad(fig, name="02_gradient.png")


# ================================================================ 03_newtons_method_and_hessian
def sheet_03_newton_hessian():
    x, y = sp.symbols("x y", real=True)
    f = x ** 4 + sp.Rational(8, 10) * y ** 4 + 4 * x ** 2 + 2 * y ** 2 - x * y \
        - sp.Rational(2, 10) * x ** 2 * y
    grad = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])
    H = sp.hessian(f, (x, y))
    assert sp.simplify(grad[0] - (4 * x ** 3 + 8 * x - y - sp.Rational(2, 5) * x * y)) == 0
    assert sp.simplify(grad[1] - (sp.Rational(16, 5) * y ** 3 + 4 * y - x - x ** 2 / 5)) == 0
    assert sp.simplify(H - sp.Matrix([[12 * x ** 2 + 8 - sp.Rational(2, 5) * y, -1 - sp.Rational(2, 5) * x],
                                      [-1 - sp.Rational(2, 5) * x, sp.Rational(48, 5) * y ** 2 + 4]])) \
        == sp.zeros(2, 2)

    def f_num(u, v):
        return u ** 4 + 0.8 * v ** 4 + 4 * u ** 2 + 2 * v ** 2 - u * v - 0.2 * u ** 2 * v

    grad_fn = sp.lambdify((x, y), grad, "numpy")
    H_fn = sp.lambdify((x, y), H, "numpy")

    def gradient(p):
        return np.array(grad_fn(p[0], p[1]), dtype=float).reshape(2)

    def hessian(p):
        return np.array(H_fn(p[0], p[1]), dtype=float).reshape(2, 2)

    p0 = np.array([3.0, 2.0])
    assert np.allclose(gradient(p0), [127.6, 28.8]) and \
        np.allclose(hessian(p0), [[115.2, -2.2], [-2.2, 42.4]])

    def newton(start, tol=1e-10, max_iter=50):
        p = np.array(start, dtype=float)
        history = [p.copy()]
        for _ in range(max_iter):
            g = gradient(p)
            step = np.linalg.solve(hessian(p), g)
            p = p - step
            history.append(p.copy())
            if np.linalg.norm(g) < tol:
                break
        return p, np.array(history)

    def gradient_descent(start, alpha=0.01, tol=1e-6, max_iter=10000):
        p = np.array(start, dtype=float)
        history = [p.copy()]
        for _ in range(max_iter):
            g = gradient(p)
            p = p - alpha * g
            history.append(p.copy())
            if np.linalg.norm(g) < tol:
                break
        return p, np.array(history)

    minimum, hist = newton([3.0, 2.0])
    assert len(hist) - 1 == 8
    table = [(f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{f_num(*pt):.6f}",
              f"{np.linalg.norm(gradient(pt)):.2e}") for pt in hist]
    assert table[:6] == [("3.000000", "2.000000", "128.200000", "1.31e+02"),
                         ("1.878278", "1.262552", "28.516566", "4.03e+01"),
                         ("1.070388", "0.727842", "6.233779", "1.28e+01"),
                         ("0.463956", "0.319311", "0.957699", "3.83e+00"),
                         ("0.076581", "0.051439", "0.024791", "5.76e-01"),
                         ("0.000249", "-0.000011", "0.000000", "2.03e-03")]
    assert [row[3] for row in table[6:]] == ["1.25e-08", "3.05e-19", "0.00e+00"]
    gd_min, gd_hist = gradient_descent([3.0, 2.0], alpha=0.01)
    assert len(gd_hist) - 1 == 399
    assert np.allclose(gd_min, [5.67045987e-08, 2.40204503e-07], rtol=1e-6)
    H_min = hessian(minimum)
    eig = np.linalg.eigvalsh(H_min)
    assert np.allclose(H_min, [[8, -1], [-1, 4]]) and np.allclose(eig, [3.76393202, 8.23606798])
    D = H_min[0, 0] * H_min[1, 1] - H_min[0, 1] * H_min[1, 0]
    assert f"{D:.4f}" == "31.0000"
    m2, h2 = newton([-4.0, 3.0])
    assert len(h2) - 1 == 8 and np.allclose(m2, 0, atol=1e-20)

    fig = new_figure("Newton's method: let the Hessian set the step",
                     r"$\mathbf{x}_{k+1} = \mathbf{x}_k - H_f(\mathbf{x}_k)^{-1}\,\nabla f(\mathbf{x}_k)$"
                     r"   for   $f = x^4 + 0.8y^4 + 4x^2 + 2y^2 - xy - 0.2x^2y$:   "
                     "8 steps from (3, 2), gradient descent (α = 0.01) needs 399")
    gs = grid(fig, top=0.84, wspace=0.3)

    xs = np.linspace(-1, 3.6, 300)
    ys = np.linspace(-1, 2.6, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = f_num(X, Y)

    ax = fig.add_subplot(gs[0, 0])
    ax.contour(X, Y, Z, levels=np.logspace(0, 3.1, 25), colors=BLUE_LIGHT, linewidths=1.0)
    ax.plot(gd_hist[:, 0], gd_hist[:, 1], "o-", color=AQUA, markersize=2.2, linewidth=1,
            label=f"gradient descent ({len(gd_hist) - 1} steps)")
    ax.plot(hist[:, 0], hist[:, 1], "s-", color=ORANGE, markersize=6, linewidth=2,
            label=f"Newton ({len(hist) - 1} steps)")
    for k in range(5):
        ax.annotate(str(k), hist[k], textcoords="offset points", xytext=(6, -12), fontsize=9.5,
                    color=ORANGE, weight="bold")
    ax.plot(0, 0, "*", color=INK, markersize=14, zorder=5)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "Both paths from (3, 2) on the contour map", "x", "y")

    ax = fig.add_subplot(gs[0, 1])
    gn = np.array([np.linalg.norm(gradient(pt)) for pt in hist])
    gg = np.array([np.linalg.norm(gradient(pt)) for pt in gd_hist])
    k_n = np.arange(len(gn))
    ax.loglog(np.arange(len(gg)) + 1, gg, color=AQUA, linewidth=2, label="gradient descent")
    ax.loglog(k_n[gn > 0] + 1, gn[gn > 0], "s-", color=ORANGE, linewidth=2, markersize=5,
              label="Newton")
    ax.annotate("Newton: 1.25e−08 → 3.05e−19,\nthen exactly 0 at step 8", (8, gn[7]),
                textcoords="offset points", xytext=(14, 4), fontsize=9.5, va="center")
    ax.annotate("stops at |∇f| < 10⁻⁶\nafter 399 steps", (len(gg), gg[-1]),
                textcoords="offset points", xytext=(-120, -40), fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlim(0.8, 600)
    ax.set_ylim(1e-21, 1e3)
    ax.legend(loc="upper right", fontsize=9.5)
    plain_axes(ax, "|∇f| per step: digits double for Newton", "iteration k + 1  (log)",
               "|∇f|  (log)")

    panel(fig.add_subplot(gs[0, 2]), "Gradient and Hessian", [
        r"$f_x = 4x^3 + 8x - y - 0.4xy$",
        r"$f_y = 3.2y^3 + 4y - x - 0.2x^2$",
        r"$f_{xx} = 12x^2 + 8 - 0.4y,\quad f_{yy} = 9.6y^2 + 4$",
        r"$f_{xy} = f_{yx} = -1 - 0.4x$   (symmetric)",
        "",
        "At the start (3, 2):",
        "`gradient : [127.6  28.8]",
        "`Hessian  : [[115.2  -2.2]",
        "`            [ -2.2  42.4]]",
        "",
        "Code solves H·step = ∇f instead of",
        "forming H⁻¹: same step, more stable.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    g = np.linspace(-1.3, 1.3, 300)
    QX, QY = np.meshgrid(g, g)
    quad = 0.5 * (8 * QX ** 2 - 2 * QX * QY + 4 * QY ** 2)
    ax.contour(QX, QY, quad, levels=[0.25, 0.75, 1.5, 2.5, 3.75], cmap=SEQ, linewidths=1.3)
    w, V = np.linalg.eigh(H_min)
    for lam, vec, color in zip(w, V.T, [AQUA, ORANGE]):
        vec = vec if vec[0] + vec[1] > 0 else -vec
        L = 1.0 / np.sqrt(lam) * 1.6
        arrow(ax, (0, 0), (L * vec[0], L * vec[1]), color, lw=2.4)
        name = "steep" if lam > 5 else "flat"
        ax.text(L * vec[0] + 0.05, L * vec[1] + (0.12 if lam > 5 else 0.1),
                f"λ = {lam:.4f} ({name})", color=color, fontsize=10, weight="bold", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none",
                          alpha=0.9))
    ax.plot(0, 0, "*", color=INK, markersize=12, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    plain_axes(ax, "H at the minimum: [[8, −1], [−1, 4]]", "x", "y")

    ax = fig.add_subplot(gs[1, 1])
    zx = np.linspace(-0.4, 3.6, 300)
    zy = np.linspace(-0.4, 2.6, 300)
    ZX, ZY = np.meshgrid(zx, zy)
    ax.contour(ZX, ZY, f_num(ZX, ZY), levels=np.logspace(0, 2.4, 18), colors=BLUE_LIGHT,
               linewidths=1.0)
    gd_step = -0.01 * gradient(p0)
    nt_step = hist[1] - hist[0]
    arrow(ax, tuple(p0), tuple(p0 + nt_step), ORANGE, lw=2.6, ms=16)
    arrow(ax, tuple(p0), tuple(p0 + gd_step), AQUA, lw=2.6, ms=16)
    ax.plot(*p0, "s", color=INK, zorder=5)
    ax.plot(0, 0, "*", color=INK, markersize=14, zorder=5)
    box = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9)
    ax.text(1.75, 0.75, f"Newton: −H⁻¹∇f\n→ ({hist[1][0]:.3f}, {hist[1][1]:.3f})", color=ORANGE,
            fontsize=10, weight="bold", ha="right", bbox=box)
    ax.text(1.2, 2.15, f"GD: −0.01·∇f\n→ ({p0[0] + gd_step[0]:.3f}, {p0[1] + gd_step[1]:.3f})",
            color=AQUA, fontsize=10, weight="bold", bbox=box)
    ax.set_aspect("equal")
    plain_axes(ax, "The first step from (3, 2)", "x", "y")

    panel(fig.add_subplot(gs[1, 2]), "Classify with the Hessian", [
        "At ∇f = 0, eigenvalues of H decide:",
        "  all > 0 → minimum,  all < 0 → maximum,",
        "  mixed signs → saddle.",
        r"2×2 shortcut: $D = f_{xx}f_{yy} - f_{xy}^2$",
        "",
        "Here at (0, 0):",
        "`Eigenvalues: [3.76393202 8.23606798]",
        "`D = 31.0000, f_xx = 8.0000",
        "→ positive definite: local minimum",
        "",
        "From (−4, 3) Newton also needs 8 steps.",
        "Catch: one n×n solve per step; and pure",
        "Newton is drawn to saddles too.",
    ], spacing=0.07)
    save_grad(fig, name="03_newtons_method_and_hessian.png")


# ================================================================ chain_rule
def sheet_chain_rule():
    xs_ = sp.symbols("x")
    examples = [sp.sin(xs_ ** 2), (3 * xs_ ** 2 + 1) ** 5, sp.exp(sp.cos(xs_)), sp.log(xs_ ** 2 + 1)]
    derivs = [str(sp.simplify(sp.diff(e, xs_))) for e in examples]
    assert derivs == ["2*x*cos(x**2)", "30*x*(3*x**2 + 1)**4", "-exp(cos(x))*sin(x)",
                      "2*x/(x**2 + 1)"]

    def numerical_derivative(func, u, h=1e-6):
        return (func(u + h) - func(u)) / h

    def composite_function(t):
        return np.sin(t ** 2)

    def chain_rule_derivative(t):
        return 2 * t * np.cos(t ** 2)

    t0 = 1.25
    num = numerical_derivative(composite_function, t0)
    exact = chain_rule_derivative(t0)
    assert (float(num), float(exact)) == (0.020737462413933372, 0.020740579059645943)
    h0 = t0 ** 2
    outer_slope = np.cos(h0)
    inner_slope = 2 * t0
    assert np.isclose(outer_slope * inner_slope, exact)

    fig = new_figure("Chain rule: the outer rate times the inner rate",
                     r"$T = f(h(t))$:   dT/dt = dT/dh · dh/dt,  i.e.  "
                     r"$T'(t) = f'(h(t))\,h'(t)$.     $\sin(t^2)\ \rightarrow\ \cos(t^2)\cdot 2t$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 2, 300)
    ax.plot(t, t ** 2, color=BLUE, linewidth=2.4, label="inner: h(t) = t²")
    tangent(ax, t, t0, h0, inner_slope, ORANGE, f"slope dh/dt = 2t = {inner_slope:g}", 0.45)
    ax.plot(t0, h0, "o", color=INK, zorder=3)
    ax.annotate(f"t = {t0}, h = {h0}", (t0, h0), textcoords="offset points", xytext=(-110, 8),
                fontsize=10)
    ax.legend(loc="upper left", fontsize=9.5)
    plain_axes(ax, "Step 1: Δt moves h", "t", "h")

    ax = fig.add_subplot(gs[0, 1])
    hh = np.linspace(0, 4, 300)
    ax.plot(hh, np.sin(hh), color=AQUA, linewidth=2.4, label="outer: f(h) = sin h")
    tangent(ax, hh, h0, np.sin(h0), outer_slope, ORANGE,
            f"slope dT/dh = cos h = {outer_slope:.6f}", 0.8)
    ax.plot(h0, np.sin(h0), "o", color=INK, zorder=3)
    ax.text(0.1, -0.5, f"h = {h0} is near the top of sin,\nso T barely moves", fontsize=10)
    ax.set_ylim(-1.2, 1.5)
    ax.legend(loc="lower left", fontsize=9.5)
    plain_axes(ax, "Step 2: Δh moves T", "h", "T")

    panel(fig.add_subplot(gs[0, 2]), "Δt → Δh → ΔT", [
        r"$\frac{\Delta T}{\Delta t} = \frac{\Delta T}{\Delta h}\cdot\frac{\Delta h}{\Delta t}$"
        r"$\;\;\rightarrow\;\; \frac{dT}{dt} = \frac{dT}{dh}\cdot\frac{dh}{dt}$",
        f"At t = {t0}:",
        f"  dh/dt = 2t = {inner_slope:g}",
        f"  dT/dh = cos(t²) = cos({h0}) = {outer_slope:.6f}",
        f"  dT/dt = {outer_slope:.6f} × {inner_slope:g} = {exact:.6f}",
        "",
        "Recipe: differentiate the outside, keep the",
        "inside unchanged, multiply by the inside's",
        "derivative.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-3, 3, 800)
    ax.plot(t, np.sin(t ** 2), color=BLUE, linewidth=2.2, label="T(t) = sin(t²)")
    ax.plot(t, 2 * t * np.cos(t ** 2), color=ORANGE, linewidth=2, linestyle="--",
            label="T′(t) = 2t cos(t²)")
    axes_cross(ax)
    ax.set_ylim(-6.5, 9)
    ax.set_xlim(-3, 3)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    plain_axes(ax, "Composite and derivative: wiggles speed up", "t", "value")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 2.2, 500)
    ax.plot(t, 2 * t * np.cos(t ** 2), color=ORANGE, linewidth=2.4, label="right: cos(t²)·2t")
    ax.plot(t, np.cos(t ** 2), color=MUTED, linewidth=2, linestyle="--",
            label="wrong: cos(t²)  (inner factor 2t forgotten)")
    pts = np.array([0.5, 1.0, 1.25, 1.75])
    ax.plot(pts, [numerical_derivative(composite_function, p) for p in pts], "o", color=INK,
            zorder=4, label="forward difference, h = 10⁻⁶")
    ax.annotate(f"t = 1.25:  numeric {num:.6f}\n            chain rule {exact:.6f}",
                (t0, num), textcoords="offset points", xytext=(-150, -58), fontsize=9.5,
                family="monospace", arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_ylim(-5, 3.2)
    ax.legend(loc="upper left", fontsize=9)
    plain_axes(ax, "Common mistake: drop the inner derivative", "t", "derivative")

    panel(fig.add_subplot(gs[1, 2]), "SymPy agrees", [
        "`sin(x**2)        → 2*x*cos(x**2)",
        "`(3*x**2 + 1)**5  → 30*x*(3*x**2 + 1)**4",
        "`exp(cos(x))      → -exp(cos(x))*sin(x)",
        "`log(x**2 + 1)    → 2*x/(x**2 + 1)",
        "",
        "The pattern:",
        r"$\frac{d}{dx}\,[g(x)]^n = n\,[g(x)]^{n-1}\,g'(x)$",
        r"$\frac{d}{dx}\,e^{g(x)} = e^{g(x)}\,g'(x)$,   $\frac{d}{dx}\ln g(x) = \frac{g'(x)}{g(x)}$",
    ], spacing=0.085)
    save_rules(fig, name="chain_rule.png")


# ================================================================ product_rule
def sheet_product_rule():
    ts = sp.symbols("t", positive=True)
    examples = [ts ** 2 * sp.sin(ts), sp.exp(ts) * sp.log(ts), (ts + 1) * (ts ** 3 - 2 * ts)]
    derivs = [str(sp.simplify(sp.diff(e, ts))) for e in examples]
    assert derivs == ["t*(t*cos(t) + 2*sin(t))", "(t*log(t) + 1)*exp(t)/t",
                      "4*t**3 + 3*t**2 - 4*t - 2"]

    def numerical_derivative(func, u, h=1e-6):
        return (func(u + h) - func(u)) / h

    def product_function(t):
        return t ** 2 * np.sin(t)

    def product_rule_derivative(t):
        return 2 * t * np.sin(t) + t ** 2 * np.cos(t)

    t0 = 1.5
    num = numerical_derivative(product_function, t0)
    rule = product_rule_derivative(t0)
    assert (float(num), float(rule)) == (3.151643750864963, 3.151643663564495)
    gph, ghp = 2 * t0 * np.sin(t0), t0 ** 2 * np.cos(t0)
    wrong = 2 * t0 * np.cos(t0)

    G, Hw, dG, dH = 3.0, 4.0, 0.8, 0.7   # the notebook's rectangle
    pieces = {"gh": G * Hw, "gΔh": G * dH, "Δgh": dG * Hw, "ΔgΔh": dG * dH}
    assert np.allclose(list(pieces.values()), [12, 2.1, 3.2, 0.56])
    assert np.isclose((G + dG) * (Hw + dH) - G * Hw, 2.1 + 3.2 + 0.56)

    fig = new_figure("Product rule: (gh)′ = g′h + gh′, never g′h′",
                     "Area of a g × h rectangle: when both sides grow, the new area is two strips "
                     "(Δg·h and g·Δh) plus a corner Δg·Δh that vanishes in the limit")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), Hw, G, facecolor=BLUE_LIGHT, edgecolor=INK2, linewidth=1.4))
    ax.add_patch(Rectangle((Hw, 0), dH, G, facecolor=AQUA, alpha=0.6, edgecolor=INK2))
    ax.add_patch(Rectangle((0, G), Hw, dG, facecolor=ORANGE, alpha=0.6, edgecolor=INK2))
    ax.add_patch(Rectangle((Hw, G), dH, dG, facecolor=INK2, alpha=0.8, edgecolor=INK2))
    ax.text(Hw / 2, G / 2, f"g·h = {pieces['gh']:g}", ha="center", va="center", fontsize=15,
            weight="bold")
    ax.text(Hw + dH / 2, G / 2, f"g·Δh = {pieces['gΔh']:g}", ha="center", va="center",
            fontsize=10.5, rotation=90)
    ax.text(Hw / 2, G + dG / 2, f"Δg·h = {pieces['Δgh']:g}", ha="center", va="center",
            fontsize=10.5)
    ax.annotate(f"Δg·Δh = {pieces['ΔgΔh']:g}", (Hw + dH / 2, G + dG / 2),
                xytext=(Hw - 0.6, G + dG + 0.55), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlim(-0.3, Hw + dH + 0.6)
    ax.set_ylim(-0.3, G + dG + 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0, Hw, Hw + dH], ["0", "4", "4.7"])
    ax.set_yticks([0, G, G + dG], ["0", "3", "3.8"])
    ax.set_xlabel("h(t) direction")
    ax.set_ylabel("g(t) direction")
    ax.set_title("The notebook's rectangle: 3 × 4 grows by 5.86")

    ax = fig.add_subplot(gs[0, 1])
    dts = np.logspace(-4, 0, 80)
    dg = (t0 + dts) ** 2 - t0 ** 2
    dh = np.sin(t0 + dts) - np.sin(t0)
    ax.plot(dts, dg * np.sin(t0) / dts, color=ORANGE, linewidth=2.2, label="Δg·h / Δt  → g′h")
    ax.plot(dts, t0 ** 2 * dh / dts, color=AQUA, linewidth=2.2, label="g·Δh / Δt  → gh′")
    ax.plot(dts, dg * dh / dts, color=INK2, linewidth=2.2, label="Δg·Δh / Δt  → 0")
    ax.axhline(gph, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.axhline(ghp, color=AQUA, linestyle=":", linewidth=1.3)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(loc="center right", fontsize=9.5)
    plain_axes(ax, "g = t², h = sin t at t = 1.5: shrink Δt", "Δt (log scale)  ← shrinking",
               "contribution to Δ(gh)/Δt")

    panel(fig.add_subplot(gs[0, 2]), "Derivation", [
        r"$\Delta f = g\,\Delta h + \Delta g\,h + \Delta g\,\Delta h$",
        "Divide by Δt:",
        r"$\frac{\Delta f}{\Delta t} = g\frac{\Delta h}{\Delta t} + h\frac{\Delta g}{\Delta t} + \frac{\Delta g\,\Delta h}{\Delta t}$",
        "Let Δt → 0: Δg → 0, so the last term → 0.",
        r"$f'(t) = g'(t)\,h(t) + g(t)\,h'(t)$",
        "",
        "Derivative of the first times the second,",
        "plus the first times derivative of the second.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-2 * np.pi, 2 * np.pi, 800)
    ax.plot(t, t ** 2 * np.sin(t), color=BLUE, linewidth=2.2, label="y = t² sin t")
    ax.plot(t, 2 * t * np.sin(t) + t ** 2 * np.cos(t), color=ORANGE, linewidth=2,
            linestyle="--", label="y′ = 2t sin t + t² cos t")
    ax.plot(t, 2 * t * np.cos(t), color=MUTED, linewidth=1.8, linestyle=":",
            label="g′h′ = 2t cos t  (wrong)")
    axes_cross(ax)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-45, 52)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    plain_axes(ax, "Function, derivative, and the tempting mistake", "t", "value")

    ax = fig.add_subplot(gs[1, 1])
    names = ["g′h\n2t sin t", "gh′\nt² cos t", "g′h + gh′", "numeric", "g′h′\n(wrong)"]
    pos = np.arange(5)
    ax.bar(pos[0], gph, width=0.6, color=ORANGE, zorder=2)
    ax.bar(pos[1], ghp, width=0.6, color=AQUA, zorder=2)
    ax.bar(pos[2], gph, width=0.6, color=ORANGE, zorder=2)
    ax.bar(pos[2], ghp, bottom=gph, width=0.6, color=AQUA, zorder=2)
    ax.bar(pos[3], num, width=0.6, color=BLUE, zorder=2)
    ax.bar(pos[4], wrong, width=0.6, color=MUTED, zorder=2)
    for p, v, fmt in zip(pos, [gph, ghp, rule, num, wrong],
                         ["{:.4f}", "{:.4f}", "{:.6f}", "{:.6f}", "{:.4f}"]):
        ax.text(p, v + 0.06, fmt.format(v), ha="center", fontsize=9.5)
    ax.set_xticks(pos, names, fontsize=10)
    ax.set_ylim(0, 3.7)
    ax.grid(axis="y", zorder=0)
    ax.set_title("At t = 1.5: the two strips sum to the slope")
    ax.set_ylabel("value")

    panel(fig.add_subplot(gs[1, 2]), "Worked examples, SymPy-checked", [
        "`t² sin t      → 2t sin t + t² cos t",
        "`eᵗ ln t       → eᵗ (ln t + 1/t)",
        "`(t+1)(t³−2t)  → 4t³ + 3t² − 4t − 2",
        "",
        "Numeric check at t = 1.5 (h = 10⁻⁶):",
        f"`Numerical derivative: {num!r}",
        f"`Product rule result : {rule!r}",
    ], spacing=0.09)
    save_rules(fig, name="product_rule.png")


if __name__ == "__main__":
    sheet_00_tangent_planes()
    sheet_01_partial_derivatives()
    sheet_02_gradient()
    sheet_03_newton_hessian()
    sheet_chain_rule()
    sheet_product_rule()
