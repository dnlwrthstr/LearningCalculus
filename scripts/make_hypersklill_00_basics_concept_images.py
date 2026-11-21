#!/usr/bin/env python3
"""Concept sheets for hypersklill/00_basics (sets, functions, coordinates, trigonometry, polynomials).

One page per notebook that summarizes its central idea. Every number is computed here
from the same sets, functions, points, angles and polynomials the notebook uses, with
asserts against the values the notebook prints. Images are written to
hypersklill/00_basics/img/.

Run from anywhere:  python scripts/make_hypersklill_00_basics_concept_images.py
"""
from functools import partial
from itertools import chain, combinations

import numpy as np
import sympy as sp
from matplotlib.patches import Circle, FancyBboxPatch, Ellipse, Polygon, Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, heat, panel)
from viz_style import save as _save

LESSON = REPO / "hypersklill" / "00_basics"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#bfe8d7"


# ================================================================ helpers
def cartesian_axes(ax, xlim, ylim, equal=True):
    """Plot frame with the x- and y-axis drawn through the origin."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.grid(True, zorder=0)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def set_text(values):
    """Sorted set literal, e.g. {1, 2, 3}; the empty set as ∅."""
    values = sorted(values, key=lambda v: (str(type(v)), v))
    return "{" + ", ".join(str(v) for v in values) + "}" if values else "∅"


def venn2(ax, sets, regions_on, centers=((-0.55, 0.0), (0.55, 0.0)), radius=1.0, labels=("A", "B")):
    """Two-circle Venn diagram with the elements written in their regions and chosen regions shaded."""
    (ca, cb), (A, B) = centers, sets
    if "10" in regions_on:
        ax.add_patch(Circle(ca, radius, facecolor=BLUE_LIGHT, edgecolor="none", zorder=1))
    if "01" in regions_on:
        ax.add_patch(Circle(cb, radius, facecolor=BLUE_LIGHT, edgecolor="none", zorder=1))
    if regions_on:
        lens_color = BLUE_LIGHT if "11" in regions_on else SURFACE
        lens = Circle(ca, radius, facecolor=lens_color, edgecolor="none", zorder=2)
        ax.add_patch(lens)
        lens.set_clip_path(Circle(cb, radius, transform=ax.transData))
    for center, name in zip(centers, labels):
        ax.add_patch(Circle(center, radius, facecolor="none", edgecolor=BLUE, linewidth=2, zorder=3))
        ax.text(center[0] + (-0.75 if center[0] < 0 else 0.75), 0.95, name, fontsize=15,
                weight="bold", color=BLUE, ha="center")
    spots = {"10": (ca[0] - 0.45, 0), "11": ((ca[0] + cb[0]) / 2, 0), "01": (cb[0] + 0.45, 0)}
    groups = {"10": A - B, "11": A & B, "01": B - A}
    for key, (x, y) in spots.items():
        if groups[key]:
            ax.text(x, y, "  ".join(str(v) for v in sorted(groups[key])), ha="center", va="center",
                    fontsize=14, weight="bold" if key in regions_on else None,
                    color=INK if key in regions_on else MUTED, zorder=4)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def mapping_diagram(ax, domain, codomain, arrows, left="X", right="Y", bad_arrows=()):
    """Arrow diagram: domain box left, codomain box right. Codomain targets that are hit are
    filled, targets hit more than once get an orange ring, arrows in bad_arrows are orange."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    for x0, name in [(0.8, left), (6.8, right)]:
        ax.add_patch(FancyBboxPatch((x0, 0.6), 2.4, 8.4, boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=CARD, edgecolor="none"))
        ax.text(x0 + 1.2, 9.35, name, ha="center", fontsize=12.5, weight="bold", color=INK2)

    def positions(items):
        if len(items) == 1:
            return {items[0]: 4.8}
        return {v: 8.2 - i * 6.8 / (len(items) - 1) for i, v in enumerate(items)}

    d_pos, c_pos = positions(domain), positions(codomain)
    hits = {}
    for _, y in arrows:
        hits[y] = hits.get(y, 0) + 1
    for v, y in d_pos.items():
        ax.text(2.0, y, str(v), ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=BLUE, linewidth=1.3))
    for v, y in c_pos.items():
        hit = hits.get(v, 0)
        ax.text(8.0, y, str(v), ha="center", va="center", fontsize=12, color=INK if hit else MUTED,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BLUE_LIGHT if hit else "white",
                          edgecolor=ORANGE if hit > 1 else (BLUE if hit else MUTED),
                          linewidth=2.4 if hit > 1 else 1.3))
    for x, y in arrows:
        color = ORANGE if (x, y) in bad_arrows else INK2
        ax.annotate("", xy=(6.75, c_pos[y]), xytext=(2.75, d_pos[x]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8 if color == INK2 else 2.4,
                                    shrinkA=0, shrinkB=0))
    return d_pos, c_pos


# ================================================================ notebook 00
def sheet_00_sets():
    A, B = {1, 2, 3}, {3, 4, 5}
    ops = {"A": A, "B": B, "A ∪ B": A | B, "A ∩ B": A & B, "A ∖ B": A - B,
           "B ∖ A": B - A, "A △ B": A ^ B}
    assert ops["A ∪ B"] == {1, 2, 3, 4, 5} and ops["A ∩ B"] == {3}
    assert ops["A ∖ B"] == {1, 2} and ops["B ∖ A"] == {4, 5} and ops["A △ B"] == {1, 2, 4, 5}
    assert ops["A △ B"] == (A - B) | (B - A)
    assert set([1, 2, 2, 3, 3, 3]) == {1, 2, 3} == {3, 2, 1}

    def power_set(iterable):                     # the notebook's function
        items = list(iterable)
        return list(chain.from_iterable(combinations(items, r) for r in range(len(items) + 1)))

    assert [set(s) for s in power_set({1, 2})] == [set(), {1}, {2}, {1, 2}]
    cart = {(a, b) for a in {1, 2, 3} for b in {"x", "y"}}
    assert len(cart) == 6

    fig = new_figure("A set is a bag of distinct elements, and four operations combine two bags",
                     "Order and duplicates don't matter: {1, 1, 2, 2, 3} = {3, 2, 1}.   "
                     r"$|\mathcal{P}(A)|$ = 2ⁿ subsets,  |A × B| = |A| · |B| ordered pairs.")
    gs = grid(fig, top=0.82, wspace=0.32)

    # membership table
    ax = fig.add_subplot(gs[0, 0])
    elements = [1, 2, 3, 4, 5]
    cols = list(ops)
    matrix = np.array([[1.0 if e in ops[c] else 0.0 for c in cols] for e in elements])
    labels = [["∈" if v else "·" for v in row] for row in matrix]
    heat(ax, matrix * 0.6, [str(e) for e in elements], cols, "", "element", "", labels=labels,
         vmax=1.0, highlight_cols=[3], fontsize=13)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_title("A = {1, 2, 3},  B = {3, 4, 5}: who is in what", pad=30)

    # number sets nested
    ax = fig.add_subplot(gs[0, 1])
    nests = [("ℂ", 3.0, 2.15, "i"), ("ℝ", 2.45, 1.75, "√2"), ("ℚ", 1.9, 1.35, "1/2"),
             ("ℤ", 1.35, 0.95, "−2"), ("ℕ", 0.75, 0.55, "3")]
    shades = ["#f3f7fc", "#e3eefb", "#cde2fb", "#a9cbf5", "#86b6ef"]
    for (name, w, h, example), shade in zip(nests, shades):
        cx = -3.0 + w
        ax.add_patch(Ellipse((cx, 0), 2 * w, 2 * h, facecolor=shade, edgecolor=BLUE, linewidth=1.4))
        ax.text(cx + w - 0.32, 0.0, name, fontsize=15, weight="bold", ha="center", va="center", color=INK)
        ax.text(cx + w - 0.32, -0.42, example, fontsize=10.5, ha="center", va="center", color=INK2)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Subsets nest: ℕ ⊆ ℤ ⊆ ℚ ⊆ ℝ ⊆ ℂ")

    # power set lattice of {1, 2, 3}, the notebook's {1, 2} part highlighted
    ax = fig.add_subplot(gs[0, 2])
    subsets = [frozenset(s) for s in power_set([1, 2, 3])]
    assert len(subsets) == 2 ** 3
    pos = {}
    for size in range(4):
        layer = [s for s in subsets if len(s) == size]
        for i, s in enumerate(layer):
            pos[s] = ((i - (len(layer) - 1) / 2) * 1.25, size * 1.2)
    for s in subsets:
        for t in subsets:
            if len(t) == len(s) + 1 and s < t:
                both = t <= {1, 2}
                ax.plot(*zip(pos[s], pos[t]), color=ORANGE if both else GRID_LINE, linewidth=2 if both else 1.2,
                        zorder=1)
    for s in subsets:
        inside = s <= {1, 2}
        ax.text(*pos[s], set_text(s), ha="center", va="center", fontsize=11, zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=ORANGE_LIGHT if inside else "white",
                          edgecolor=ORANGE if inside else BLUE, linewidth=1.3))
    ax.text(0, -0.75, r"orange: $\mathcal{P}(\{1, 2\})$ has 2² = 4 members", ha="center", fontsize=10.5, color=ORANGE)
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.0, 4.0)
    ax.axis("off")
    ax.set_title("Power set of {1, 2, 3}: 2³ = 8 subsets")

    # Cartesian product grid
    ax = fig.add_subplot(gs[1, 0])
    xs, ys = [1, 2, 3], ["x", "y"]
    for i, a in enumerate(xs):
        for j, b in enumerate(ys):
            assert (a, b) in cart
            ax.plot(i, j, "o", color=BLUE, markersize=11, zorder=3)
            ax.text(i, j + 0.2, f"({a}, {b})", ha="center", fontsize=11.5)
    ax.set_xticks(range(3), [str(v) for v in xs])
    ax.set_yticks(range(2), ys)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlabel("first entry a ∈ A = {1, 2, 3}")
    ax.set_ylabel("second entry b ∈ B = {x, y}")
    ax.grid(True)
    ax.set_title("A × B: 3 · 2 = 6 ordered pairs")
    ax.text(1, -0.42, "(1, x) ≠ (x, 1): order matters in a pair", ha="center", fontsize=10.5, color=INK2)

    panel(fig.add_subplot(gs[1, 1]), "Math ↔ Python", [
        "`x in A          x ∈ A`",
        "`len(A)          |A|`",
        "`A <= B / A < B  A ⊆ B / A ⊂ B`",
        "`A | B           A ∪ B`",
        "`A & B           A ∩ B`",
        "`A - B           A \\ B`",
        "`A ^ B           A △ B`",
        "`set()           ∅   ({} is a dict!)`",
        "`frozenset(...)  immutable, hashable`",
    ], spacing=0.092)

    panel(fig.add_subplot(gs[1, 2]), "Rules worth keeping", [
        "∅ ⊆ A and A ⊆ A for every set A.",
        "Disjoint: A ∩ B = ∅, e.g. {1, 2} and {3, 4}.",
        r"Set-builder: $\{x \in \mathbb{R} : x^2 = 1\} = \{-1, 1\}$",
        "",
        "Python removes duplicates for you:",
        "`set([1, 2, 2, 3, 3, 3]) -> {1, 2, 3}`",
        "",
        "Elements must be hashable: tuples yes,",
        "lists no. Printed order may vary.",
    ], spacing=0.09)
    save(fig, name="00_sets_basics.png")


GRID_LINE = "#c9c7bf"


# ================================================================ notebook 01
def sheet_01_venn():
    A, B = {1, 2, 3}, {3, 4, 5}
    fig = new_figure("Venn diagrams: every set operation is a shaded region",
                     "A = {1, 2, 3},  B = {3, 4, 5}.   Union = both circles, intersection = the overlap, "
                     "difference = one side only, symmetric difference = everything but the overlap.")
    gs = grid(fig, top=0.82, hspace=0.32, wspace=0.12, bottom=0.04)

    cases = [("Union", "A ∪ B", A | B, {"10", "11", "01"}),
             ("Intersection", "A ∩ B", A & B, {"11"}),
             ("Difference", "A ∖ B", A - B, {"10"}),
             ("Symmetric difference", "A △ B", A ^ B, {"10", "01"})]
    assert [c[2] for c in cases] == [{1, 2, 3, 4, 5}, {3}, {1, 2}, {1, 2, 4, 5}]
    for k, (name, symbol, result, regions) in enumerate(cases):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        venn2(ax, (A, B), regions)
        ax.set_title(f"{name}: {symbol} = {set_text(result)}")

    # complement in U
    ax = fig.add_subplot(gs[1, 1])
    U, A_c = {1, 2, 3, 4, 5, 6}, {1, 2, 3}
    complement = U - A_c
    assert complement == {4, 5, 6}
    ax.add_patch(Rectangle((-1.8, -1.15), 3.6, 2.4, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=2))
    ax.add_patch(Circle((-0.55, 0.05), 0.95, facecolor=SURFACE, edgecolor=BLUE, linewidth=2))
    ax.text(-0.55, 0.05, "1  2  3", ha="center", va="center", fontsize=14, color=MUTED)
    ax.text(-0.55, 0.62, "A", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(1.0, 0.05, "4  5  6", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(1.0, -0.45, "Aᶜ", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(1.62, 0.95, "U", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Complement: Aᶜ = U ∖ A = {set_text(complement)}")

    # three sets
    ax = fig.add_subplot(gs[1, 2])
    S = {"A": {1, 2, 3, 7}, "B": {3, 4, 5, 7}, "C": {2, 5, 6, 7}}
    assert S["A"] & S["B"] & S["C"] == {7}
    centers = {"A": (-0.55, 0.32), "B": (0.55, 0.32), "C": (0.0, -0.6)}
    for name, c in centers.items():
        ax.add_patch(Circle(c, 1.0, facecolor=BLUE_LIGHT, alpha=0.25, edgecolor="none"))
        ax.add_patch(Circle(c, 1.0, facecolor="none", edgecolor=BLUE, linewidth=2))
    ax.text(-1.45, 1.2, "A", fontsize=15, weight="bold", color=BLUE)
    ax.text(1.3, 1.2, "B", fontsize=15, weight="bold", color=BLUE)
    ax.text(0.95, -1.5, "C", fontsize=15, weight="bold", color=BLUE)
    anchors = {"A": (-0.95, 0.6), "B": (0.95, 0.6), "C": (0.0, -1.15), "AB": (0.0, 0.95),
               "AC": (-0.62, -0.4), "BC": (0.62, -0.4), "ABC": (0.0, 0.0)}
    regions = {}
    for e in sorted(set().union(*S.values())):
        key = "".join(n for n in "ABC" if e in S[n])
        regions.setdefault(key, []).append(e)
    for key, members in regions.items():
        triple = key == "ABC"
        ax.text(*anchors[key], "  ".join(map(str, members)), ha="center", va="center",
                fontsize=14, weight="bold" if triple else None, color=ORANGE if triple else INK)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.75, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Three sets, 7 regions: A ∩ B ∩ C = {7}")
    save(fig, name="01_set_theory_venn_diagrams.png")


# ================================================================ notebook 03
def sheet_03_functions():
    def f(x):
        return 5 * x + 10

    assert (f(2), f(7)) == (20, 45)
    X, Y = [1, 2, 3, 4], [0, 1, 4, 9, 16, 25]
    square = {x: x * x for x in X}
    range_f = set(square.values())
    assert range_f == {1, 4, 9, 16} and set(Y) - range_f == {0, 25}

    def is_function_relation(pairs):             # the notebook's checker
        seen = {}
        for x, y in pairs:
            if x in seen and seen[x] != y:
                return False
            seen[x] = y
        return True

    relation = [(1, "a"), (2, "b"), (3, "c"), (3, "d")]
    pairs = [(1, "b"), (2, "b"), (3, 8), (4, "python"), (5, "C++"), (6, 1)]
    assert not is_function_relation(relation) and is_function_relation(pairs)
    checks = [([(1, "a"), (2, "a"), (3, "b")], True), ([(1, "a"), (2, "b"), (2, "c")], False),
              ([(1, "a"), (2, "b"), (3, "c"), (3, "c")], True)]
    assert all(is_function_relation(r) == ok for r, ok in checks)

    fig = new_figure("A function gives every input exactly one output",
                     "f : X → Y.  Domain X = allowed inputs, codomain Y = allowed outputs, "
                     "range f(X) = {f(x) : x ∈ X} = the outputs actually produced ⊆ Y.")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-1, 8, 100)
    ax.plot(x, f(x), color=BLUE, linewidth=2.2, label="f(x) = 5x + 10")
    for x0 in (2, 7):
        ax.plot([x0, x0, -1], [0, f(x0), f(x0)], color=ORANGE, linestyle="--", linewidth=1.3)
        ax.plot(x0, f(x0), "o", color=ORANGE, markersize=8, zorder=3)
        ax.text(x0 + (0.25 if x0 < 5 else -0.25), f(x0) + (-4.5 if x0 < 5 else 2.5),
                f"f({x0}) = 5·{x0} + 10 = {f(x0)}", fontsize=10.5, ha="left" if x0 < 5 else "right")
    ax.set_xlim(-1, 8)
    ax.set_ylim(0, 60)
    ax.set_xlabel("input x")
    ax.set_ylabel("output y = f(x)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("f(x) means 'apply f to x', not f · x")

    ax = fig.add_subplot(gs[0, 1])
    mapping_diagram(ax, X, Y, list(square.items()), "domain X", "codomain Y")
    ax.set_title("f(x) = x²: range {1, 4, 9, 16} ⊊ codomain")
    ax.text(8.0, 0.05, "0 and 25 unused", ha="center", fontsize=10, color=MUTED)

    ax = fig.add_subplot(gs[0, 2])
    mapping_diagram(ax, [p[0] for p in pairs], ["b", 8, "python", "C++", 1], pairs, "A", "outputs")
    ax.set_title("Many-to-one is fine: f(1) = f(2) = b")

    ax = fig.add_subplot(gs[1, 0])
    mapping_diagram(ax, [1, 2, 3], ["a", "b", "c", "d"], relation, "inputs", "outputs",
                    bad_arrows={(3, "c"), (3, "d")})
    ax.set_title("Not a function: input 3 has two outputs")

    ax = fig.add_subplot(gs[1, 1])
    domain = [-2, -1, 0, 1, 2]
    images = {x: x ** 3 for x in domain}
    assert set(images.values()) == {0, 1, 8, -8, -1}
    t = np.linspace(-2.2, 2.2, 200)
    ax.plot(t, t ** 3, color=BLUE_LIGHT, linewidth=2, label="y = x³ on ℝ")
    ax.plot(domain, [images[x] for x in domain], "o", color=BLUE, markersize=9, zorder=3,
            label="y = x³ on X = {−2, …, 2}")
    for x0 in domain:
        ax.text(x0 + (0.12 if x0 < 2 else -0.12), images[x0] + 0.6, f"({x0}, {images[x0]})".replace("-", "−"), fontsize=10,
                ha="left" if x0 < 2 else "right")
    ax.axvline(1, color=ORANGE, linestyle=":", linewidth=1.5)
    ax.text(1.08, -9, "a vertical line meets\nthe graph once", fontsize=9.5, color=ORANGE)
    cartesian_axes(ax, (-2.5, 2.5), (-10, 10), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Unnamed rule y = x³: one y per x")

    panel(fig.add_subplot(gs[1, 2]), "Checking a finite relation", [
        "A relation is a function when no input",
        "appears with two different outputs.",
        "",
        "`{(1,a),(2,a),(3,b)}        -> True`",
        "`{(1,a),(2,b),(2,c)}        -> False`",
        "`{(1,a),(2,b),(3,c),(3,c)}  -> True`",
        "",
        "Repeating the same pair (3, c) is harmless:",
        "input 3 still has one output.",
        "In Python: a dict is a finite function.",
    ], spacing=0.085)
    save(fig, name="03_functions_in_mathematics.png")


# ================================================================ notebook 04
def sheet_04_injective_surjective():
    def classify_mapping(mapping, codomain):    # the notebook's checker
        outputs = list(mapping.values())
        injective = len(outputs) == len(set(outputs))
        surjective = set(outputs) == set(codomain)
        return injective, surjective, injective and surjective

    codomain = {"a", "b", "c"}
    results = [classify_mapping(m, codomain) for m in
               ({1: "a", 2: "b", 3: "c"}, {1: "a", 2: "a", 3: "b"}, {1: "a", 2: "b"})]
    assert results == [(True, True, True), (False, False, False), (True, False, False)]

    fig = new_figure("Injective, surjective, bijective: count the arrows landing on each target",
                     "Injective: f(x₁) = f(x₂) ⇒ x₁ = x₂ (no target hit twice).   Surjective: every y ∈ Y is hit.   "
                     "Bijective: both, so f⁻¹ exists.")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    mapping_diagram(ax, [1, 2, 3], ["a", "b", "c", "d"], [(1, "a"), (2, "b"), (3, "c")], "domain", "codomain")
    ax.set_title("Injective, not surjective: d is missed")

    ax = fig.add_subplot(gs[0, 1])
    arrows = [(1, "a"), (2, "b"), (3, "c"), (4, "a")]
    mapping_diagram(ax, [1, 2, 3, 4], ["a", "b", "c"], arrows, "domain", "codomain",
                    bad_arrows={(1, "a"), (4, "a")})
    ax.set_title("Surjective, not injective: a is hit twice")

    ax = fig.add_subplot(gs[0, 2])
    mapping_diagram(ax, [1, 2, 3], ["a", "b", "c"], [(1, "a"), (2, "b"), (3, "c")], "domain", "codomain")
    ax.set_title("Bijective: exactly one arrow per target")

    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(-3, 4)
    h = xs * xs
    assert list(h) == [9, 4, 1, 0, 1, 4, 9]
    g_out = sorted({3 * x for x in range(-5, 6)})
    assert g_out[:3] == [-15, -12, -9]
    ax.plot(xs, h, "o", color=BLUE, markersize=9, zorder=3)
    ax.axhline(4, color=ORANGE, linestyle="--", linewidth=1.5)
    for x0 in (-2, 2):
        ax.plot(x0, 4, "o", color=ORANGE, markersize=12, zorder=4, markerfacecolor="none", markeredgewidth=2)
    ax.text(0, 4.6, "h(−2) = h(2) = 4: not injective", ha="center", fontsize=10.5, color=ORANGE)
    ax.axhspan(-3.5, -0.3, color=CARD, zorder=0)
    ax.text(0, -2.0, "negative integers are never squares:\nnot surjective onto ℤ", ha="center",
            va="center", fontsize=10.5, color=INK2)
    cartesian_axes(ax, (-3.6, 3.6), (-3.5, 10.5), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_title("h(x) = x² on ℤ fails both tests")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(-3, 7, 50)
    ax.plot(t, t + 1, color=BLUE, linewidth=2, label="f(x) = x + 1")
    ax.plot(t, t - 1, color=AQUA, linewidth=2, label="f⁻¹(y) = y − 1")
    ax.plot(t, t, color=MUTED, linestyle=":", linewidth=1.3, label="mirror line y = x")
    for x0 in (-2, 0, 5):
        y0 = x0 + 1
        assert y0 - 1 == x0
        ax.plot(x0, y0, "o", color=BLUE, markersize=8, zorder=3)
        ax.plot(y0, x0, "o", color=AQUA, markersize=8, zorder=3)
        ax.plot([x0, y0], [y0, x0], color=ORANGE, linestyle="--", linewidth=1.2)
        ax.text(x0 - 0.3, y0 + 0.35, f"({x0}, {y0})".replace("-", "−"), fontsize=9.5, ha="right", color=BLUE)
        ax.text(y0 + 0.3, x0 - 0.55, f"({y0}, {x0})".replace("-", "−"), fontsize=9.5, color=AQUA)
    cartesian_axes(ax, (-3.5, 7.5), (-3.5, 7.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("A bijection reverses: x → x + 1 → back to x")

    yes = {True: "yes", False: "no"}
    shown = ["{1:a, 2:b, 3:c}", "{1:a, 2:a, 3:b}", "{1:a, 2:b}"]
    rows = [f"`{m:<16} {yes[r[0]]:<4} {yes[r[1]]:<4} {yes[r[2]]}`" for m, r in zip(shown, results)]
    panel(fig.add_subplot(gs[1, 2]), "classify_mapping, Y = {a, b, c}", [
        "`mapping          inj  surj bij`",
        *rows,
        "",
        "On ℤ: x + 1 is bijective; 3x is injective",
        "but misses 1, 2, 4, 5, …",
        "x³ on ℝ is a bijection, inverse ∛y.",
        "",
        "A cipher must be injective, or two",
        "messages share one ciphertext.",
    ], spacing=0.085)
    save(fig, name="04_properties_of_functions.png")


# ================================================================ notebook 05
def sheet_05_numerical_properties():
    def h(x):
        if x < 0:
            return -x
        elif 0 <= x <= 3:
            return x ** 2
        return 1

    assert [h(v) for v in [-2, -0.5, 0, 1.5, 3, 4]] == [2, 0.5, 0, 2.25, 9, 1]

    def f3(x):
        return np.sin(np.pi * x / 2)

    def f5(x):
        return x ** 6 - 10

    def f6(x):
        return x ** 5

    def f7(x):
        return 0.5 * x + np.cos(x)

    def sgn(x):
        return -1 if x < 0 else (0 if x == 0 else 1)

    def f4(x):
        return -np.sqrt(x)

    def sample_is_increasing(func, xs, strict=False):
        ys = [func(x) for x in xs]
        return all((ys[i + 1] > ys[i]) if strict else (ys[i + 1] >= ys[i]) for i in range(len(ys) - 1))

    def sample_is_even(func, xs, tol=1e-9):
        return all(abs(func(x) - func(-x)) < tol for x in xs)

    def sample_is_odd(func, xs, tol=1e-9):
        return all(abs(func(-x) + func(x)) < tol for x in xs)

    xs1, xs2, sym = np.linspace(-1, 1, 100), np.linspace(0, 9, 100), np.linspace(-2, 2, 50)
    checks = {"sgn increasing": sample_is_increasing(sgn, xs1),
              "sgn strictly increasing": sample_is_increasing(sgn, xs1, strict=True),
              "f4 decreasing": sample_is_increasing(lambda x: -f4(x), xs2),
              "f5 even": sample_is_even(f5, sym), "f6 odd": sample_is_odd(f6, sym),
              "f7 even": sample_is_even(f7, sym), "f7 odd": sample_is_odd(f7, sym)}
    assert list(checks.values()) == [True, False, True, True, True, False, False]
    assert np.isclose(f7(1.0), 1.0403023058681398) and np.isclose(f7(-1.0), 0.040302305868139765)

    fig = new_figure("Read a function's character off its graph: bounded, monotone, symmetric",
                     "Bounded: m ≤ f(x) ≤ M.   Increasing: x₂ > x₁ ⇒ f(x₂) ≥ f(x₁).   "
                     "Even: f(−x) = f(x).   Odd: f(−x) = −f(x).")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for lo, hi in [(-4, 0), (0, 3), (3, 6)]:
        t = np.linspace(lo, hi, 200)
        inside = (t + 1e-9) if hi == 0 else ((t - 1e-9) if hi == 3 else (t + 1e-9))
        ax.plot(t, [h(v) for v in inside], color=BLUE, linewidth=2.2)   # one formula per piece
    ax.plot(3, 9, "o", color=BLUE, markersize=8, zorder=3)
    ax.plot(3, 1, "o", color=BLUE, markersize=8, markerfacecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.text(3.35, 9.0, "h(3) = 3² = 9", fontsize=10.5, va="center")
    ax.text(3.2, 1.6, "x > 3: h = 1", fontsize=10.5)
    ax.text(-3.8, 5.0, "x < 0: h = −x", fontsize=10.5, color=INK2)
    ax.text(0.15, 6.5, "0 ≤ x ≤ 3:\nh = x²", fontsize=10.5, color=INK2)
    cartesian_axes(ax, (-4, 6), (-1, 10.5), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_title("Piecewise h: three formulas, one function")

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(-3, 3, 400)
    ax.axhspan(-1, 1, color=AQUA_LIGHT, alpha=0.6, zorder=0)
    ax.plot(t, t ** 2, color=BLUE, linewidth=2, label="x²: below by 0, not above")
    ax.plot(t, -t ** 2, color=ORANGE, linewidth=2, label="−x²: above by 0, not below")
    ax.plot(t, f3(t), color=AQUA, linewidth=2.4, label="sin(πx/2): stays in [−1, 1]")
    cartesian_axes(ax, (-3, 3), (-5.5, 7.5), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("Boundedness: is there a ceiling, a floor?")

    ax = fig.add_subplot(gs[0, 2])
    t = np.linspace(-3, 4, 700)
    ax.axvspan(-1, 1, color=AQUA_LIGHT, alpha=0.7, zorder=0)
    ax.axvspan(1, 3, color=ORANGE_LIGHT, alpha=0.7, zorder=0)
    ax.plot(t, f3(t), color=BLUE, linewidth=2.2)
    ax.text(0, 1.3, "increasing\non [−1, 1]", ha="center", fontsize=10.5, color=AQUA, weight="bold")
    ax.text(2, 1.3, "decreasing\non [1, 3]", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    cartesian_axes(ax, (-3, 4), (-1.4, 1.9), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("sin(πx/2)")
    ax.set_title("Monotone on intervals, not everywhere")

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-1.75, 1.75, 400)
    ax.plot(t, f5(t), color=BLUE, linewidth=2.2, label="f₅(x) = x⁶ − 10")
    v = f5(1.2)
    assert np.isclose(v, -7.014016)
    ax.plot([-1.2, 1.2], [v, v], "o", color=ORANGE, markersize=8, zorder=3)
    ax.annotate("", xy=(1.15, v), xytext=(-1.15, v),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.5))
    ax.text(0, v + 0.8, f"f₅(±1.2) = {v:.3f}".replace("-", "−"), ha="center", fontsize=10.5, color=ORANGE)
    cartesian_axes(ax, (-2, 2), (-11, 15), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title("Even: mirror image in the y-axis")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(-1.5, 1.5, 400)
    ax.plot(t, f6(t), color=BLUE, linewidth=2.2, label="f₆(x) = x⁵")
    v = f6(1.2)
    assert np.isclose(-v, -2.4883199999999994)
    ax.plot([-1.2, 1.2], [-v, v], "o", color=ORANGE, markersize=8, zorder=3)
    ax.plot([-1.2, 1.2], [-v, v], color=ORANGE, linestyle="--", linewidth=1.3)
    ax.text(-1.45, 4.2, f"f₆(1.2) = {v:.3f}\nf₆(−1.2) = {-v:.3f}".replace("-", "−"), fontsize=10.5,
            color=ORANGE, va="center", linespacing=1.5)
    cartesian_axes(ax, (-1.6, 1.6), (-7.8, 7.8), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Odd: 180° turn about the origin")

    panel(fig.add_subplot(gs[1, 2]), "Sample checks (not proofs)", [
        f"sgn on [−1, 1]: increasing {checks['sgn increasing']},",
        f"strictly increasing {checks['sgn strictly increasing']} (flat stretches).",
        f"f₄ = −√x decreasing on [0, 9]: {checks['f4 decreasing']}",
        f"f₅ even: {checks['f5 even']},   f₆ odd: {checks['f6 odd']}",
        "",
        "f₇(x) = 0.5x + cos x is neither:",
        f"f₇(1) = {f7(1.0):.3f},  f₇(−1) = {f7(-1.0):.3f},",
        f"−f₇(1) = {-f7(1.0):.3f}: no match either way.".replace("-", "−"),
        "",
        "Domain (−2, 4]: −2 excluded, 4 included.",
    ], spacing=0.085)
    save(fig, name="05_properties_of_numerical_functions.png")


# ================================================================ notebook 06
def sheet_06_cartesian():
    def quadrant(x, y):                          # the notebook's rule
        if x > 0 and y > 0:
            return "I"
        elif x < 0 and y > 0:
            return "II"
        elif x < 0 and y < 0:
            return "III"
        elif x > 0 and y < 0:
            return "IV"
        return "axis"

    def distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

    test_points = [(2, 3), (-3, 1), (-1.5, -2.5), (4, -2), (0, 5), (0, 0)]
    assert [quadrant(*p) for p in test_points] == ["I", "II", "III", "IV", "axis", "axis"]
    d0 = distance((0, 0), (-3, 1))
    d12 = distance((1, 2), (4, 6))
    d3 = distance((0, 0, 0), (2, 1, 2.5))
    assert np.isclose(d0, 3.1622776601683795) and d12 == 5.0 and np.isclose(d3, 3.3541019662496847)

    fig = new_figure("Cartesian coordinates: a point is a list of numbers, distance is Pythagoras",
                     "d = √((x₁ − x₂)² + (y₁ − y₂)²) in the plane; add (z₁ − z₂)² in 3D.   "
                     "n orthogonal axes → n coordinates per point.")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for (sx, sy), name in [((1, 1), "I"), ((-1, 1), "II"), ((-1, -1), "III"), ((1, -1), "IV")]:
        ax.add_patch(Rectangle((0, 0), 5.5 * sx, 5.5 * sy, facecolor=BLUE_LIGHT if name in ("I", "III") else CARD,
                               alpha=0.45, edgecolor="none", zorder=0))
        ax.text(4.6 * sx, 3.6 * sy, name, fontsize=17, weight="bold", ha="center", va="center", color=BLUE)
    offsets = {(2, 3): (0.2, 0.2), (-3, 1): (-0.2, 0.3), (-1.5, -2.5): (0.2, -0.5),
               (4, -2): (-0.2, 0.35), (0, 5): (0.25, -0.2), (0, 0): (0.2, 0.2)}
    for p in test_points:
        q = quadrant(*p)
        ax.plot(*p, "o", color=ORANGE if q == "axis" else INK, markersize=8, zorder=3)
        dx, dy = offsets[p]
        label = f"({p[0]:g}, {p[1]:g})".replace("-", "−") + (": on an axis" if q == "axis" else "")
        ax.text(p[0] + dx, p[1] + dy, label, fontsize=10,
                ha="right" if dx < 0 else "left", color=ORANGE if q == "axis" else INK)
    cartesian_axes(ax, (-5.5, 5.5), (-5.5, 5.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Signs of (x, y) pick the quadrant")

    ax = fig.add_subplot(gs[0, 1])
    ax.add_patch(Polygon([(0, 0), (-3, 0), (-3, 1)], closed=True, facecolor=BLUE_LIGHT, alpha=0.5, edgecolor="none"))
    ax.plot([0, -3], [0, 0], color=ORANGE, linestyle="--", linewidth=2)
    ax.plot([-3, -3], [0, 1], color=ORANGE, linestyle="--", linewidth=2)
    ax.plot([0, -3], [0, 1], color=BLUE, linewidth=2.5)
    ax.plot(-3, 1, "o", color=BLUE, markersize=9)
    ax.text(-3.1, 1.25, "(−3, 1)", fontsize=11, ha="center")
    ax.text(-1.5, -0.35, "3", fontsize=12, color=ORANGE, ha="center")
    ax.text(-3.3, 0.45, "1", fontsize=12, color=ORANGE, ha="center")
    ax.text(-1.3, 1.35, f"√(3² + 1²) = √10 ≈ {d0:.3f}", fontsize=11, color=BLUE, ha="center")
    cartesian_axes(ax, (-4.2, 1.2), (-1.2, 2.2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Distance to the origin: legs 3 and 1")

    ax = fig.add_subplot(gs[0, 2])
    p1, p2 = (1, 2), (4, 6)
    ax.add_patch(Polygon([p1, (4, 2), p2], closed=True, facecolor=BLUE_LIGHT, alpha=0.5, edgecolor="none"))
    ax.plot([1, 4], [2, 2], color=ORANGE, linestyle="--", linewidth=2)
    ax.plot([4, 4], [2, 6], color=ORANGE, linestyle="--", linewidth=2)
    ax.plot(*zip(p1, p2), color=BLUE, linewidth=2.5)
    for p, name in [(p1, "P₁ (1, 2)"), (p2, "P₂ (4, 6)")]:
        ax.plot(*p, "o", color=BLUE, markersize=9, zorder=3)
        ax.text(p[0] - 0.2, p[1] + 0.3, name, fontsize=11, ha="right")
    ax.text(2.5, 1.45, "x₂ − x₁ = 3", fontsize=11, color=ORANGE, ha="center")
    ax.text(4.2, 4.0, "y₂ − y₁ = 4", fontsize=11, color=ORANGE)
    ax.text(2.5, 0.8, f"d = √(9 + 16) = {d12:g}", fontsize=11, color=BLUE, ha="center", weight="bold")
    cartesian_axes(ax, (-0.5, 6.5), (-0.5, 7.3))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Between two points: a 3-4-5 triangle")

    ax = fig.add_subplot(gs[1, 0])
    t = np.linspace(-4, 4, 400)
    ax.plot(t, t, color=BLUE, linewidth=2.2, label="y = x: quadrants I and III")
    ax.plot(t, t ** 2, color=ORANGE, linewidth=2.2, label="y = x²: II, origin, I")
    cartesian_axes(ax, (-4, 4), (-4, 10), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center", fontsize=10)
    ax.set_title("A graph is the set of points (x, f(x))")

    ax = fig.add_subplot(gs[1, 1], projection="3d")
    P = (2, 1, 2.5)
    for end in [(3, 0, 0), (0, 3, 0), (0, 0, 3)]:
        ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=INK2, linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot([2, 2], [1, 1], [0, 2.5], color=ORANGE, linestyle="--")
    ax.plot([2, 2], [0, 1], [0, 0], color=ORANGE, linestyle="--")
    ax.plot([0, 2], [1, 1], [0, 0], color=ORANGE, linestyle="--")
    ax.plot([0, 2], [0, 1], [0, 2.5], color=BLUE, linewidth=2.2)
    ax.scatter(*P, s=70, color=BLUE, depthshade=False)
    ax.text(2.05, 1.05, 2.75, "(2, 1, 2.5)", fontsize=11)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_zticks([0, 1, 2, 3])
    ax.view_init(elev=22, azim=-58)
    ax.set_title(f"3D: distance to O = √(4 + 1 + 6.25) ≈ {d3:.3f}", pad=0)

    panel(fig.add_subplot(gs[1, 2]), "The rules", [
        "x > 0, y > 0 → I        x < 0, y > 0 → II",
        "x < 0, y < 0 → III      x > 0, y < 0 → IV",
        "x = 0 or y = 0 → on an axis",
        "",
        "Exercises: (6, 8) is 10 from O;",
        "(1, 1) to (4, 5) is 5 apart.",
        "",
        "More axes, same idea: a city as a 4D point",
        "(lat, lon, temp, pop) = (47.37, 8.54, 10.5, 0.43).",
    ], spacing=0.09)
    save(fig, name="06_cartesian_coordinates.png")


# ================================================================ notebook 07
def sheet_07_trig_functions():
    opp, adj, hyp = 4, 3, 5
    ratios = {"sin": opp / hyp, "cos": adj / hyp, "tan": opp / adj, "cot": adj / opp}
    assert ratios == {"sin": 0.8, "cos": 0.6, "tan": 4 / 3, "cot": 0.75}
    angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    names = ["0", "π/6", "π/4", "π/3", "π/2"]
    height = 1200 * np.sin(np.deg2rad(30))
    V, a = 20, np.deg2rad(35)
    vx, vy = V * np.cos(a), V * np.sin(a)
    assert np.isclose(height, 600) and np.isclose(vx, 16.383040885779835) and np.isclose(vy, 11.471528727020921)

    fig = new_figure("sin, cos, tan, cot: side ratios of a right triangle, coordinates on the unit circle",
                     "sin α = opp/hyp,  cos α = adj/hyp,  tan α = opp/adj,  cot α = adj/opp,   "
                     "and sin²x + cos²x = 1 for every x.")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Polygon([(0, 0), (3, 0), (3, 4)], closed=True, facecolor=BLUE_LIGHT, alpha=0.5,
                         edgecolor=BLUE, linewidth=2))
    ax.add_patch(Rectangle((2.7, 0), 0.3, 0.3, facecolor="none", edgecolor=INK2))
    ax.text(1.5, -0.4, "adjacent = 3", ha="center", fontsize=11)
    ax.text(3.12, 1.3, "opposite\n= 4", fontsize=11, va="center")
    ax.text(1.15, 2.3, "hypotenuse = 5", fontsize=11, rotation=53, ha="center", va="center")
    ax.text(0.55, 0.22, "α", fontsize=14, color=ORANGE, weight="bold")
    ax.text(5.0, 3.6, f"sin α = 4/5 = {ratios['sin']:.1f}\ncos α = 3/5 = {ratios['cos']:.1f}\n"
            f"tan α = 4/3 ≈ {ratios['tan']:.3f}\ncot α = 3/4 = {ratios['cot']:.2f}",
            fontsize=11, va="top", linespacing=1.6)
    ax.set_xlim(-0.3, 8.6)
    ax.set_ylim(-0.8, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A 3-4-5 triangle gives all four ratios")

    ax = fig.add_subplot(gs[0, 1])
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, linewidth=1.8)
    for ang, name in zip(angles, names):
        c, s = np.cos(ang), np.sin(ang)
        ax.plot([0, c], [0, s], color=BLUE_LIGHT, linewidth=1.5)
        ax.plot(c, s, "o", color=ORANGE, markersize=7, zorder=3)
        ax.text(c + 0.07, s + 0.03, f"{name}: ({c:.3f}, {s:.3f})", fontsize=9.5)
    ang = np.pi / 3
    ax.plot([np.cos(ang), np.cos(ang)], [0, np.sin(ang)], color=ORANGE, linestyle="--")
    ax.text(np.cos(ang) + 0.03, 0.1, "sin", fontsize=10, color=ORANGE)
    ax.text(0.12, -0.12, "cos", fontsize=10, color=ORANGE)
    cartesian_axes(ax, (-1.15, 2.2), (-1.15, 1.25))
    ax.set_xlabel("x = cos α")
    ax.set_ylabel("y = sin α")
    ax.set_title("Unit circle: (cos α, sin α), so cos² + sin² = 1")

    ax = fig.add_subplot(gs[0, 2])
    x = np.linspace(-2 * np.pi, 2 * np.pi, 800)
    ax.plot(x, np.sin(x), color=BLUE, linewidth=2, label="sin x")
    ax.plot(x, np.cos(x), color=ORANGE, linewidth=2, label="cos x")
    ax.plot(angles, np.sin(angles), "o", color=BLUE, markersize=6, zorder=3)
    ax.plot(angles, np.cos(angles), "o", color=ORANGE, markersize=6, zorder=3)
    ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi], ["−2π", "−π", "0", "π", "2π"])
    cartesian_axes(ax, (-2 * np.pi, 2 * np.pi), (-1.3, 1.6), equal=False)
    ax.set_xlabel("x (radians)")
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.set_title("Both waves live in [−1, 1], period 2π")

    ax = fig.add_subplot(gs[1, 0])
    L = 1200
    base = L * np.cos(np.deg2rad(30))
    ax.add_patch(Polygon([(0, 0), (base, 0), (base, height)], closed=True, facecolor=BLUE_LIGHT,
                         alpha=0.5, edgecolor="none"))
    ax.plot([0, base], [0, height], color=BLUE, linewidth=3)
    ax.plot([base, base], [0, height], color=ORANGE, linestyle="--", linewidth=2)
    ax.plot([0, base], [0, 0], color=INK2, linewidth=1.2)
    ax.plot([0, base], [0, height], "o", color=BLUE, markersize=8)
    ax.text(base / 2 - 60, height / 2 + 40, "cable 1200 m", rotation=30, ha="center", fontsize=11, color=BLUE)
    ax.text(base + 30, height / 2, f"height = 1200 · sin 30°\n= {height:.0f} m", fontsize=11,
            color=ORANGE, va="center")
    ax.text(160, 25, "30°", fontsize=12, weight="bold")
    ax.set_xlim(-50, 1750)
    ax.set_ylim(-80, 720)
    ax.set_xlabel("horizontal (m)")
    ax.set_ylabel("vertical (m)")
    ax.grid(True)
    ax.set_title("Cable car: height = length · sin θ")

    ax = fig.add_subplot(gs[1, 1])
    ax.annotate("", xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6))
    ax.annotate("", xy=(vx, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
    ax.annotate("", xy=(vx, vy), xytext=(vx, 0), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2.2))
    ax.text(2.0, 6.0, "V = 20", fontsize=12, color=BLUE, weight="bold")
    ax.text(vx / 2, -1.6, f"Vx = 20 cos 35° = {vx:.2f}", fontsize=11, color=ORANGE, ha="center")
    ax.text(vx + 0.5, vy / 2, f"Vy = 20 sin 35°\n= {vy:.2f}", fontsize=11, color=AQUA, va="center")
    ax.text(2.6, 0.5, "35°", fontsize=12, weight="bold")
    cartesian_axes(ax, (-1, 25), (-3, 14))
    ax.set_xlabel("x-component")
    ax.set_ylabel("y-component")
    ax.set_title("Velocity split into components")

    rows = [f"`{n:<5} {np.sin(t):.4f}   {np.cos(t):.4f}`" for n, t in zip(names, angles)]
    panel(fig.add_subplot(gs[1, 2]), "Common angles", [
        "`angle sin      cos`",
        *rows,
        "",
        "cos(π/2) prints 6e−17: floating-point",
        "rounding of an exact 0.",
        "tan = sin/cos,  cot = cos/sin = 1/tan.",
    ], spacing=0.088)
    save(fig, name="07_trigonometric_functions.png")


# ================================================================ notebook 08
def sheet_08_trig_identities():
    cos_x = 3 / 5
    sin_x = np.sqrt(1 - cos_x ** 2)
    assert np.isclose(sin_x, 0.8)
    alpha = np.pi / 6
    assert np.isclose(np.sin(np.pi / 2 - alpha), np.cos(alpha))
    assert np.isclose(np.sin(1.1 + 2 * np.pi), 0.8912073600614351)
    x0 = 0.9
    assert np.isclose(np.sin(-x0), -0.7833269096274834) and np.isclose(np.cos(-x0), 0.6216099682706644)
    xd = 0.4
    assert np.isclose(np.sin(2 * xd), 0.7173560908995228) and np.isclose(2 * np.sin(xd) * np.cos(xd), np.sin(2 * xd))
    a, b = 0.8, 0.3
    assert np.isclose(np.sin(a + b), np.sin(a) * np.cos(b) + np.cos(a) * np.sin(b))
    tan2 = 2
    c2 = 1 / np.sqrt(1 + tan2 ** 2)
    assert np.isclose(c2, 0.4472135954999579) and np.isclose(tan2 * c2, 0.8944271909999159)

    fig = new_figure("Trig identities: one unit circle, many ways to rewrite the same number",
                     "sin²x + cos²x = 1,   sin(π/2 − α) = cos α,   sin(a ± b) = sin a cos b ± cos a sin b,   "
                     "sin 2x = 2 sin x cos x")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, linewidth=1.8)
    for s, color, quad in [(sin_x, AQUA, "QI"), (-sin_x, ORANGE, "QIV")]:
        ax.plot([0, cos_x], [0, s], color=color, linewidth=2)
        ax.plot([cos_x, cos_x], [0, s], color=color, linestyle="--", linewidth=1.5)
        ax.plot(cos_x, s, "o", color=color, markersize=8, zorder=3)
        ax.text(cos_x + 0.08, s + (0.05 if s > 0 else -0.15), f"{quad}: sin = {s:+.1f}".replace("-", "−"), fontsize=10.5, color=color)
    ax.axvline(cos_x, color=MUTED, linestyle=":", linewidth=1)
    ax.text(cos_x - 0.05, -1.32, "cos x = 3/5", fontsize=10.5, ha="right")
    cartesian_axes(ax, (-1.2, 1.9), (-1.4, 1.3))
    ax.set_xlabel("cos")
    ax.set_ylabel("sin")
    ax.set_title("sin x = ±√(1 − cos²x): the quadrant picks the sign")

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(-4 * np.pi, 4 * np.pi, 1200)
    ax.plot(x, np.sin(x), color=BLUE, linewidth=2)
    pts = 1.1 + 2 * np.pi * np.arange(-2, 2)
    ax.plot(pts, np.sin(pts), "o", color=ORANGE, markersize=8, zorder=3)
    for p in pts[:-1]:
        ax.annotate("", xy=(p + 2 * np.pi, 1.2), xytext=(p, 1.2),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.text(0, 1.42, "+2π", ha="center", fontsize=10.5, color=ORANGE)
    ax.text(-4 * np.pi + 0.3, -1.4, f"sin(1.1 + 2πk) = {np.sin(1.1):.4f} for every k", fontsize=10.5)
    ax.set_xticks([-4 * np.pi, -2 * np.pi, 0, 2 * np.pi, 4 * np.pi], ["−4π", "−2π", "0", "2π", "4π"])
    cartesian_axes(ax, (-4 * np.pi, 4 * np.pi), (-1.6, 1.65), equal=False)
    ax.set_xlabel("x")
    ax.set_title("Periodicity: sin(x + 2π) = sin x")

    ax = fig.add_subplot(gs[0, 2])
    x = np.linspace(-np.pi, np.pi, 400)
    ax.plot(x, np.cos(x), color=BLUE, linewidth=4, alpha=0.45, label="cos α")
    ax.plot(x, np.sin(np.pi / 2 - x), color=ORANGE, linewidth=1.8, linestyle="--", label="sin(π/2 − α)")
    ax.plot(alpha, np.cos(alpha), "o", color=INK, markersize=8, zorder=3)
    ax.text(alpha + 0.15, np.cos(alpha) + 0.08, f"α = π/6: both = {np.cos(alpha):.4f}", fontsize=10.5)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    cartesian_axes(ax, (-np.pi, np.pi), (-1.2, 1.4), equal=False)
    ax.set_xlabel("α")
    ax.legend(loc="lower center", fontsize=10)
    ax.set_title("Complementary angles: curves coincide")

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(-np.pi, np.pi, 400)
    ax.plot(x, np.sin(x), color=BLUE, linewidth=2, label="sin: odd")
    ax.plot(x, np.cos(x), color=ORANGE, linewidth=2, label="cos: even")
    ax.plot([-x0, x0], [np.sin(-x0), np.sin(x0)], "o--", color=BLUE, markersize=7, linewidth=1)
    ax.plot([-x0, x0], [np.cos(-x0), np.cos(x0)], "o--", color=ORANGE, markersize=7, linewidth=1)
    ax.text(-x0 + 0.15, np.sin(-x0) - 0.12, f"sin(±0.9) = ±{np.sin(x0):.3f}", fontsize=10, color=BLUE,
            va="top")
    ax.text(-x0 - 0.1, np.cos(x0) + 0.08, f"cos(±0.9) = {np.cos(x0):.3f}", fontsize=10, color=ORANGE, ha="right")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    cartesian_axes(ax, (-np.pi, np.pi), (-1.25, 1.5), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("sin(−x) = −sin x,  cos(−x) = cos x")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 2 * np.pi, 400)
    ax.plot(x, np.sin(2 * x), color=BLUE, linewidth=4, alpha=0.45, label="sin 2x")
    ax.plot(x, 2 * np.sin(x) * np.cos(x), color=ORANGE, linewidth=1.8, linestyle="--", label="2 sin x cos x")
    ax.plot(x, np.sin(x) ** 2, color=AQUA, linewidth=4, alpha=0.45, label="sin²x")
    ax.plot(x, (1 - np.cos(2 * x)) / 2, color=INK, linewidth=1.3, linestyle=":", label="(1 − cos 2x)/2")
    ax.plot(xd, np.sin(2 * xd), "o", color=INK, markersize=7, zorder=3)
    ax.text(0.15, 1.45, f"● x = 0.4: both = {np.sin(2 * xd):.4f}", fontsize=10)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    cartesian_axes(ax, (0, 2 * np.pi), (-1.2, 2.45), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title("Double angle and half-angle: same curves")

    panel(fig.add_subplot(gs[1, 2]), "Recovering values (Quadrant I)", [
        "Given sin x = 4/5:",
        r"$\cos x = \sqrt{1 - 16/25} = 3/5,\quad \tan x = 4/3$",
        "",
        "Given tan x = 2:",
        r"$\cos x = 1/\sqrt{1 + 2^2} = 1/\sqrt{5} \approx 0.447$",
        r"$\sin x = \tan x \cdot \cos x = 2/\sqrt{5} \approx 0.894$",
        "",
        r"$1 + \tan^2 x = 1/\cos^2 x$ (divide by cos²x)",
        "",
        "sum formula, a = 0.8, b = 0.3:",
        f"sin(1.1) = {np.sin(a + b):.4f} both ways",
    ], spacing=0.087)
    save(fig, name="08_trigonometric_identities.png")


# ================================================================ notebook 09
def sheet_09_polynomials():
    x = sp.symbols("x")
    P1, P2 = 2 * x ** 2 + 5 * x + 6, x ** 2 + 2 * x + 1
    total = sp.expand(P1 + P2)
    assert total == 3 * x ** 2 + 7 * x + 7
    P7, P8 = x ** 2 + 5 * x + 3, 3 * x - 4
    product = sp.expand(P7 * P8)
    assert product == 3 * x ** 3 + 11 * x ** 2 - 11 * x - 12
    P_rep = x ** 3 - 4 * x ** 2 + 5 * x - 2
    assert sp.factor(P_rep) == (x - 2) * (x - 1) ** 2
    assert sp.solve(sp.Eq(4 * x ** 2 + 2 * x - 6, 0), x) == [sp.Rational(-3, 2), 1]
    complex_roots = sp.solve(sp.Eq(x ** 2 - x + 3, 0), x)
    assert complex_roots == [sp.Rational(1, 2) - sp.sqrt(11) * sp.I / 2, sp.Rational(1, 2) + sp.sqrt(11) * sp.I / 2]
    assert (5 * x ** 4 - 2 * x ** 2 + 4 * x + 11).subs(x, 2) == 91
    A, B = 2 * x ** 2 + 5, 4 * x - 6
    assert sp.Poly(sp.expand(A + B), x).degree() == 2 and sp.Poly(sp.expand(A * B), x).degree() == 3
    assert sp.expand(x ** 2 + (-x ** 2 + 1)) == 1

    fig = new_figure("Polynomials: add matching powers, multiply term by term, read roots off the x-axis",
                     "P(x) = c₀ + c₁x + … + cₙxⁿ.   deg(P·Q) = deg P + deg Q,  deg(P + Q) ≤ max.   "
                     "Every root r gives a factor (x − r).")
    gs = grid(fig, top=0.82, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    powers = ["x²", "x", "1"]
    c1 = [int(sp.Poly(P1, x).coeff_monomial(x ** k)) for k in (2, 1, 0)]
    c2 = [int(sp.Poly(P2, x).coeff_monomial(x ** k)) for k in (2, 1, 0)]
    pos = np.arange(3)
    ax.bar(pos, c1, width=0.55, color=BLUE, label="2x² + 5x + 6", zorder=2)
    ax.bar(pos, c2, width=0.55, bottom=c1, color=ORANGE, label="x² + 2x + 1", zorder=2)
    for p, u, v in zip(pos, c1, c2):
        ax.text(p, u / 2, str(u), ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(p, u + v / 2, str(v), ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(p, u + v + 0.25, f"= {u + v}", ha="center", fontsize=12, weight="bold")
    ax.set_xticks(pos, [f"coefficient of {p}" for p in powers])
    ax.set_ylim(0, 9.8)
    ax.set_ylabel("coefficient")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Addition: 3x² + 7x + 7")

    ax = fig.add_subplot(gs[0, 1])
    rows, cols = [x ** 2, 5 * x, sp.Integer(3)], [3 * x, sp.Integer(-4)]
    prods = [[sp.expand(r * c) for c in cols] for r in rows]
    assert sp.expand(sum(sum(row) for row in prods)) == product
    degree = np.array([[sp.degree(p, x) for p in row] for row in prods], float)

    def pretty(e):
        return str(e).replace("**2", "²").replace("**3", "³").replace("*", "").replace("-", "−")

    heat(ax, degree + 0.6, [pretty(r) for r in rows], [pretty(c) for c in cols], "", "", "",
         labels=[[pretty(p) for p in row] for row in prods], vmax=4.2, fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_title("Multiplication: every term × every term", pad=14)
    ax.text(1.0, 3.3, "same shade = same power:  x²: −4 + 15 = 11,   x: −20 + 9 = −11\n"
            "(x² + 5x + 3)(3x − 4) = 3x³ + 11x² − 11x − 12", ha="center", va="top", fontsize=10.5,
            color=INK2, clip_on=False)

    panel(fig.add_subplot(gs[0, 2]), "Degree = highest power with c ≠ 0", [
        "3x⁴ + 2x² − x² − 14 = 3x⁴ + x² − 14: degree 4",
        "",
        "Sum:  (2x² + 5) + (4x − 6) = 2x² + 4x − 1",
        "degree max(2, 1) = 2 ...",
        "... unless leading terms cancel:",
        "x² + (−x² + 1) = 1, degree 0.",
        "",
        "Product:  (2x² + 5)(4x − 6)",
        "= 8x³ − 12x² + 20x − 30, degree 2 + 1 = 3",
        "",
        "Evaluate: P(x) = 5x⁴ − 2x² + 4x + 11 → P(2) = 91",
    ], spacing=0.082, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    f = sp.lambdify(x, P_rep, "numpy")
    t = np.linspace(-0.5, 3, 500)
    ax.plot(t, f(t), color=BLUE, linewidth=2.2, label="x³ − 4x² + 5x − 2 = (x − 2)(x − 1)²")
    for r, note, dy in [(1, "x = 1, double root:\ntouches", 0.35), (2, "x = 2, simple root:\ncrosses", -0.75)]:
        assert P_rep.subs(x, r) == 0
        ax.plot(r, 0, "o", color=ORANGE, markersize=9, zorder=3)
        ax.text(r, dy, note, ha="center", fontsize=10.5, color=ORANGE)
    cartesian_axes(ax, (-0.5, 3), (-1.6, 2.2), equal=False)
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Roots: where the graph meets the x-axis")

    ax = fig.add_subplot(gs[1, 1])
    t = np.linspace(-2.5, 3.8, 400)
    quads = [(4 * x ** 2 + 2 * x - 6, BLUE, "4x² + 2x − 6: roots −1.5, 1"),
             (x ** 2 - 5 * x + 6, AQUA, "x² − 5x + 6 = (x − 2)(x − 3)"),
             (x ** 2 - x + 3, ORANGE, "x² − x + 3: no real root")]
    for q, color, label in quads:
        ax.plot(t, sp.lambdify(x, q, "numpy")(t), color=color, linewidth=2.2, label=label)
        for r in sp.solve(q, x):
            if r.is_real:
                ax.plot(float(r), 0, "o", color=color, markersize=8, zorder=3)
    cartesian_axes(ax, (-2.5, 3.8), (-8, 14), equal=False)
    ax.set_xlabel("x")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
    ax.set_title("Degree 2: at most 2 roots")

    ax = fig.add_subplot(gs[1, 2])
    for q, color, label in quads:
        roots = [complex(r) for r in sp.solve(q, x)]
        ax.plot([z.real for z in roots], [z.imag for z in roots], "o", color=color, markersize=10,
                zorder=3, label=label.split(":")[0].split(" =")[0])
    for z in [complex(r) for r in complex_roots]:
        ax.text(z.real + 0.15, z.imag, f"1/2 {'+' if z.imag > 0 else '−'} i√11/2", fontsize=10.5,
                va="center", color=ORANGE)
    ax.text(-1.9, 0.25, "real roots sit on the real axis", fontsize=10, color=INK2)
    cartesian_axes(ax, (-2, 3.6), (-2.3, 2.3))
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("Missing real roots are complex pairs")
    save(fig, name="09_introduction_to_polynomials.png")


if __name__ == "__main__":
    sheet_00_sets()
    sheet_01_venn()
    sheet_03_functions()
    sheet_04_injective_surjective()
    sheet_05_numerical_properties()
    sheet_06_cartesian()
    sheet_07_trig_functions()
    sheet_08_trig_identities()
    sheet_09_polynomials()
