#!/usr/bin/env python3
"""Concept sheets for coursera/04_optimization_neuronal_networks (perceptrons to backpropagation).

One page per notebook that summarizes its central idea. Every number is computed here from
the same data, weights, learning rates, epochs and random seeds the notebook uses, and is
checked against the notebook's printed outputs with asserts.
Images are written to coursera/04_optimization_neuronal_networks/img/.

Run from anywhere:  python scripts/make_coursera_04_neural_networks_concept_images.py
"""
from functools import partial

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "04_optimization_neuronal_networks"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"
# P(class 1) maps: blue for class 0 (sad / outside), orange for class 1 (happy / inside)
PROB = LinearSegmentedColormap.from_list("prob", ["#86b6ef", "#e4eefa", SURFACE, "#fbe0d3", "#f2a07c"])

# the house data of notebooks 00-02
SIZE = np.array([1000, 2000, 3000], dtype=float)
PRICE = np.array([20000, 30000, 50000], dtype=float)
ROOMS = np.array([2, 4, 7], dtype=float)

# the alien-sentiment data of notebooks 03-06: [# aack, # beep], 1 = happy, 0 = sad
ALIEN_X = np.array([[3, 0], [0, 2], [1, 3], [2, 1]], dtype=float)
ALIEN_Y = np.array([1, 0, 0, 1], dtype=float)

# XOR of notebooks 07-08
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0], dtype=float)
NN_KEYS = ['w11', 'w21', 'b1', 'w12', 'w22', 'b2', 'w1', 'w2', 'b']


# ================================================================ helpers
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss_mean(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))


def node(ax, x, y, label, color, r=0.32, fontsize=12, text_color="white"):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=text_color,
            weight="bold", zorder=4)


def edge(ax, p, q, r=0.32, label=None, frac=0.5, color=MUTED, label_color=INK2, offset=(0, 0.14),
         fontsize=10):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = (q - p) / np.linalg.norm(q - p)
    start, end = p + d * r, q - d * (r + 0.02)
    ax.annotate("", xy=end, xytext=start, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2, mutation_scale=12))
    if label:
        m = start + (end - start) * frac
        ax.text(m[0] + offset[0], m[1] + offset[1], label, ha="center", va="center",
                fontsize=fontsize, color=label_color,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0), zorder=5)


def diagram_axes(ax, xlim, ylim, title):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def class_scatter(ax, X, Y, labels=("class 1", "class 0"), s=130):
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=s, color=ORANGE, edgecolor=INK, linewidth=0.8,
               zorder=4, label=labels[0])
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=s, color=BLUE, edgecolor=INK, linewidth=0.8,
               zorder=4, label=labels[1])


def prob_map(ax, GX, GY, GZ):
    ax.contourf(GX, GY, GZ, levels=np.linspace(0, 1, 21), cmap=PROB)
    ax.contour(GX, GY, GZ, levels=[0.5], colors=INK, linewidths=2)


# ================================================================ notebook 00
def sheet_00_perceptron():
    w_fit, b_fit = np.polyfit(SIZE, PRICE, 1)
    assert f"{w_fit:.2f}" == "15.00" and f"{b_fit:.1f}" == "3333.3"
    x, w, b = np.array([2000, 4]), np.array([14.0, 1500.0]), 2000.0
    y_hat = np.dot(w, x) + b
    assert y_hat == 36000.0

    def mse(wv, bv):
        return np.mean((wv * SIZE[:, None, None] + bv - PRICE[:, None, None]) ** 2, axis=0)

    loss_poor = float(mse(np.array([[5.0]]), np.array([[0.0]]))[0, 0])
    loss_best = float(mse(np.array([[w_fit]]), np.array([[b_fit]]))[0, 0])
    assert f"{loss_poor:,.0f}" == "616,666,667" and f"{loss_best:,.0f}" == "5,555,556"
    x4, w4, b4 = np.array([2500, 5, 2, 1990]), np.array([12.0, 1500.0, 3000.0, 10.0]), 500.0
    assert np.dot(w4, x4) + b4 == 63900.0

    fig = new_figure("A perceptron with no activation is linear regression",
                     "ŷ = w₁x₁ + w₂x₂ + b = w·x + b: weight each input, add a bias. "
                     "Training means choosing w and b that make the error small.")
    gs = grid(fig, top=0.84, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    diagram_axes(ax, (-1.9, 6.3), (-0.2, 4.3), "One perceptron, two features")
    ins = [((0.6, 3.4), "x₁", "size 2000"), ((0.6, 2.0), "x₂", "rooms 4"), ((0.6, 0.6), "1", "bias")]
    for (pos, sym, name), wl in zip(ins, ["w₁ = 14", "w₂ = 1500", "b = 2000"]):
        node(ax, *pos, sym, BLUE)
        ax.text(pos[0] - 0.45, pos[1], name, ha="right", va="center", fontsize=10.5, color=INK2)
        edge(ax, pos, (3.2, 2.0), r=0.32, label=wl, frac=0.5, offset=(0, 0.28 if pos[1] == 2.0 else 0.0))
    node(ax, 3.2, 2.0, "Σ", ORANGE, r=0.42, fontsize=16)
    edge(ax, (3.2, 2.0), (5.3, 2.0), r=0.42)
    node(ax, 5.3, 2.0, "ŷ", AQUA)
    ax.text(5.3, 1.35, f"{y_hat:,.0f}", ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(5.3, 0.95, "(no activation)", ha="center", fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[0, 1])
    xs = np.linspace(800, 3200, 100)
    ax.plot(xs, w_fit * xs + b_fit, color=BLUE, linewidth=2.2,
            label=f"best fit: 15·size + {b_fit:,.0f}")
    ax.plot(xs, 5 * xs, color=MUTED, linewidth=1.8, linestyle="--", label="poor guess: 5·size")
    for s, p in zip(SIZE, PRICE):
        ax.plot([s, s], [w_fit * s + b_fit, p], color=BLUE, linewidth=1, linestyle=":")
        ax.plot([s, s], [5 * s, p], color=MUTED, linewidth=1, linestyle=":")
    ax.scatter(SIZE, PRICE, s=70, color=ORANGE, zorder=4, label="the three houses")
    ax.set_xlabel("size (sq ft)")
    ax.set_ylabel("price ($)")
    ax.set_ylim(0, 58000)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("One feature: the perceptron is a line")

    panel(fig.add_subplot(gs[0, 2]), "Inside the summation", [
        "Each input × its weight (its importance),",
        "then add them up and add the bias:",
        r"$\hat{y} = \Sigma_i\, w_i x_i + b = \mathbf{w}\cdot\mathbf{x} + b$",
        "",
        "House with 2000 sq ft and 4 rooms:",
        "ŷ = 14·2000 + 1500·4 + 2000 = 36,000",
        "",
        "No activation ⇒ ŷ is linear in x:",
        "exactly the linear-regression model,",
        "for 1, 2 or 100 features alike.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    ww, bb = np.meshgrid(np.linspace(0, 25, 300), np.linspace(-25000, 30000, 300))
    surface = mse(ww, bb)
    image = ax.contourf(ww, bb / 1000, np.log10(surface), levels=20, cmap=SEQ.reversed())
    ax.contour(ww, bb / 1000, np.log10(surface), levels=10, colors="white", linewidths=0.6, alpha=0.6)
    ax.plot(5, 0, "o", color=MUTED, markeredgecolor=INK, markersize=9)
    ax.annotate(f"poor guess\nloss {loss_poor / 1e6:,.0f} M", (5, 0), textcoords="offset points",
                xytext=(10, -34), fontsize=10, color=INK)
    ax.plot(w_fit, b_fit / 1000, "*", color=ORANGE, markeredgecolor=INK, markersize=16)
    ax.annotate(f"best fit\nloss {loss_best / 1e6:.2f} M", (w_fit, b_fit / 1000),
                textcoords="offset points", xytext=(10, 6), fontsize=10, color=INK)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("log₁₀\nMSE", fontsize=9.5, color=INK2)
    cbar.outline.set_visible(False)
    ax.set_xlabel("weight w")
    ax.set_ylabel("bias b (thousand $)")
    ax.set_title("Training = finding the lowest point")

    ax = fig.add_subplot(gs[1, 1])
    parts = list(w4 * x4) + [b4]
    names = ["12·2500\nsize", "1500·5\nrooms", "3000·2\nbaths", "10·1990\nyear", "bias\n500"]
    bottom = 0.0
    for k, (v, name) in enumerate(zip(parts, names)):
        ax.bar(k, v, bottom=bottom, width=0.62, color=BLUE if k < 4 else MUTED, zorder=2)
        ax.text(k, bottom + v + 1200, f"+{v:,.0f}", ha="center", fontsize=10)
        bottom += v
    ax.bar(5, bottom, width=0.62, color=ORANGE, zorder=2)
    ax.text(5, bottom + 1200, f"{bottom:,.0f}", ha="center", fontsize=10.5, weight="bold")
    ax.set_xticks(range(6), names + ["sum\nŷ"], fontsize=9.5)
    ax.set_ylim(0, 75000)
    ax.set_ylabel("contribution to ŷ ($)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Four features: the same sum, just longer")

    panel(fig.add_subplot(gs[1, 2]), "What comes next", [
        "Measure how wrong ŷ is: a loss function",
        "(squared error, as in least squares).",
        "",
        "Mean squared error on the three houses:",
        f"poor guess (w = 5, b = 0):  {loss_poor:,.0f}",
        f"best fit (w = 15, b = 3333):  {loss_best:,.0f}",
        "",
        "Minimize it with gradient descent;",
        "add an activation and stack perceptrons",
        "in layers to get a neural network.",
    ], spacing=0.083)
    save(fig, name="00_perceptron.png")


# ================================================================ notebook 01
def sheet_01_loss_function():
    w = 15.0
    pred = w * SIZE
    errors = PRICE - pred
    assert errors.tolist() == [5000, 0, 5000] and errors.sum() == 10000

    def total_loss(wv):
        wv = np.asarray(wv, float)
        return 0.5 * np.sum((PRICE[:, None] - wv[None] * SIZE[:, None]) ** 2, axis=0)

    assert total_loss([15.0])[0] == 25_000_000
    w_star = np.sum(SIZE * PRICE) / np.sum(SIZE ** 2)
    assert f"{w_star:.4f}" == "16.4286"
    slope_15 = -np.sum(SIZE * (PRICE - w * SIZE))
    assert slope_15 == -20_000_000
    loss_star = total_loss([w_star])[0]
    w_cancel = PRICE.sum() / SIZE.sum()            # the slope whose signed errors add to 0
    err_cancel = PRICE - w_cancel * SIZE
    assert abs(err_cancel.sum()) < 1e-6

    fig = new_figure("Square the error so that mistakes cannot cancel",
                     "L(y, ŷ) = ½(y − ŷ)² per point; the total loss L(w) = ½ Σ (yᵢ − w xᵢ)² "
                     f"is a parabola in w, lowest at w* = Σxy / Σx² = {w_star:.2f}.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(800, 3200, 100)
    ax.plot(xs, w * xs, color=BLUE, linewidth=2.2, label="model ŷ = 15·size")
    label_at = {1000: (1080, 11500, "left"), 2000: (2080, 27000, "left"), 3000: (2920, 48500, "right")}
    for s, y, yh in zip(SIZE, PRICE, pred):
        ax.plot([s, s], [yh, y], color=ORANGE, linewidth=2, linestyle="--")
        tx, ty, ha = label_at[int(s)]
        ax.text(tx, ty, f"y − ŷ = {y - yh:,.0f}", fontsize=10, va="center", ha=ha,
                color=ORANGE if y != yh else INK2)
    ax.scatter(SIZE, PRICE, s=70, color=INK, zorder=4, label="actual prices")
    ax.set_xlim(800, 3900)
    ax.set_ylim(0, 58000)
    ax.set_xlabel("size (sq ft)")
    ax.set_ylabel("price ($)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The error = vertical distance to the line")

    ax = fig.add_subplot(gs[0, 1])
    pos = np.arange(3)
    ax.bar(pos, err_cancel, width=0.6, color=[ORANGE if e > 0 else BLUE for e in err_cancel], zorder=2)
    for p, e in zip(pos, err_cancel):
        ax.text(p, e + (350 if e >= 0 else -350), f"{e:+,.0f}" if abs(e) > 0.5 else "0", ha="center",
                va="bottom" if e >= 0 else "top", fontsize=10.5)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(pos, [f"{s:.0f} sq ft" for s in SIZE])
    ax.set_ylim(-5200, 5200)
    ax.set_ylabel("signed error y − ŷ ($)")
    ax.grid(axis="y", zorder=0)
    ax.text(1, 4300, f"line ŷ = {w_cancel:.2f}·size:\nsigned errors sum to 0,\n"
                     f"squared loss = {0.5 * np.sum(err_cancel ** 2) / 1e6:.1f} M",
            ha="center", va="top", fontsize=10, color=INK)
    ax.set_title("The sign problem: +3333 and −3333 cancel")

    panel(fig.add_subplot(gs[0, 2]), "Why square, and why ½?", [
        "Squaring removes the sign: every point",
        "costs ≥ 0, and big misses cost much more.",
        "",
        r"$L(y,\hat{y}) = \frac{1}{2}(y-\hat{y})^2$",
        "",
        r"$\frac{dL}{d\hat{y}} = -(y-\hat{y})$   (the ½ eats the 2)",
        "",
        "Scaling a loss by any constant does",
        "not move its minimum: same best w, b.",
    ], spacing=0.092)

    ax = fig.add_subplot(gs[1, 0])
    w_grid = np.linspace(5, 25, 300)
    ax.plot(w_grid, total_loss(w_grid) / 1e6, color=BLUE, linewidth=2.2, label="L(w)")
    tw = np.linspace(12.2, 17.8, 10)
    ax.plot(tw, (total_loss([w])[0] + slope_15 * (tw - w)) / 1e6, color=ORANGE, linewidth=1.8,
            label="tangent at w = 15")
    ax.plot(w, total_loss([w])[0] / 1e6, "o", color=ORANGE, markersize=8, zorder=4)
    ax.plot(w_star, loss_star / 1e6, "*", color=AQUA, markeredgecolor=INK, markersize=16, zorder=4)
    ax.annotate(f"L'(15) = {slope_15 / 1e6:.0f} M < 0\n→ increase w", (w, total_loss([w])[0] / 1e6),
                xytext=(5.0, 75), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8))
    ax.annotate(f"w* = {w_star:.2f}, L'(w*) = 0", (w_star, loss_star / 1e6),
                xytext=(18.6, -6), va="center", fontsize=10, color=INK)
    ax.set_ylim(-20, 460)
    ax.set_xlabel("slope w (bias 0)")
    ax.set_ylabel("total loss L(w) (millions)")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Loss as a function of the parameter")

    ax = fig.add_subplot(gs[1, 1])
    loss15 = 0.5 * errors ** 2
    loss_s = 0.5 * (PRICE - w_star * SIZE) ** 2
    width = 0.38
    ax.bar(pos - width / 2, loss15 / 1e6, width=width - 0.04, color=MUTED, label="w = 15", zorder=2)
    ax.bar(pos + width / 2, loss_s / 1e6, width=width - 0.04, color=AQUA,
           label=f"w* = {w_star:.2f}", zorder=2)
    for p, a, c in zip(pos, loss15, loss_s):
        ax.text(p - width / 2, a / 1e6 + 0.3, f"{a / 1e6:.1f}", ha="center", fontsize=9.5)
        ax.text(p + width / 2, c / 1e6 + 0.3, f"{c / 1e6:.1f}", ha="center", fontsize=9.5)
    ax.set_xticks(pos, [f"{s:.0f} sq ft" for s in SIZE])
    ax.set_ylim(0, 17)
    ax.set_ylabel("½(y − ŷ)² (millions)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title(f"Totals: 25.0 M at w = 15, {loss_star / 1e6:.1f} M at w*")

    panel(fig.add_subplot(gs[1, 2]), "The goal", [
        "The loss is a function of the parameters:",
        r"$L(w) = \frac{1}{2}\,\Sigma_i\,(y_i - w\,x_i)^2$",
        "",
        "Set the derivative to zero:",
        r"$L'(w) = -\Sigma_i\, x_i\,(y_i - w\,x_i) = 0$",
        "",
        r"$\Rightarrow\; w^\ast = \Sigma\, x_i y_i \,/\, \Sigma\, x_i^2$" + f" = {w_star:.2f}",
        "",
        "w = 15 gives 25.0 M; the parabola has one",
        "minimum, so gradient descent walks",
        "downhill straight to it.",
    ], spacing=0.085)
    save(fig, name="01_loss_function.png")


# ================================================================ notebook 02
def sheet_02_gradient_descent():
    xs = (SIZE - SIZE.mean()) / SIZE.std()
    ys = (PRICE - PRICE.mean()) / PRICE.std()

    def train(alpha, epochs=200):
        w, b, path, losses = 0.0, 0.0, [(0.0, 0.0)], []
        for _ in range(epochs):
            error = ys - (w * xs + b)
            dw, db = -np.mean(xs * error), -np.mean(error)
            w -= alpha * dw
            b -= alpha * db
            losses.append(0.5 * np.mean(error ** 2))
            path.append((w, b))
        return w, b, np.array(path), np.array(losses)

    w_t, b_t, path, losses = train(0.3)
    _, _, _, losses_slow = train(0.05)
    assert np.abs(path[:, 1]).max() < 1e-12    # standardized data: b never leaves 0
    w_exact, b_exact = np.polyfit(xs, ys, 1)
    assert f"{w_t:.4f}" == f"{w_exact:.4f}" == "0.9820"
    assert f"{losses_slow[19]:.5f}" == "0.08651" and f"{losses[19]:.5f}" == "0.01786"

    X1, X2 = xs, (ROOMS - ROOMS.mean()) / ROOMS.std()
    w1, w2, b, alpha2, two_losses = 0.0, 0.0, 0.0, 0.2, []
    for _ in range(500):
        err = ys - (w1 * X1 + w2 * X2 + b)
        two_losses.append(0.5 * np.mean(err ** 2))
        w1 += alpha2 * np.mean(X1 * err)
        w2 += alpha2 * np.mean(X2 * err)
        b += alpha2 * np.mean(err)
    final_two = 0.5 * np.mean((ys - (w1 * X1 + w2 * X2 + b)) ** 2)
    assert (f"{w1:.4f}", f"{w2:.4f}", f"{final_two:.6f}") == ("-0.0600", "1.0529", "0.002334")

    fig = new_figure("Gradient descent trains the perceptron, one chain-rule step at a time",
                     "∂L/∂wᵢ = ∂L/∂ŷ · ∂ŷ/∂wᵢ = −xᵢ(y − ŷ),  ∂L/∂b = −(y − ŷ);   "
                     "update  wᵢ ← wᵢ + α xᵢ(y − ŷ),  b ← b + α(y − ŷ)")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "Chain rule: one shared factor", [
        r"$\hat{y} = w_1x_1 + w_2x_2 + b,\quad L = \frac{1}{2}(y-\hat{y})^2$",
        "",
        "",
        r"shared:  $\frac{\partial L}{\partial \hat{y}} = -(y-\hat{y})$",
        "",
        r"easy:  $\frac{\partial \hat{y}}{\partial w_1} = x_1,\;"
        r"\frac{\partial \hat{y}}{\partial w_2} = x_2,\;\frac{\partial \hat{y}}{\partial b} = 1$",
        "",
        "",
        "Multiply shared × easy for each parameter.",
        "Prediction too low (y − ŷ > 0)? Every",
        "weight moves up, in proportion to its input.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[0, 1])
    w_grid = np.linspace(-0.15, 1.6, 300)
    L_of_w = 0.5 * np.mean((ys[:, None] - w_grid[None] * xs[:, None]) ** 2, axis=0)
    ax.plot(w_grid, L_of_w, color=BLUE, linewidth=2.2, label="L(w), b = 0 throughout")
    for k in range(5):
        wk, wn = path[k, 0], path[k + 1, 0]
        lk = 0.5 * np.mean((ys - wk * xs) ** 2)
        ln = 0.5 * np.mean((ys - wn * xs) ** 2)
        gk = -np.mean(xs * (ys - wk * xs))
        if k < 2:
            ax.annotate(f"slope {gk:.3f} → Δw = {-0.3 * gk:+.3f}", (wk, lk), textcoords="offset points",
                        xytext=(14, 8), fontsize=10, color=INK2)
        ax.annotate("", xy=(wn, ln), xytext=(wk, lk),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.5, mutation_scale=11))
        ax.plot(wk, lk, "o", color=ORANGE, markeredgecolor=INK, markersize=7, zorder=4)
    ax.plot(w_t, losses[-1], "*", color=ORANGE, markeredgecolor=INK, markersize=15, zorder=5)
    ax.annotate(f"after 200 steps\nw = {w_t:.4f}, slope 0", (w_t, losses[-1]), xytext=(1.25, 0.42),
                fontsize=10, ha="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("w (standardized)")
    ax.set_ylabel("mean loss L")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("α = 0.3: each step is −α · slope")

    ax = fig.add_subplot(gs[0, 2])
    epochs = np.arange(1, 201)
    ax.plot(epochs, losses, color=ORANGE, linewidth=2.2, label="α = 0.3")
    ax.plot(epochs, losses_slow, color=AQUA, linewidth=2.2, label="α = 0.05")
    ax.axhline(losses[-1], color=INK2, linestyle=":", linewidth=1.2)
    ax.text(200, losses[-1] * 1.1, f"minimum {losses[-1]:.5f} (3 points, not a perfect line)",
            ha="right", fontsize=9.5, color=INK2)
    ax.annotate(f"epoch 20: {losses_slow[19]:.5f}\n(α = 0.3 is already at the minimum)",
                (20, losses_slow[19]), textcoords="offset points", xytext=(14, 12), fontsize=10,
                color=AQUA)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean loss (log scale)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("A smaller α gets there more slowly")

    ax = fig.add_subplot(gs[1, 0])
    line_x = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 50)
    for k, shade in zip([0, 1, 2, 4, 200], ["#d7d6d1", "#b9b8b2", MUTED, INK2, ORANGE]):
        wk, bk = path[k]
        ax.plot(line_x, wk * line_x + bk, color=shade, linewidth=2.2 if k == 200 else 1.6,
                label=f"after {k} step{'s' if k != 1 else ''}: w = {wk:.3f}")
    ax.scatter(xs, ys, s=70, color=BLUE, zorder=4, label="houses (standardized)")
    ax.set_xlabel("size (standardized)")
    ax.set_ylabel("price (standardized)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(-1.6, 2.4)
    ax.grid(True)
    ax.set_title("The line tilts into place")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(1, 501), two_losses, color=BLUE, linewidth=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean loss (log scale)")
    ax.grid(True)
    ax.text(490, two_losses[0] * 0.6,
            f"w₁ (size) = {w1:.4f}\nw₂ (rooms) = {w2:.4f}\nb = {abs(b):.4f}\nfinal loss = {final_two:.6f}",
            ha="right", va="top", fontsize=10.5, color=INK)
    ax.set_title("Two features (size, rooms), α = 0.2")

    panel(fig.add_subplot(gs[1, 2]), "What the run shows", [
        "Start at w = b = 0, average the per-house",
        "gradients, step 200 times with α = 0.3:",
        f"w = {w_t:.4f}, b = {abs(b_t):.4f}",
        f"= the exact least-squares fit ({w_exact:.4f}).",
        "",
        "Mean-loss gradient:",
        r"$\frac{\partial L}{\partial w} = -\frac{1}{n}\,\Sigma_i\, x_i\,(y_i-\hat{y}_i)$",
        "",
        "",
        "The same rules train w₁, w₂, b at once;",
        "with 3 houses rooms alone fit almost exactly.",
    ], spacing=0.083)
    save(fig, name="02_gradient_descent.png")
    return losses, losses_slow


# ================================================================ notebook 03
def sheet_03_classification():
    w, b = np.array([1.0, -1.0]), -0.5
    z = ALIEN_X @ w + b
    probs = sigmoid(z)
    assert [f"{p:.3f}" for p in probs] == ["0.924", "0.076", "0.076", "0.622"]
    p_new = sigmoid(np.array([4, 1]) @ w + b)
    assert f"{p_new:.3f}" == "0.924"

    fig = new_figure("A sigmoid turns the perceptron into a classifier",
                     "z = w₁x₁ + w₂x₂ + b,  ŷ = σ(z) = 1 / (1 + e⁻ᶻ) ∈ (0, 1) is P(happy); "
                     "the decision boundary σ = 0.5 is the straight line z = 0.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    GX, GY = np.meshgrid(np.linspace(-0.5, 4, 200), np.linspace(-0.5, 4, 200))
    prob_map(ax, GX, GY, sigmoid(w[0] * GX + w[1] * GY + b))
    class_scatter(ax, ALIEN_X, ALIEN_Y, labels=("happy", "sad"))
    ax.text(2.5, 1.45, "x₂ = x₁ − 0.5\n(σ = 0.5)", fontsize=10, rotation=0, color=INK, va="center")
    ax.text(3.3, 0.6, "happy side", fontsize=10, color=ORANGE, ha="center", weight="bold")
    ax.text(0.5, 3.55, "sad side", fontsize=10, color=BLUE, ha="center", weight="bold")
    ax.set_xlabel("# aack (x₁)")
    ax.set_ylabel("# beep (x₂)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("w = (1, −1), b = −0.5 splits the sentences")

    ax = fig.add_subplot(gs[0, 1])
    zz = np.linspace(-6, 6, 400)
    ax.plot(zz, sigmoid(zz), color=INK2, linewidth=2.2)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    tags = {2.5: "(3,0)", -2.5: "(0,2) and (1,3)", 0.5: "(2,1)"}
    for zi, pi, yi in zip(z, probs, ALIEN_Y):
        ax.plot(zi, pi, "o", color=ORANGE if yi else BLUE, markersize=10, markeredgecolor=INK, zorder=4)
    for zi, tag in tags.items():
        pi = sigmoid(zi)
        ax.annotate(f"{tag}: z = {zi:+.1f}, ŷ = {pi:.3f}", (zi, pi), textcoords="offset points",
                    xytext={2.5: (-12, 4), 0.5: (12, -10), -2.5: (12, 8)}[zi], fontsize=9.5,
                    ha="left" if zi < 2 else "right")
    ax.set_xlabel("z = x₁ − x₂ − 0.5")
    ax.set_ylabel("σ(z)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.set_title("Each sentence lands on the S-curve")

    panel(fig.add_subplot(gs[0, 2]), "Regression vs classification", [
        "Same summation, one new step:",
        "regression:  ŷ = z  (any number)",
        "classification:  ŷ = σ(z)  ∈ (0, 1)",
        "",
        r"$\sigma(z) = \frac{1}{1+e^{-z}}$",
        "",
        "z → −∞: σ → 0 (confidently sad)",
        "z = 0: σ = 0.5 (unsure)",
        "z → +∞: σ → 1 (confidently happy)",
        "",
        "Predict happy when ŷ ≥ 0.5, i.e. z ≥ 0.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    x1s = np.linspace(-1, 6, 300)
    z_line = w[0] * x1s + w[1] * 1 + b
    ax.plot(x1s, z_line, color=BLUE, linewidth=2.2, label="z: unbounded, a straight line")
    ax.plot(x1s, sigmoid(z_line), color=ORANGE, linewidth=2.2, label="σ(z): squashed into (0, 1)")
    ax.axhspan(0, 1, color=ORANGE_LIGHT, alpha=0.35, linewidth=0)
    ax.plot(2, sigmoid(0.5), "o", color=ORANGE, markeredgecolor=INK, markersize=9, zorder=4)
    ax.plot(4, p_new, "o", color=ORANGE, markeredgecolor=INK, markersize=9, zorder=4)
    ax.annotate(f"(2,1): {sigmoid(0.5):.3f}", (2, sigmoid(0.5)), textcoords="offset points",
                xytext=(-10, 14), fontsize=9.5, ha="right")
    ax.annotate(f"new (4,1): {p_new:.3f}", (4, p_new), textcoords="offset points",
                xytext=(0, -22), fontsize=9.5, ha="center")
    ax.set_xlabel("# aack (x₁), with 1 beep")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The raw sum vs the probability")

    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(4)
    ax.bar(pos, probs, width=0.6, color=[ORANGE if yi else BLUE for yi in ALIEN_Y], zorder=2)
    for p, pr in zip(pos, probs):
        ax.text(p, pr + 0.02, f"{pr:.3f}", ha="center", fontsize=10.5)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(2.5, 0.52, "0.5", fontsize=10, color=INK2, ha="center")
    ax.set_xticks(pos, [f"({a:.0f},{bp:.0f})\n{'happy' if yi else 'sad'}"
                        for (a, bp), yi in zip(ALIEN_X, ALIEN_Y)])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ŷ = P(happy)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("All four sentences classified correctly")

    panel(fig.add_subplot(gs[1, 2]), "Reading the weights", [
        "w₁ = 1 > 0: more aack → more happy.",
        "w₂ = −1 < 0: beep is a sad word.",
        "b = −0.5 shifts the boundary line.",
        "",
        "σ(z) = 0.5  ⟺  z = 0  ⟺",
        "x₁ − x₂ − 0.5 = 0, a straight line.",
        "",
        "These weights were picked by hand;",
        "training them needs a loss (log loss)",
        "and σ's derivative: the next notebooks.",
    ], spacing=0.083)
    save(fig, name="03_classification_perceptron.png")


# ================================================================ notebook 04
def sheet_04_sigmoid_derivative():
    def dsig(z):
        s = sigmoid(z)
        return s * (1 - s)

    def numeric(z, h=1e-6):
        return (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)

    checks = [-3.0, -1.0, 0.0, 1.0, 3.0]
    assert [f"{dsig(z):.6f}" for z in checks] == ["0.045177", "0.196612", "0.250000", "0.196612", "0.045177"]
    assert all(abs(numeric(z) - dsig(z)) < 1e-9 for z in checks)

    fig = new_figure("The sigmoid's derivative is built from the sigmoid itself",
                     "σ(z) = 1 / (1 + e⁻ᶻ)  ⇒  σ′(z) = e⁻ᶻ / (1 + e⁻ᶻ)² = σ(z)(1 − σ(z)):  "
                     "peak 0.25 at z = 0, nearly 0 in the flat tails.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    zz = np.linspace(-8, 8, 400)
    ax.plot(zz, sigmoid(zz), color=BLUE, linewidth=2.4, label="σ(z)")
    for z0, color in [(-3, AQUA), (0, ORANGE), (3, AQUA)]:
        t = np.linspace(z0 - 1.8, z0 + 1.8, 10)
        ax.plot(t, sigmoid(z0) + dsig(z0) * (t - z0), color=color, linewidth=1.8)
        ax.plot(z0, sigmoid(z0), "o", color=color, markeredgecolor=INK, markersize=8, zorder=4)
        ax.annotate(f"slope {dsig(z0):.3f}", (z0, sigmoid(z0)), textcoords="offset points",
                    xytext=(-80, 18) if z0 >= 0 else (8, -20), fontsize=10, color=color)
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("z")
    ax.set_ylabel("σ(z)")
    ax.grid(True)
    ax.set_title("Tangent slopes along the S-curve")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(zz, sigmoid(zz), color=BLUE, linewidth=1.6, alpha=0.5, label="σ(z)")
    ax.plot(zz, dsig(zz), color=ORANGE, linewidth=2.4, label="σ′(z) = σ(z)(1 − σ(z))")
    ax.plot(checks, [numeric(z) for z in checks], "s", color=INK, markersize=6,
            label="central difference, h = 10⁻⁶")
    ax.annotate("max 0.25 at z = 0", (0, 0.25), textcoords="offset points", xytext=(12, 12), fontsize=10)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("z")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("The derivative is a bell")

    panel(fig.add_subplot(gs[0, 2]), "The derivation in three moves", [
        r"1. Chain rule on $(1+e^{-z})^{-1}$:",
        r"$\sigma'(z) = -(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$",
        "",
        r"2. Add and subtract 1:  $e^{-z} = (1+e^{-z}) - 1$",
        r"$\sigma'(z) = \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^2}$",
        "",
        r"3. Factor:  $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$",
        "",
        "SymPy: σ′ − σ(1 − σ) simplifies to 0.",
    ], spacing=0.094, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    s = np.linspace(0, 1, 200)
    ax.plot(s, s * (1 - s), color=ORANGE, linewidth=2.4)
    for z0 in checks:
        ax.plot(sigmoid(z0), dsig(z0), "o", color=BLUE, markeredgecolor=INK, markersize=8, zorder=4)
        ax.annotate(f"z = {z0:+.0f}".replace("-", "−") if z0 else "z = 0", (sigmoid(z0), dsig(z0)),
                    textcoords="offset points",
                    xytext=(0, -18) if z0 else (0, 9), ha="center", fontsize=9.5)
    ax.set_ylim(0, 0.29)
    ax.set_xlabel("σ, the sigmoid's output")
    ax.set_ylabel("σ′ = σ(1 − σ)")
    ax.grid(True)
    ax.set_title("Know σ, get σ′ for free: a parabola")

    ax = fig.add_subplot(gs[1, 1])
    zpos = np.linspace(0, 12, 300)
    ax.plot(zpos, dsig(zpos), color=ORANGE, linewidth=2.4)
    for z0 in [0, 2, 5, 10]:
        ax.plot(z0, dsig(z0), "o", color=ORANGE, markeredgecolor=INK, markersize=8, zorder=4)
        ax.annotate(f"σ′({z0}) = {dsig(z0):.1e}" if z0 else "σ′(0) = 0.25", (z0, dsig(z0)),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("|z|")
    ax.set_ylabel("σ′(z) (log scale)")
    ax.grid(True)
    ax.set_title("Saturated tails: the gradient vanishes")

    panel(fig.add_subplot(gs[1, 2]), "Why this matters for training", [
        "Gradient descent multiplies by σ′ at",
        "every sigmoid it passes through.",
        "",
        "Cheap: reuse the stored output σ(z);",
        "no new exponentials to evaluate.",
        "",
        "Small: σ′ ≤ 0.25 everywhere, and at",
        f"|z| = 10 it is only {dsig(10):.1e}. A neuron",
        "stuck in a flat tail barely learns:",
        "the vanishing-gradient problem.",
    ], spacing=0.083)
    save(fig, name="04_sigmoid_derivative.png")


# ================================================================ notebook 05
def sheet_05_log_loss():
    w1, w2, b, x1, x2 = 4.5, 1.5, 2.0, 1, 3
    z_ex = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z_ex)
    assert z_ex == 11.0 and f"{y_hat:.6f}" == "0.999983"

    def log_loss(y, p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)

    L0, L1 = log_loss(0, y_hat), log_loss(1, y_hat)
    assert f"{L0:.4f}" == "11.0000" and f"{L1:.6f}" == "0.000017"
    q = np.linspace(0.01, 0.99, 999)
    nll = -(7 * np.log(q) + 3 * np.log(1 - q))
    assert f"{q[np.argmin(nll)]:.3f}" == "0.700"

    fig = new_figure("Log loss: small when ŷ agrees with y, huge for confident mistakes",
                     "L(y, ŷ) = −y log ŷ − (1 − y) log(1 − ŷ):  y = 1 → −log ŷ,  y = 0 → −log(1 − ŷ); "
                     "it is the negative log-likelihood of the label.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    p = np.linspace(0.001, 0.999, 400)
    ax.plot(p, log_loss(1, p), color=ORANGE, linewidth=2.2, label="y = 1 (happy): −log ŷ")
    ax.plot(p, log_loss(0, p), color=BLUE, linewidth=2.2, label="y = 0 (sad): −log(1 − ŷ)")
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("prediction ŷ")
    ax.set_ylabel("log loss L")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Each label punishes the far end")

    ax = fig.add_subplot(gs[0, 1])
    zz = np.linspace(-4, 13, 400)
    ax.plot(zz, np.log1p(np.exp(zz)), color=BLUE, linewidth=2.2, label="log loss, y = 0")
    ax.plot(zz, 0.5 * sigmoid(zz) ** 2, color=MUTED, linewidth=2.2, linestyle="--",
            label="½(y − ŷ)², y = 0")
    ax.plot(z_ex, L0, "o", color=ORANGE, markeredgecolor=INK, markersize=10, zorder=4)
    ax.annotate(f"the example: z = 11,\nŷ = {y_hat:.6f}, L = {L0:.4f}", (z_ex, L0),
                xytext=(-3.8, 9.6), ha="left", va="top", fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.annotate("squared error caps at 0.5", (12, 0.5), textcoords="offset points", xytext=(-4, 12),
                ha="right", fontsize=10, color=INK2)
    ax.set_xlabel("score z  (ŷ = σ(z))")
    ax.set_ylabel("loss")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Truth is sad, model says happy")

    panel(fig.add_subplot(gs[0, 2]), "The worked example", [
        "\"aack beep beep beep\": x₁ = 1, x₂ = 3",
        "weights w₁ = 4.5, w₂ = 1.5, b = 2",
        "z = 4.5 + 4.5 + 2 = 11",
        f"ŷ = σ(11) = {y_hat:.6f}: sure it's happy",
        "",
        f"true y = 0 (sad):  L = −log(1 − ŷ) = {L0:.4f}",
        f"if y = 1 (happy):  L = −log ŷ = {L1:.6f}",
        "",
        "Large loss → large correction to",
        "w₁, w₂, b by gradient descent.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    s = sigmoid(zz)
    ax.plot(zz, s, color=BLUE, linewidth=2.2, label="log loss: dL/dz = ŷ − y")
    ax.plot(zz, s * s * (1 - s), color=MUTED, linewidth=2.2, linestyle="--",
            label="squared error: dL/dz = (ŷ − y)·ŷ(1 − ŷ)")
    ax.plot(z_ex, sigmoid(z_ex), "o", color=ORANGE, markeredgecolor=INK, markersize=9, zorder=4)
    sq_grad = sigmoid(z_ex) ** 2 * (1 - sigmoid(z_ex))
    ax.text(12.8, 0.75, f"at z = 11: {sigmoid(z_ex):.5f} vs {sq_grad:.1e}", ha="right", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("score z  (true y = 0)")
    ax.set_ylabel("gradient dL/dz")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True)
    ax.set_title("Only the log loss pushes back hard")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(q, nll, color=BLUE, linewidth=2.2)
    ax.axvline(0.7, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.plot(0.7, nll[np.argmin(nll)], "o", color=ORANGE, markeredgecolor=INK, markersize=9, zorder=4)
    ax.text(0.72, 17, f"minimum at q = 0.7\nloss = {nll.min():.3f}", ha="left", fontsize=10)
    ax.set_ylim(0, 25)
    ax.set_xlabel("coin probability q")
    ax.set_ylabel("−(7 log q + 3 log(1 − q))")
    ax.grid(True)
    ax.set_title("7 heads, 3 tails: log loss picks q = 7/10")

    panel(fig.add_subplot(gs[1, 2]), "Why log loss for classification", [
        "ŷ is a probability: ŷ = 0.8 means an",
        "80% chance the sentence is happy.",
        "",
        "Summing log losses over the data is the",
        "negative log-likelihood, so minimizing it",
        "finds the most likely model (as for the coin).",
        "",
        "Squared error on a sigmoid saturates:",
        "confident mistakes get almost no gradient.",
        "Next: σ′ and log loss cancel to ŷ − y.",
    ], spacing=0.083)
    save(fig, name="05_log_loss.png")


# ================================================================ notebook 06
def sheet_06_classification_derivatives():
    alpha, epochs = 0.5, 3000
    w, bias = np.zeros(2), 0.0
    losses, hist = [], [(w.copy(), bias)]
    for _ in range(epochs):
        y_hat = sigmoid(ALIEN_X @ w + bias)
        error = ALIEN_Y - y_hat
        w += alpha * (ALIEN_X.T @ error) / len(ALIEN_Y)
        bias += alpha * np.mean(error)
        losses.append(log_loss_mean(ALIEN_Y, y_hat))
        hist.append((w.copy(), bias))
    assert (f"{w[0]:.3f}", f"{w[1]:.3f}", f"{bias:.3f}", f"{losses[-1]:.5f}") == \
        ("5.182", "-4.271", "0.485", "0.00062")
    preds = sigmoid(ALIEN_X @ w + bias)
    assert [f"{p:.3f}" for p in preds] == ["1.000", "0.000", "0.001", "0.999"]

    fig = new_figure("Sigmoid + log loss: the messy factors cancel to −(y − ŷ)x",
                     "∂L/∂wᵢ = ∂L/∂ŷ · ∂ŷ/∂wᵢ = −(y − ŷ)/(ŷ(1 − ŷ)) · ŷ(1 − ŷ)xᵢ = −(y − ŷ)xᵢ,  "
                     "∂L/∂b = −(y − ŷ): the same form as regression.")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "The three pieces", [
        r"$L = -y\log\hat{y} - (1-y)\log(1-\hat{y})$",
        "",
        r"$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}"
        r" = -\frac{y-\hat{y}}{\hat{y}(1-\hat{y})}$",
        "",
        r"$\frac{\partial \hat{y}}{\partial w_i} = \sigma'(z)\,x_i = \hat{y}(1-\hat{y})\,x_i$",
        "",
        r"product:  $\frac{\partial L}{\partial w_i} = -(y-\hat{y})\,x_i$",
        "",
        "Gradient size ∝ error: tiny when ŷ ≈ y,",
        "large when the prediction is far off.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[0, 1])
    yh = np.linspace(0.02, 0.98, 300)
    ax.plot(yh, -(1 - yh) / (yh * (1 - yh)), color=BLUE, linewidth=2.2,
            label="∂L/∂ŷ = −1/ŷ  (blows up)")
    ax.plot(yh, yh * (1 - yh), color=AQUA, linewidth=2.2, label="∂ŷ/∂z = ŷ(1 − ŷ)  (vanishes)")
    ax.plot(yh, yh - 1, color=ORANGE, linewidth=2.8, label="product ∂L/∂z = ŷ − 1  (a line)")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(-6, 1)
    ax.set_xlabel("prediction ŷ   (true label y = 1)")
    ax.set_ylabel("value")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)
    ax.set_title("The cancellation, seen as curves")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.arange(1, epochs + 1), losses, color=ORANGE, linewidth=2.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.annotate(f"start: log 2 = {losses[0]:.3f}", (1, losses[0]), textcoords="offset points",
                xytext=(18, -6), fontsize=10)
    ax.annotate(f"epoch 3000: {losses[-1]:.5f}", (epochs, losses[-1]), textcoords="offset points",
                xytext=(-6, 14), ha="right", fontsize=10)
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("mean log loss (log scale)")
    ax.grid(True)
    ax.set_title("Training on the alien sentences, α = 0.5")

    ax = fig.add_subplot(gs[1, 0])
    GX, GY = np.meshgrid(np.linspace(-0.5, 4, 200), np.linspace(-0.5, 4, 200))
    prob_map(ax, GX, GY, sigmoid(w[0] * GX + w[1] * GY + bias))
    x_line = np.linspace(-0.5, 4, 50)
    for k, shade in [(1, "#b9b8b2"), (10, MUTED), (100, INK2)]:
        wk, bk = hist[k]
        ax.plot(x_line, -(wk[0] * x_line + bk) / wk[1], color=shade, linewidth=1.4, linestyle="--",
                label=f"after {k} step{'s' if k > 1 else ''}")
    class_scatter(ax, ALIEN_X, ALIEN_Y, labels=("happy", "sad"))
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("# aack (x₁)")
    ax.set_ylabel("# beep (x₂)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Boundary after 3000 steps (solid)")

    ax = fig.add_subplot(gs[1, 1])
    steps = np.arange(epochs + 1)
    W = np.array([h[0] for h in hist])
    B = np.array([h[1] for h in hist])
    ax.plot(steps, W[:, 0], color=ORANGE, linewidth=2.2, label=f"w₁ (aack) → {w[0]:.3f}")
    ax.plot(steps, W[:, 1], color=BLUE, linewidth=2.2, label=f"w₂ (beep) → {w[1]:.3f}".replace("-", "−"))
    ax.plot(steps, B, color=INK2, linewidth=2.0, label=f"b → {bias:.3f}")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameter value")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=9.5)
    ax.grid(True)
    ax.set_title("Weights grow while (y − ŷ) shrinks")

    panel(fig.add_subplot(gs[1, 2]), "Update rules and the result", [
        "w₁ ← w₁ + α (y − ŷ) x₁",
        "w₂ ← w₂ + α (y − ŷ) x₂",
        "b ← b + α (y − ŷ)     (averaged over data)",
        "",
        f"After 3000 steps from 0: w₁ = {w[0]:.3f},",
        f"w₂ = {w[1]:.3f}, b = {bias:.3f}, loss {losses[-1]:.5f}.",
        "ŷ = 1.000, 0.000, 0.001, 0.999: all four",
        "sentences right, and confidently so.",
        "",
        "Chained across layers: backpropagation.",
    ], spacing=0.083)
    save(fig, name="06_classification_derivatives.png")


# ================================================================ notebook 07
XOR_PARAMS = dict(w11=20, w21=20, b1=-10, w12=20, w22=20, b2=-30, w1=20, w2=-20, b=-10)


def two_layer(x1, x2, P):
    a1 = sigmoid(x1 * P['w11'] + x2 * P['w21'] + P['b1'])
    a2 = sigmoid(x1 * P['w12'] + x2 * P['w22'] + P['b2'])
    return sigmoid(a1 * P['w1'] + a2 * P['w2'] + P['b']), a1, a2


def network_diagram(ax, title, weights=None):
    diagram_axes(ax, (-1.2, 7.4), (-0.4, 4.6), title)
    pos = {"x1": (0.4, 3.4), "x2": (0.4, 0.8), "a1": (3.4, 3.4), "a2": (3.4, 0.8), "y": (6.4, 2.1)}
    for k, sym in [("x1", "x₁"), ("x2", "x₂")]:
        node(ax, *pos[k], sym, BLUE)
    node(ax, *pos["a1"], "a₁", ORANGE)
    node(ax, *pos["a2"], "a₂", AQUA)
    node(ax, *pos["y"], "ŷ", INK2)
    w = weights or {}
    edge(ax, pos["x1"], pos["a1"], label=w.get("w11"), frac=0.42, offset=(0, 0.2))
    edge(ax, pos["x2"], pos["a1"], label=w.get("w21"), frac=0.22, offset=(-0.15, 0.0))
    edge(ax, pos["x1"], pos["a2"], label=w.get("w12"), frac=0.22, offset=(-0.15, 0.0))
    edge(ax, pos["x2"], pos["a2"], label=w.get("w22"), frac=0.42, offset=(0, -0.2))
    edge(ax, pos["a1"], pos["y"], label=w.get("w1"), frac=0.45, offset=(0.1, 0.22))
    edge(ax, pos["a2"], pos["y"], label=w.get("w2"), frac=0.45, offset=(0.1, -0.22))
    for k, label in [("a1", w.get("b1")), ("a2", w.get("b2")), ("y", w.get("b"))]:
        if label:
            x, y = pos[k]
            ax.text(x, y + (0.5 if k != "a2" else -0.55), label, ha="center", va="center",
                    fontsize=10, color=INK2)
    for x, name in [(0.4, "input"), (3.4, "hidden"), (6.4, "output")]:
        ax.text(x, 4.45, name, ha="center", fontsize=10.5, weight="bold", color=INK2)


def sheet_07_neural_network():
    P = XOR_PARAMS
    outs = [two_layer(x1, x2, P) for x1, x2 in XOR_X]
    assert [f"{o[0]:.3f}" for o in outs] == ["0.000", "1.000", "1.000", "0.000"]
    a1s = np.array([o[1] for o in outs])
    a2s = np.array([o[2] for o in outs])

    fig = new_figure("A hidden layer combines two straight lines into a non-linear boundary",
                     "a₁ = σ(w₁₁x₁ + w₂₁x₂ + b₁),  a₂ = σ(w₁₂x₁ + w₂₂x₂ + b₂),  ŷ = σ(w₁a₁ + w₂a₂ + b): "
                     "OR and NOT AND gives XOR, which no single line can split.")
    gs = grid(fig, top=0.84, wspace=0.3)

    network_diagram(fig.add_subplot(gs[0, 0]), "Perceptrons in layers",
                    dict(w11="20", w21="20", w12="20", w22="20", w1="20", w2="−20",
                         b1="b₁ = −10", b2="b₂ = −30", b="b = −10"))

    GX, GY = np.meshgrid(np.linspace(-0.3, 1.3, 220), np.linspace(-0.3, 1.3, 220))
    GZ, A1, A2 = two_layer(GX, GY, P)
    for col, (A, title, eq) in enumerate([(A1, "Hidden a₁ ≈ OR", "x₁ + x₂ = 0.5"),
                                          (A2, "Hidden a₂ ≈ AND", "x₁ + x₂ = 1.5")], start=1):
        ax = fig.add_subplot(gs[0, col])
        prob_map(ax, GX, GY, A)
        class_scatter(ax, XOR_X, XOR_Y)
        ax.text(1.27, -0.25, f"line {eq}", ha="right", fontsize=10, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=2))
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(title)

    ax = fig.add_subplot(gs[1, 0])
    prob_map(ax, GX, GY, GZ)
    class_scatter(ax, XOR_X, XOR_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="center", fontsize=9, bbox_to_anchor=(0.5, 0.5))
    ax.set_title("Output ŷ: a band between the lines")

    ax = fig.add_subplot(gs[1, 1])
    a = np.linspace(-0.1, 1.1, 200)
    AA1, AA2 = np.meshgrid(a, a)
    prob_map(ax, AA1, AA2, sigmoid(20 * AA1 - 20 * AA2 - 10))
    ax.scatter(a1s[XOR_Y == 1], a2s[XOR_Y == 1], s=170, color=ORANGE, edgecolor=INK, zorder=4)
    ax.scatter(a1s[XOR_Y == 0], a2s[XOR_Y == 0], s=130, color=BLUE, edgecolor=INK, zorder=4)
    ax.annotate("(0,0)", (a1s[0], a2s[0]), textcoords="offset points", xytext=(10, 6), fontsize=10)
    ax.annotate("(0,1) and (1,0)", (a1s[1], a2s[1]), textcoords="offset points", xytext=(-14, -1),
                ha="right", va="top", fontsize=10)
    ax.annotate("(1,1)", (a1s[3], a2s[3]), textcoords="offset points", xytext=(-10, -16),
                ha="right", fontsize=10)
    ax.text(0.5, 0.08, "20a₁ − 20a₂ − 10 = 0", rotation=45, fontsize=10, ha="center", color=INK)
    ax.set_aspect("equal")
    ax.set_xlabel("a₁ (hidden OR)")
    ax.set_ylabel("a₂ (hidden AND)")
    ax.set_title("In hidden space, one line suffices")

    rows = [f"({x1:.0f},{x2:.0f})     {o[1]:.3f}   {o[2]:.3f}   {o[0]:.3f}    {int(t)}"
            for (x1, x2), o, t in zip(XOR_X, outs, XOR_Y)]
    panel(fig.add_subplot(gs[1, 2]), "Forward pass on XOR", [
        "`input      a₁      a₂      ŷ      target`",
        *[f"`{r}`" for r in rows],
        "",
        "One perceptron draws one straight line;",
        "XOR's 1s sit on a diagonal, so it fails.",
        "The hidden layer re-maps the inputs so",
        "the output's single line can split them.",
        "Loss: log loss, as for one perceptron.",
    ], spacing=0.083)
    save(fig, name="07_neural_network_xor.png")


# ================================================================ notebook 08
def nn_forward(X, P):
    x1, x2 = X[:, 0], X[:, 1]
    a1 = sigmoid(x1 * P['w11'] + x2 * P['w21'] + P['b1'])
    a2 = sigmoid(x1 * P['w12'] + x2 * P['w22'] + P['b2'])
    return sigmoid(a1 * P['w1'] + a2 * P['w2'] + P['b']), a1, a2


def nn_gradients(X, Y, P):
    x1, x2 = X[:, 0], X[:, 1]
    y_hat, a1, a2 = nn_forward(X, P)
    e = Y - y_hat
    g = {'w1': -np.mean(e * a1), 'w2': -np.mean(e * a2), 'b': -np.mean(e)}
    c1 = -e * P['w1'] * a1 * (1 - a1)
    c2 = -e * P['w2'] * a2 * (1 - a2)
    g.update(w11=np.mean(c1 * x1), w21=np.mean(c1 * x2), b1=np.mean(c1),
             w12=np.mean(c2 * x1), w22=np.mean(c2 * x2), b2=np.mean(c2))
    return g


def nn_train(seed, alpha=3.0, epochs=8000):
    rng = np.random.default_rng(seed)
    P = {k: rng.normal() for k in NN_KEYS}
    losses = []
    for _ in range(epochs):
        g = nn_gradients(XOR_X, XOR_Y, P)
        for k in P:
            P[k] -= alpha * g[k]
        losses.append(log_loss_mean(XOR_Y, nn_forward(XOR_X, P)[0]))
    return P, np.array(losses)


def sheet_08_backprop_xor():
    rng = np.random.default_rng(1)
    P0 = {k: rng.normal() for k in NN_KEYS}
    analytic = nn_gradients(XOR_X, XOR_Y, P0)
    h = 1e-6
    numeric = {}
    for k in P0:
        Pp, Pm = dict(P0), dict(P0)
        Pp[k] += h
        Pm[k] -= h
        numeric[k] = (log_loss_mean(XOR_Y, nn_forward(XOR_X, Pp)[0])
                      - log_loss_mean(XOR_Y, nn_forward(XOR_X, Pm)[0])) / (2 * h)
    assert f"{analytic['w1']:.6f}" == "0.050078" and f"{analytic['b']:.6f}" == "0.076061"

    runs = [nn_train(seed) for seed in range(10)]
    finals = np.array([r[1][-1] for r in runs])
    best_seed = int(np.argmin(finals))      # the notebook keeps the first strictly-lower run
    P_best, best_losses = runs[best_seed]
    solved = int((finals < 0.05).sum())
    assert f"{best_losses[-1]:.5f}" == "0.00039" and solved == 8
    stuck_seed = int(np.argmax(finals))
    P_stuck = runs[stuck_seed][0]
    stuck_preds = nn_forward(XOR_X, P_stuck)[0]

    fig = new_figure("Backpropagation: a long chain rule through the layers",
                     "∂L/∂w₁₁ = ∂z₁/∂w₁₁ · ∂a₁/∂z₁ · ∂z/∂a₁ · ∂ŷ/∂z · ∂L/∂ŷ = "
                     "−(y − ŷ) · w₁ · a₁(1 − a₁) · x₁;  output layer: ∂L/∂w₁ = −(y − ŷ) a₁")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "Every factor is easy", [
        r"$L \leftarrow \hat{y} \leftarrow z \leftarrow a_1 \leftarrow z_1 \leftarrow w_{11}$",
        "",
        r"$\partial z_1/\partial w_{11} = x_1$,   $\partial a_1/\partial z_1 = a_1(1-a_1)$",
        r"$\partial z/\partial a_1 = w_1$,   $\partial \hat{y}/\partial z = \hat{y}(1-\hat{y})$",
        r"$\partial L/\partial \hat{y} = -(y-\hat{y})\,/\,\hat{y}(1-\hat{y})$",
        "",
        "ŷ(1 − ŷ) cancels again. The error (y − ŷ)",
        "travels back through w₁ and σ′ = a₁(1 − a₁);",
        "b₁ is the same without the x₁.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[0, 1])
    pos = np.arange(len(NN_KEYS))
    ax.bar(pos - 0.2, [analytic[k] for k in NN_KEYS], width=0.36, color=BLUE, label="backprop", zorder=2)
    ax.bar(pos + 0.2, [numeric[k] for k in NN_KEYS], width=0.36, color=ORANGE,
           label="finite difference, h = 10⁻⁶", zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    labels = ["w₁₁", "w₂₁", "b₁", "w₁₂", "w₂₂", "b₂", "w₁", "w₂", "b"]
    ax.set_xticks(pos, labels)
    ax.set_ylabel("∂L/∂parameter")
    ax.set_ylim(-0.015, 0.095)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    max_diff = max(abs(analytic[k] - numeric[k]) for k in NN_KEYS)
    ax.text(8.4, 0.087, f"max difference {max_diff:.1e}", ha="right", fontsize=10, color=INK2)
    ax.set_title("Gradient check at random weights (seed 1)")

    ax = fig.add_subplot(gs[0, 2])
    epochs = np.arange(1, 8001)
    for seed, (_, ls) in enumerate(runs):
        if seed == best_seed:
            continue
        color = BLUE_LIGHT if ls[-1] < 0.05 else BLUE
        ax.plot(epochs, ls, color=color, linewidth=1.4 if ls[-1] < 0.05 else 1.8)
    ax.plot(epochs, best_losses, color=ORANGE, linewidth=2.4, label=f"best (seed {best_seed})")
    ax.plot([], [], color=BLUE_LIGHT, linewidth=1.6, label=f"other solved runs ({solved - 1})")
    ax.plot([], [], color=BLUE, linewidth=1.8, label=f"stuck runs ({10 - solved})")
    ax.axhline(0.05, color=INK2, linestyle=":", linewidth=1.2)
    ax.text(1.2, 0.058, "solved: loss < 0.05", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("mean log loss (log scale)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)
    ax.set_title(f"10 random starts, α = 3: {solved}/10 solve XOR")

    GX, GY = np.meshgrid(np.linspace(-0.3, 1.3, 200), np.linspace(-0.3, 1.3, 200))
    flat = np.column_stack([GX.ravel(), GY.ravel()])
    for col, (P, title) in enumerate([(P_best, f"Learned from scratch: loss {best_losses[-1]:.5f}"),
                                      (P_stuck, f"Stuck (seed {stuck_seed}): loss {finals[stuck_seed]:.3f}")]):
        ax = fig.add_subplot(gs[1, col])
        prob_map(ax, GX, GY, nn_forward(flat, P)[0].reshape(GX.shape))
        class_scatter(ax, XOR_X, XOR_Y)
        if col == 1:
            for (x1, x2), pr in zip(XOR_X, stuck_preds):
                ax.annotate(f"ŷ = {pr:.2f}", (x1, x2), textcoords="offset points",
                            xytext=(0, 14), ha="center", fontsize=9.5,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1))
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(title)

    panel(fig.add_subplot(gs[1, 2]), "Update rules (gradient descent)", [
        "Output layer, same as one perceptron:",
        "w₁ ← w₁ + α (y − ŷ) a₁,   b ← b + α (y − ŷ)",
        "",
        "Hidden layer, one more link in the chain:",
        "w₁₁ ← w₁₁ + α (y − ŷ) w₁ a₁(1 − a₁) x₁",
        "b₁ ← b₁ + α (y − ŷ) w₁ a₁(1 − a₁)",
        "",
        "The loss surface is not convex: some",
        "starts end in a local minimum, so try",
        "several random starts and keep the best.",
    ], spacing=0.083)
    save(fig, name="08_backprop_xor.png")


# ================================================================ notebook 09
def mlp_forward(X, Ws, bs):
    a, acts = X, [X]
    for W, b in zip(Ws, bs):
        a = sigmoid(a @ W + b)
        acts.append(a)
    return acts[-1], acts


def mlp_loss(Y, P, eps=1e-12):
    Y = Y.ravel()
    P = np.clip(P.ravel(), eps, 1 - eps)
    return np.mean(-Y * np.log(P) - (1 - Y) * np.log(1 - P))


def mlp_backprop(X, Y, Ws, bs):
    n = len(Y)
    _, acts = mlp_forward(X, Ws, bs)
    delta = (acts[-1] - Y.reshape(-1, 1)) / n
    gWs, gbs = [None] * len(Ws), [None] * len(bs)
    for l in reversed(range(len(Ws))):
        gWs[l] = acts[l].T @ delta
        gbs[l] = delta.sum(axis=0, keepdims=True)
        if l > 0:
            a_prev = acts[l]
            delta = (delta @ Ws[l].T) * a_prev * (1 - a_prev)
    return gWs, gbs


def sheet_09_deep_backprop():
    # gradient check, in the notebook's draw order
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(5, 2))
    Yc = np.array([0, 1, 1, 0, 1], dtype=float)
    sizes = [2, 3, 3, 1]
    Wc = [rng.normal(size=(sizes[i], sizes[i + 1])) for i in range(3)]
    bc = [np.zeros((1, sizes[i + 1])) for i in range(3)]
    gWc, _ = mlp_backprop(Xc, Yc, Wc, bc)
    h = 1e-6
    analytic, numeric = [], []
    for i in range(3):
        for j in range(3):
            Wp = [w.copy() for w in Wc]
            Wm = [w.copy() for w in Wc]
            Wp[1][i, j] += h
            Wm[1][i, j] -= h
            numeric.append((mlp_loss(Yc, mlp_forward(Xc, Wp, bc)[0])
                            - mlp_loss(Yc, mlp_forward(Xc, Wm, bc)[0])) / (2 * h))
            analytic.append(gWc[1][i, j])
    max_err = max(abs(a - b) for a, b in zip(analytic, numeric))
    assert f"{max_err:.2e}" == "7.93e-11"

    # the circle dataset and the 2-8-8-1 network
    rng = np.random.default_rng(2)
    N = 400
    X = rng.uniform(-1, 1, size=(N, 2))
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5).astype(float)
    sizes = [2, 8, 8, 1]
    W_init = [rng.normal(size=(sizes[i], sizes[i + 1])) * 1.0 for i in range(3)]
    b_init = [np.zeros((1, sizes[i + 1])) for i in range(3)]

    def train(alpha, epochs=4000):
        Ws = [w.copy() for w in W_init]
        bs = [b.copy() for b in b_init]
        losses = []
        for _ in range(epochs):
            gWs, gbs = mlp_backprop(X, Y, Ws, bs)
            for l in range(3):
                Ws[l] -= alpha * gWs[l]
                bs[l] -= alpha * gbs[l]
            losses.append(mlp_loss(Y, mlp_forward(X, Ws, bs)[0]))
        return Ws, bs, np.array(losses)

    Ws, bs, losses = train(1.0)
    acc = ((mlp_forward(X, Ws, bs)[0].ravel() >= 0.5).astype(float) == Y).mean()
    assert f"{losses[-1]:.4f}" == "0.0142" and f"{acc * 100:.1f}" == "99.8"
    other = {a: train(a)[2] for a in (0.1, 3.0)}

    fig = new_figure("Deeper networks: carry the error δ backward and reuse it",
                     "δ⁽³⁾ = ŷ − y,   ∂L/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾ᵀ δ⁽ˡ⁾,   δ⁽ˡ⁻¹⁾ = (δ⁽ˡ⁾ W⁽ˡ⁾ᵀ) ⊙ a⁽ˡ⁻¹⁾(1 − a⁽ˡ⁻¹⁾): "
                     "one backward sweep gives every gradient.")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "Store and reuse", [
        r"$\delta^{(3)} = \frac{\partial \hat{y}}{\partial z^{(3)}}\,\frac{\partial L}{\partial \hat{y}}$",
        "",
        "",
        r"$\frac{\partial L}{\partial w^{(3)}} = \frac{\partial z^{(3)}}{\partial w^{(3)}}\,\delta^{(3)}$",
        "",
        "",
        r"$\frac{\partial L}{\partial w^{(2)}} = \frac{\partial z^{(2)}}{\partial w^{(2)}}"
        r"\frac{\partial a^{(2)}}{\partial z^{(2)}}\frac{\partial z^{(3)}}{\partial a^{(2)}}\,\delta^{(3)}$",
        "",
        "",
        "The deeper chain ends in the piece already",
        "computed for layer 3. Work backward,",
        "keep δ, and each layer costs about one",
        "matrix product: no chain is recomputed.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(analytic, numeric, "o", color=ORANGE, markeredgecolor=INK, markersize=9, zorder=3)
    lim = max(abs(np.array(analytic))) * 1.2
    ax.plot([-lim, lim], [-lim, lim], color=INK2, linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("backprop ∂L/∂W⁽²⁾ entry")
    ax.set_ylabel("finite difference")
    ax.text(lim * 0.95, -lim * 0.85, f"max |difference|\n= {max_err:.2e}", ha="right", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Check: 2→3→3→1 net, layer-2 weights")

    ax = fig.add_subplot(gs[0, 2])
    GX, GY = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    flat = np.column_stack([GX.ravel(), GY.ravel()])
    prob_map(ax, GX, GY, mlp_forward(flat, Ws, bs)[0].reshape(GX.shape))
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=9, color=ORANGE, zorder=3, label="inside (class 1)")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=9, color=BLUE, zorder=3, label="outside (class 0)")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.sqrt(0.5) * np.cos(t), np.sqrt(0.5) * np.sin(t), color=INK2, linestyle=":",
            linewidth=1.5, label="true circle x₁² + x₂² = 0.5")
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper center", fontsize=8.5, bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=0.8,
              handletextpad=0.3)
    ax.set_title(f"2→8→8→1 learns a circle: {acc * 100:.1f}% correct")

    ax = fig.add_subplot(gs[1, 0])
    epochs = np.arange(1, 4001)
    ax.plot(epochs, other[0.1], color=AQUA, linewidth=2, label=f"α = 0.1 → {other[0.1][-1]:.4f}")
    ax.plot(epochs, losses, color=ORANGE, linewidth=2.4, label=f"α = 1 (notebook) → {losses[-1]:.4f}")
    ax.plot(epochs, other[3.0], color=BLUE, linewidth=2, label=f"α = 3 → {other[3.0][-1]:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean log loss (log scale)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Loss during training, three learning rates")

    ax = fig.add_subplot(gs[1, 1])
    diagram_axes(ax, (-0.6, 6.6), (-1.3, 5.3), "Forward stores a, backward reuses δ")
    cols = [(0.3, sizes[0], BLUE, "x"), (2.2, sizes[1], AQUA, "a⁽¹⁾"), (4.1, sizes[2], AQUA, "a⁽²⁾"),
            (6.0, sizes[3], ORANGE, "ŷ")]
    ys_of = [np.linspace(2.0 - 0.5 * (n - 1) / 2, 2.0 + 0.5 * (n - 1) / 2, n) for _, n, _, _ in cols]
    for (x0, _, _, _), (x1, _, _, _), y0s, y1s in zip(cols[:-1], cols[1:], ys_of[:-1], ys_of[1:]):
        for ya in y0s:
            for yb in y1s:
                ax.plot([x0, x1], [ya, yb], color=MUTED, linewidth=0.4, alpha=0.5, zorder=1)
    for (x, n, color, name), yv in zip(cols, ys_of):
        for yy in yv:
            ax.add_patch(Circle((x, yy), 0.15, facecolor=color, edgecolor="none", zorder=3))
        ax.text(x, 4.1, name, ha="center", fontsize=11, color=INK)
    ax.annotate("", xy=(6.0, 4.75), xytext=(0.3, 4.75),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.8, mutation_scale=14))
    ax.text(3.15, 5.0, "forward: a⁽ˡ⁾ = σ(a⁽ˡ⁻¹⁾W⁽ˡ⁾ + b⁽ˡ⁾), keep every a", ha="center",
            fontsize=10, color=BLUE)
    ax.annotate("", xy=(0.3, -0.45), xytext=(6.0, -0.45),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.8, mutation_scale=14))
    ax.text(3.15, -0.95, "backward: δ⁽³⁾ → δ⁽²⁾ → δ⁽¹⁾, each built from the last",
            ha="center", fontsize=10, color=ORANGE)

    panel(fig.add_subplot(gs[1, 2]), "Reading the sheet", [
        "Superscript (ℓ) = layer: w⁽ˡ⁾, z⁽ˡ⁾, a⁽ˡ⁾ = σ(z⁽ˡ⁾).",
        "Every link is linear (a weight or input)",
        "or a sigmoid (a(1 − a)): each is easy.",
        "",
        "Each step back multiplies δ by Wᵀ and",
        "by σ′ = a(1 − a) ≤ 0.25.",
        "",
        "Learning rate after 4000 epochs: 0.1 is",
        "too slow; 3 is lower but spikes on the way.",
        "",
        "Gradient check matches to ~10⁻¹⁰; autodiff",
        "libraries do this sweep for you.",
    ], spacing=0.083)
    save(fig, name="09_deep_backprop.png")


if __name__ == "__main__":
    sheet_00_perceptron()
    sheet_01_loss_function()
    sheet_02_gradient_descent()
    sheet_03_classification()
    sheet_04_sigmoid_derivative()
    sheet_05_log_loss()
    sheet_06_classification_derivatives()
    sheet_07_neural_network()
    sheet_08_backprop_xor()
    sheet_09_deep_backprop()
