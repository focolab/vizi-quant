"""The traces panel image: one line per region, the frame on screen marked.

Rendered to PNG with matplotlib's Agg backend, imported on first use rather
than at module import -- the host gives a plugin 10 s to answer its handshake,
and matplotlib is the slowest import here.  ``render`` also returns where the
axes landed in the image, so a click on the image can be turned back into a
frame (``frame_at``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

#: The size drawn until vizi says how large the image widget is.
WIDTH_PX, HEIGHT_PX, DPI = 470, 210, 100
#: The colours a plot is drawn with when vizi says none (a host before
#: 1.0a103's appearance push): matplotlib's light defaults, as before.
DEFAULT_COLORS = {
    "panel_bg": "#ffffff", "text": "#000000", "text_disabled": "#808080",
    "accent": "#1f77b4", "border": "#000000",
}
#: Lines beyond this many get no legend entry; the plot stays readable.
LEGEND_MAX = 12


@dataclass(frozen=True)
class Axes:
    """The axes box in image pixels (origin top left) and its x data range."""

    left: float
    right: float
    top: float
    bottom: float
    x0: float
    x1: float
    frames: int


def frame_at(axes: "Axes | None", x: float, y: float) -> "int | None":
    """The frame (0-based, as vizi's view counts) under an image pixel, or
    ``None`` outside the axes.  The axis is labelled from 1, as vizi's T."""
    if axes is None or not (axes.left <= x <= axes.right and axes.top <= y <= axes.bottom):
        return None
    shown = axes.x0 + (x - axes.left) / (axes.right - axes.left) * (axes.x1 - axes.x0)
    return int(min(max(round(shown) - 1, 0), axes.frames - 1))


def render(series, frames: int, current_t: int, *, selected=frozenset(), ylabel: str = "", title: str = "",
           size: "tuple[int, int, float]" = (WIDTH_PX, HEIGHT_PX, 1.0), upto: "int | None" = None,
           colors: "dict[str, str] | None" = None):
    """``series``: ``[(region_id, name, values[frames])]`` with NaN where
    unmeasured; ``selected`` the region ids drawn bold; ``size`` the image's
    ``(width, height)`` in pixels and the screen's pixel ratio, which scales
    the text and lines so they look the same on any screen.  Returns
    ``(png bytes, Axes)`` with the axes in the image's own pixels.

    Frames are labelled from 1, as vizi's T shows them.  ``upto`` ends the
    axis at that frame (1-based) -- the last one measured while a run is
    going -- so the traces fill the plot as they grow; ``None`` spans them
    all.

    ``colors`` are vizi's panel colours by name (``Appearance.hex_colors``):
    the background is the panel's, text and ticks its text colour, spines and
    the frame marker its border; each region keeps its own colour, a selected
    one haloed in the accent and, while some are selected, the others in the
    disabled-text colour.  ``None`` draws matplotlib's light defaults."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    width, height, dpr = size
    dpi = DPI * max(dpr, 0.5)
    c = {**DEFAULT_COLORS, **(colors or {})}
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=c["panel_bg"])
    ax = fig.add_subplot(1, 1, 1, facecolor=c["panel_bg"])
    for spine in ax.spines.values():
        spine.set_color(c["border"])
    ax.tick_params(colors=c["text"], labelcolor=c["text"])
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = np.arange(1, frames + 1)
    marker = "." if frames <= 50 else None
    for i, (region_id, name, values) in enumerate(series):
        chosen = region_id in selected
        own = cycle[i % len(cycle)]
        if chosen:
            # the accent as a halo under the region's own colour
            ax.plot(xs, values, color=c["accent"], linewidth=5.0, alpha=0.6, zorder=3, marker=marker,
                    markersize=9, solid_capstyle="round")
        ax.plot(
            xs,
            values,
            color=own if chosen or not selected else c["text_disabled"],
            linewidth=2.2 if chosen else 1.0,
            label=name if i < LEGEND_MAX else None,
            zorder=4 if chosen else 2,
            marker=marker,
        )
    ax.axvline(current_t + 1, color=c["border"], linewidth=0.8, linestyle="--", zorder=1)
    last = frames if upto is None else min(max(int(upto), 1), frames)
    ax.set_xlim(0.5, last + 0.5)
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if last == 1:
        ax.set_xticks([1])
    # A short image keeps the traces and drops what does not fit: the labels
    # below 140 px, the legend below 170 (heights in screen points).
    points = height / max(dpr, 0.5)
    if points >= 140:
        ax.set_xlabel("frame", color=c["text"])
        if ylabel:
            ax.set_ylabel(ylabel, color=c["text"])
    if title:
        ax.set_title(title, fontsize=9, color=c["text"])
    if series and any(np.isfinite(v).any() for _, _, v in series):
        if len(series) > 1 and points >= 170:
            ax.legend(fontsize=7, loc="upper right", ncol=2 if len(series) > 6 else 1,
                      facecolor=c["panel_bg"], edgecolor=c["border"], labelcolor=c["text"])
    else:
        ax.text(0.5, 0.5, "no measurements yet", transform=ax.transAxes, ha="center", va="center",
                color=c["text_disabled"])
    ax.tick_params(labelsize=8)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a figure too small for its labels is still drawn
        fig.tight_layout(pad=0.4 if points < 140 else 1.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=c["panel_bg"])
    box = ax.get_window_extent()  # display pixels, origin bottom left
    height = fig.bbox.height
    x0, x1 = ax.get_xlim()
    axes = Axes(box.x0, box.x1, height - box.y1, height - box.y0, x0, x1, frames)
    return buf.getvalue(), axes
