"""Figures for fields, images and thermal results. Needs matplotlib."""

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:      # pragma: no cover
        raise ImportError("sonosk.plot needs matplotlib: pip install 'sonosk[plot]'") from e
    return plt


def _extent(x, z):
    return [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]


def bmode(image, x, z, dynamic_range=60.0, ax=None, title=None):
    """Log-compressed B-mode image. image is complex or envelope, (len(z), len(x))."""
    plt = _plt()
    from .imaging import log_compress
    ax = plt.gca() if ax is None else ax
    h = ax.imshow(log_compress(image, dynamic_range), cmap="gray", vmin=-dynamic_range,
                  vmax=0, extent=_extent(x, z), aspect="equal")
    ax.set_xlabel("Lateral [mm]")
    ax.set_ylabel("Depth [mm]")
    if title:
        ax.set_title(title)
    return h


def field_map(p, x, z, db=True, dynamic_range=40.0, ax=None, title=None, cmap="inferno"):
    """Pressure amplitude map in the x-z plane. p has shape (len(x), len(z))."""
    plt = _plt()
    ax = plt.gca() if ax is None else ax
    a = np.abs(p).T
    if db:
        a = 20 * np.log10(np.maximum(a / a.max(), 1e-12))
        h = ax.imshow(a, cmap=cmap, vmin=-dynamic_range, vmax=0, extent=_extent(x, z),
                      aspect="equal")
    else:
        h = ax.imshow(a, cmap=cmap, extent=_extent(x, z), aspect="equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    if title:
        ax.set_title(title)
    return h


def temperature_map(T, x, z, ax=None, title=None, lesion=None, vmin=37.0, vmax=None):
    """Temperature in the x-z plane, optionally with the lesion outline.
    T and lesion have shape (len(x), len(z))."""
    plt = _plt()
    ax = plt.gca() if ax is None else ax
    h = ax.imshow(np.asarray(T).T, cmap="hot", vmin=vmin, vmax=vmax, extent=_extent(x, z),
                  aspect="equal")
    if lesion is not None and np.any(lesion):
        ax.contour(x * 1e3, z * 1e3, np.asarray(lesion, float).T, levels=[0.5],
                   colors="cyan", linewidths=1.0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    if title:
        ax.set_title(title)
    return h


def labels_map(labels, x, z, names=None, ax=None, title=None):
    """Label map of a 2-D phantom."""
    plt = _plt()
    ax = plt.gca() if ax is None else ax
    lab = np.asarray(labels).T
    h = ax.imshow(lab, cmap="tab10", vmin=-0.5, vmax=9.5, extent=_extent(x, z), aspect="equal",
                  interpolation="nearest")
    if names:
        for i, nm in enumerate(names):
            if np.any(lab == i):
                ax.plot([], [], "s", color=plt.cm.tab10(i), label=nm)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    if title:
        ax.set_title(title)
    return h


def orthoslices(volume, grid, index=None, cmap="viridis", title=None):
    """Three orthogonal slices through a 3-D volume on grid."""
    plt = _plt()
    v = np.asarray(volume)
    if v.ndim != 3:
        raise ValueError("orthoslices needs a 3-D volume")
    i, j, k = index if index is not None else np.unravel_index(np.argmax(np.abs(v)), v.shape)
    x, y, z = [a * 1e3 for a in grid.axes()]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    ax[0].imshow(np.abs(v[:, j, :]).T, extent=[x[0], x[-1], z[-1], z[0]], cmap=cmap)
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("z [mm]")
    ax[1].imshow(np.abs(v[i, :, :]).T, extent=[y[0], y[-1], z[-1], z[0]], cmap=cmap)
    ax[1].set_xlabel("y [mm]")
    ax[1].set_ylabel("z [mm]")
    ax[2].imshow(np.abs(v[:, :, k]).T, extent=[x[0], x[-1], y[-1], y[0]], cmap=cmap)
    ax[2].set_xlabel("x [mm]")
    ax[2].set_ylabel("y [mm]")
    if title:
        fig.suptitle(title)
    return fig
