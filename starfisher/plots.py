#!/usr/bin/env python
# encoding: utf-8
"""
Plotting functions for starfisher.
"""

import numpy as np
import matplotlib as mpl
from matplotlib.collections import PolyCollection

from starfisher.hess import read_hess


def plot_synth_hess(synthfile, ax, cmd, dpix, imshow_args=None,
                    log_age=None, z=None):
    if cmd.is_cmd:
        flipy = True
    else:
        flipy = False

    hess, extent, origin = read_hess(synthfile, cmd.x_span, cmd.y_span, dpix,
                                     flipy=flipy)

    _imshow = dict(cmap=mpl.cm.gray_r, norm=None,
                   aspect='auto',
                   interpolation='none',
                   extent=extent,
                   origin=origin,
                   alpha=None, vmin=None, vmax=None)
    if imshow_args is not None:
        _imshow.update(imshow_args)
    ax.imshow(np.log10(hess), **_imshow)
    ax.set_xlabel(cmd.x_label)
    ax.set_ylabel(cmd.y_label)

    if log_age is not None:
        age_gyr = 10. ** (log_age - 9.)
        if age_gyr >= 1.:
            age_str = r"$\log(A)=%.2f$; $%.1f$ Gyr" % (log_age, age_gyr)
        else:
            age_str = r"$\log(A)=%.2f$; $%i$ Myr" % (log_age,
                                                     age_gyr * 10. ** 3.)
        ax.text(0.1, 0.9, age_str, ha='left', va='baseline',
                transform=ax.transAxes)

    if z is not None:
        ZZsol = np.log10(z / 0.019)
        z_str = r"$Z=%.4f$; $\log(Z/Z_\odot)=%.2f$" % (z, ZZsol)
        ax.text(0.1, 0.8, z_str, ha='left', va='baseline',
                transform=ax.transAxes)


def plot_isochrone_logage_logzsol(ax, library, **args):
    scatter_args = dict(s=4, marker='o', c='r',
                        linewidths=0., zorder=5)
    scatter_args.update(args)
    ax.scatter(library.isoc_logages,
               library.isoc_logzsol,
               **scatter_args)


def plot_lock_polygons(ax, lockfile, **args):
    defaults = dict(edgecolors='k', linewidths=1.)
    defaults.update(args)
    all_verts = []
    for multipoly in lockfile.group_polygons:
        for verts in multipoly.logage_logzsol_verts:
            all_verts.append(np.array(verts))
    ax.add_collection(PolyCollection(all_verts, **defaults))
