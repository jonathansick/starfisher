#!/usr/bin/env python
# encoding: utf-8
"""
Plotting functions for starfisher.
"""

import numpy as np
import matplotlib as mpl

from starfisher.hess import read_hess


def plot_synth_hess(synthfile, ax, xlim, ylim, dpix, imshow_args=None,
                    xlabel=None, ylabel=None, flipx=False, flipy=False,
                    log_age=None, z=None):
    # FIXME should refactor synthfiles so they know their own xlim, ylim, dpix

    _ = read_hess(synthfile,
                  xlim, ylim, dpix, flipx=flipx,
                  flipy=flipy)
    hess, extent, origin = _

    _imshow = dict(cmap=mpl.cm.gray_r, norm=None,
                   aspect='auto',
                   interpolation='none',
                   extent=extent,
                   origin=origin,
                   alpha=None, vmin=None, vmax=None)
    if imshow_args is not None:
        _imshow.update(imshow_args)
    ax.imshow(np.log10(hess), **_imshow)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if log_age is not None:
        logA = 10. ** (log_age - 9.)
        age_gyr = 10. ** (logA - 9.)
        if age_gyr >= 1.:
            age_str = r"$\log(A)=%.2f$; $%.1f$ Gyr" % (logA, age_gyr)
        else:
            age_str = r"$\log(A)=%.2f$; $%i$ Myr" % (logA, age_gyr * 10. ** 3.)
        ax.text(0.1, 0.9, age_str, ha='left', va='baseline',
                transform=ax.transAxes)

    if z is not None:
        ZZsol = np.log10(z / 0.019)
        z_str = r"$Z=%.4f$; $\log(Z/Z_\odot)=%.2f$" % (z, ZZsol)
        ax.text(0.1, 0.8, z_str, ha='left', va='baseline',
                transform=ax.transAxes)
