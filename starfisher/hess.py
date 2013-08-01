#!/usr/bin/env python
# encoding: utf-8
"""
Read and plot Hess diagram files produced by StarFISH.
"""

import numpy as np


def read_hess(path, xspan, yspan, dpix, flipx=False, flipy=False):
    """Read the Hess diagrams produced by StarFISH and convert them into
    plot-ready numpy arrays.

    Parameters
    ----------

    path : str
        Path to the StarFISH Hess diagram file.
    xspan : tuple
        Tuple of ``(x_min, x_max)`` extent of the x-axis, in units of
        magnitudes.
    yspan : tuple
        Tuple of ``(y_min, y_max)`` extent of the y-axis, in units of
        magnitudes.
    dpix : float
        Size of each CMD pixel (``dpix`` by ``dpix`` in area) in mags.
    flipx : bool
        Reverse orientation of x-axis if ``True``.
    flipy : bool
        Reverse orientation of y-axis if ``True`` (e.g., for CMDs).
    """
    indata = np.loadtxt(path)
    nx = int((max(xspan) - min(xspan)) / dpix)
    ny = int((max(yspan) - min(yspan)) / dpix)
    hess = indata.reshape((ny, nx), order='C')

    # extent format is (left, right, bottom, top)
    if flipx:
        extent = [max(xspan), min(xspan)]
    else:
        extent = [min(xspan), max(xspan)]
    if flipy:
        extent.extend([max(yspan), min(yspan)])
    else:
        extent.extend([min(yspan), max(yspan)])
    if flipy:
        origin = 'lower'
    else:
        origin = 'upper'

    return hess, extent, origin


def main():
    pass


if __name__ == '__main__':
    main()
