#!/usr/bin/env python
# encoding: utf-8
"""
Read and plot Hess diagram files produced by StarFISH.
"""

import logging
import numpy as np
import math


def reshape_pixarray(pixarray, xspan, yspan, dpix):
    """Reshape a 1D pixel array into a 2D array.

    Parameters
    ----------
    xspan : tuple
        Tuple of ``(x_min, x_max)`` extent of the x-axis, in units of
        magnitudes.
    yspan : tuple
        Tuple of ``(y_min, y_max)`` extent of the y-axis, in units of
        magnitudes.
    dpix : float
        Size of each CMD pixel (``dpix`` by ``dpix`` in area) in mags.
    """
    nx = int(math.ceil((max(xspan) - min(xspan)) / dpix))
    ny = int(math.ceil((max(yspan) - min(yspan)) / dpix))
    if nx * ny != pixarray.shape[0]:
        logging.error("reshape_pixarray %s to (%i, %i)"
                      % (pixarray.shape, nx, ny))
    hess = pixarray.reshape((ny, nx), order='C')
    return hess


def compute_cmd_extent(xspan, yspan, dpix, flipx=False, flipy=False):
    """Compute the extent of a CMD array.

    Parameters
    ----------
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
    return extent, origin


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
    hess = reshape_pixarray(indata, xspan, yspan, dpix)
    extent, origin = compute_cmd_extent(xspan, yspan, dpix,
                                        flipx=flipx, flipy=flipy)
    return hess, extent, origin


def read_chi(path, icmd, xspan, yspan, dpix, flipx=False, flipy=False):
    """Read the chi files output by ``sfh``, converting them into numpy
    arrays.

    Parameters
    ----------
    path : str
        Path to the StarFISH chi output file.
    icmd : int
        Index of the CMD. The first CMD has an index of ``1``.
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
    dt = [('icmd', np.int), ('ibox', np.int), ('Nmod', np.float),
          ('Nobs', np.float), ('dchi', np.float)]
    indata = np.loadtxt(path, dtype=np.dtype(dt))
    # Extract rows for just the CMD of interest
    sel = np.where(indata['icmd'] == icmd)[0]
    indata = indata[sel]
    mod_hess = reshape_pixarray(indata['Nmod'], xspan, yspan, dpix)
    obs_hess = reshape_pixarray(indata['Nobs'], xspan, yspan, dpix)
    chi_hess = reshape_pixarray(indata['dchi'], xspan, yspan, dpix)
    extent, origin = compute_cmd_extent(xspan, yspan, dpix,
                                        flipx=flipx, flipy=flipy)
    return mod_hess, obs_hess, chi_hess, extent, origin
