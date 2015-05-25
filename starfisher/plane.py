#!/usr/bin/env python
# encoding: utf-8
"""
Colour Plane definitions and masks.

2015-04-07 - Created by Jonathan Sick
"""

import os
import logging

import numpy as np
import matplotlib as mpl
from astropy.table import Table

from starfisher.pathutils import starfish_dir


class ColorPlane(object):
    """Define a CMD or color-color plane for synth to build.

    Parameters
    ----------
    x_mag : str or tuple
        Band labels to form the x-axis. If `x_mag` is a
        str, then the x-axis is that magnitude. If `x_mag` is a
        length-2 tuple, then the x-axis is the difference (colour) of
        those two magnitudes.
    y_mag : str or tuple
        Equivalent to `x_mag`, but defines the y-axis.
    x_span : tuple (length-2)
        Tuple of the minimum and maximum values along the x-axis.
    y_span : tuple (length-2)
        Tuple of the minimum and maximum values along the y-axis.
    y_crowding_max : float
        Maximum value along the y-axis to use in the crowding table.
    suffix : str
        Label for this CMD. E.g., if this CMD is B-V, then the suffix
        should be `.bv`.
    x_label : str
        Optional label for x-axis of this CMD. Used by `starfisher`'s
        plotting methods to properly label axes. Can use matplotlib's
        latex formatting.
    y_label : str
        Optional label for y-axis of this CMD. Used by `starfisher`'s
        plotting methods to properly label axes. Can use matplotlib's
        latex formatting.
    dpix : float
        Size of CMD pixels (in magnitudes).
    """
    def __init__(self, x_mag, y_mag, x_span, y_span, y_crowding_max,
                 suffix='.plane', x_label="x", y_label="y", dpix=0.05,
                 nx=None, ny=None):
        super(ColorPlane, self).__init__()
        if isinstance(y_mag, basestring):
            self._is_cmd = True
        else:
            self._is_cmd = False
        self.suffix = suffix
        self.x_mag = x_mag
        self.y_mag = y_mag
        self.x_span = x_span
        self.y_span = y_span
        self.y_crowding_max = y_crowding_max
        self.x_label = x_label
        self.y_label = y_label
        self.dpix = dpix
        self._nx = nx
        self._ny = ny

        self._msk = self._init_mask()

    @property
    def is_cmd(self):
        return self._is_cmd

    def x_str(self, bands):
        if not isinstance(self.x_mag, basestring):
            x = "-".join([str(bands.index(m) + 1) for m in self.x_mag])
        else:
            x = str(bands.index(self.x_mag) + 1)
        return x

    def y_str(self, bands):
        if not isinstance(self.y_mag, basestring):
            y = "-".join([str(bands.index(m) + 1) for m in self.y_mag])
        else:
            y = str(bands.index(self.y_mag) + 1)
        return y

    def synth_config(self, bands):
        lines = []
        lines.append(self.x_str(bands))
        lines.append(self.y_str(bands))
        lines.append("%.2f" % min(self.x_span))
        lines.append("%.2f" % max(self.x_span))
        lines.append("%.2f" % min(self.y_span))
        lines.append("%.2f" % self.y_crowding_max)
        lines.append("%.2f" % max(self.y_span))
        lines.append(self.suffix)
        return lines

    @property
    def nx(self):
        if self._nx is None:
            return int(np.ceil((max(self.x_span) - min(self.x_span))
                       / self.dpix))
        else:
            return self._nx

    @property
    def ny(self):
        if self._ny is None:
            return int(np.ceil((max(self.y_span) - min(self.y_span))
                       / self.dpix))
        else:
            return self._ny

    @property
    def extent(self):
        """The matplotlib-compatible extent description for this color plane.
        """
        if self.is_cmd:
            return [self.x_span[0], self.x_span[-1],
                    self.y_span[-1], self.y_span[0]]
        else:
            return [self.x_span[0], self.x_span[-1],
                    self.y_span[0], self.y_span[-1]]

    @property
    def origin(self):
        """The origin for matplotlib's ``imshow``."""
        return 'lower'

    @property
    def xlim(self):
        return self.x_span

    @property
    def ylim(self):
        if self.is_cmd:
            return (self.y_span[-1], self.y_span[0])
        else:
            return (self.y_span[0], self.y_span[-1])

    def _init_mask(self):
        """Create an empty color plane mask."""
        npix = self.nx * self.ny

        dt = [("icmd", np.int), ("ibox", np.int), ("maskflag", np.int),
              ("x", np.float), ("y", np.float)]
        msk = np.empty(npix, dtype=dt)

        # Fill in data
        msk['ibox'] = np.arange(1, npix + 1, dtype=np.int)
        msk['maskflag'][:] = 0

        # Produce a coordinate grid
        xgrid = np.linspace(self.x_span[0], self.x_span[-1], self.nx)
        ygrid = np.linspace(self.y_span[0], self.y_span[-1], self.ny)
        x, y = np.meshgrid(xgrid, ygrid, indexing='xy')
        y = np.flipud(y)  # magic needed to be read in properly; does work
        msk['x'] = x.reshape((npix,), order='C')
        msk['y'] = y.reshape((npix,), order='C')
        return msk

    def mask_region(self, xspan, yspan):
        """Mask a region of the CMD."""
        s = np.where((self._msk['x'] >= min(xspan)) &
                     (self._msk['x'] <= max(xspan)) &
                     (self._msk['y'] >= min(yspan)) &
                     (self._msk['y'] <= max(yspan)))[0]
        self._msk['maskflag'][s] = 1

    def plot_mask(self, ax, imshow_args=None):
        _args = dict(cmap=mpl.cm.gray_r, norm=None,
                     aspect='auto',
                     interpolation='none',
                     extent=self.extent, origin=self.origin,
                     alpha=None, vmin=None, vmax=None)
        if imshow_args is not None:
            _args.update(imshow_args)
        ax.imshow(self.mask_array, **_args)

    @property
    def mask_array(self):
        return self.reshape_pixarray(np.array(self._msk['maskflag']))

    def reshape_pixarray(self, pixarray):
        """Reshape a 1D pixel array into a 2D Hess array.

        Parameters
        ----------
        pixarray : ndarray
            A 1D pixel array that can be reshaped into a 2D Hess plane image.
        """
        if self.nx * self.ny != pixarray.shape[0]:
            err_msg = "reshape_pixarray {0} to ({1:d}, {2:d})".format(
                pixarray.shape, self.nx, self.ny)
            logging.error(err_msg)
        hess = pixarray.reshape((self.ny, self.nx), order='C')
        return hess

    def read_hess(self, path):
        """Read the Hess diagrams produced by StarFISH and convert them into
        plot-ready numpy arrays.

        Parameters
        ----------
        path : str
            Path to the StarFISH Hess diagram file.
        """
        indata = np.loadtxt(path)
        return self.reshape_pixarray(indata)

    def read_chi(self, path, icmd):
        """Read the chi files output by ``sfh``, converting them into numpy
        arrays.

        Parameters
        ----------
        path : str
            Path to the StarFISH chi output file.
        icmd : int
            Index of the CMD. The first CMD has an index of ``1``.
        """
        dt = [('icmd', np.int), ('ibox', np.int), ('Nmod', np.float),
              ('Nobs', np.float), ('dchi', np.float)]
        indata = np.loadtxt(path, dtype=np.dtype(dt))
        # Extract rows for just the CMD of interest
        sel = np.where(indata['icmd'] == icmd)[0]
        indata = indata[sel]
        mod_hess = self.reshape_pixarray(indata['Nmod'])
        obs_hess = self.reshape_pixarray(indata['Nobs'])
        chi_hess = self.reshape_pixarray(indata['dchi'])
        return mod_hess, obs_hess, chi_hess


class Mask(object):
    """Create a mask file for regular CMD gridding."""
    def __init__(self, color_planes):
        super(Mask, self).__init__()
        self.mask_path = None
        self._cmds = []  # Masks for each CMD
        self._current_cmd_index = 1
        for i, plane in enumerate(color_planes):
            msk = plane._msk
            msk['icmd'][:] = i + 1
            self._cmds.append(msk)

    @property
    def full_mask_path(self):
        return os.path.join(starfish_dir, self.mask_path)

    def write(self, mask_path):
        """Write the mask file."""
        self.mask_path = mask_path

        dirname = os.path.dirname(self.full_mask_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        mskdata = np.concatenate(tuple(self._cmds))
        t = Table(mskdata)
        t.write(self.full_mask_path,
                format="ascii.fixed_width_no_header",
                delimiter=' ',
                bookend=False,
                delimiter_pad=None,
                include_names=['icmd', 'ibox', 'maskflag'],
                formats={"icmd": "%i", "ibox": "%i", "maskflag": "%i"})


class StarCatalogHess(object):
    """Bin star catalog to create a Hess diagram.

    Parameters
    ----------
    x : ndarray
        Star magnitudes/colours for the x-coordinate of the Hess diagram.
    y : ndarray
        Star magnitudes/colours for the y-coordinate of the Hess diagram.
    plane : :class:`starfisher.plane.ColorPlane`
        The Hess plane's geometry.
    """
    def __init__(self, x, y, plane):
        super(StarCatalogHess, self).__init__()
        self._plane = plane

        range_ = np.array([self._plane.x_span, self._plane.y_span])
        self._h, self._xedges, self._yedges = np.histogram2d(
            x, y, bins=[self._plane.nx, self._plane.ny],
            range=range_)
        self._h = np.flipud(self._h.T)

    @property
    def hess(self):
        """The Hess diagram as numpy array."""
        return self._h

    @property
    def masked_hess(self):
        """Hess where masked pixels are set to NaN."""
        mask = self._plane.mask_array
        H = np.array(self._h)
        H[mask == 1] = np.nan
        return H

    @property
    def origin(self):
        return 'lower'

    @property
    def extent(self):
        return self._plane.extent


class Hess(object):
    """A generic hess data structure."""
    def __init__(self, data, colorplane):
        super(Hess, self).__init__()
        self._plane = colorplane
        self._h = data

    @property
    def hess(self):
        """The Hess diagram as numpy array."""
        return self._h

    @property
    def masked_hess(self):
        """Hess where masked pixels are set to NaN."""
        mask = self._plane.mask_array
        H = np.array(self._h)
        H[mask == 1] = np.nan
        return H

    @property
    def origin(self):
        return 'lower'

    @property
    def extent(self):
        return self._plane.extent


class SynthHess(object):
    """Hess diagram made by Synth.

    Informally uses the same API and ColorPlane and SimHess.
    (I should formalize this API).
    """
    def __init__(self, synth_path, colorplane):
        super(SynthHess, self).__init__()
        self._plane = colorplane
        self._h = self._plane.read_hess(synth_path)

    @property
    def hess(self):
        """The Hess diagram as numpy array."""
        return self._h

    @property
    def masked_hess(self):
        """Hess where masked pixels are set to NaN."""
        mask = self._plane.mask_array
        H = np.array(self._h)
        H[mask == 1] = np.nan
        return H

    @property
    def origin(self):
        return 'lower'

    @property
    def extent(self):
        return self._plane.extent


class SimHess(object):
    """Builds Hess diagrams from synthetic SSP hess diagrams made by `synth`.

    Parameters
    ----------
    synth : :class:`starfisher.synth.Synth`
        A `Synth` instance where simulated Hess diagrams have been pre-built.
    colorplane : :class:`starfisher.plane.ColorPlane`
        The `ColorPlane` that will be simulated. This `ColorPlane` must be
        in `Synth`.
    amplitudes : ndarray
        Star formation amplitudes for each isochrone (group).
    """
    def __init__(self, synth, colorplane, amplitudes):
        super(SimHess, self).__init__()
        self._synth = synth
        self._plane = colorplane
        self._amps = amplitudes

        assert self._plane in self._synth._cmds
        assert len(self._amps) == len(self._synth.lockfile.active_groups)

        # build hess diagram
        self._h = self._build_hess()

    @classmethod
    def from_sfh_solution(cls, sfh, colorplane):
        """Construct a :class:`SimHess` from a SFH fitted star formation
        history.
        """
        t = sfh.solution_table()
        amps = t['amp_nstars']
        return cls(sfh.synth, colorplane, amps)

    @property
    def hess(self):
        """The Hess diagram as numpy array."""
        return self._h

    @property
    def masked_hess(self):
        """Hess where masked pixels are set to NaN."""
        mask = self._plane.mask_array
        H = np.array(self._h)
        H[mask == 1] = np.nan
        return H

    @property
    def origin(self):
        return 'lower'

    @property
    def extent(self):
        return self._plane.extent

    @property
    def mean_logage_hess(self):
        hess_stack = self._build_hess_stack()
        msk = np.zeros(hess_stack.shape, dtype=np.bool)
        msk[hess_stack == 0] = True
        hess_stack = np.ma.array(hess_stack, mask=msk)
        logages = np.swapaxes(
            np.atleast_3d(self._synth.lockfile.group_logages),
            1, 2)
        ages_stack = logages * np.ma.ones(hess_stack.shape, dtype=np.float)
        ages_stack.mask = msk
        return np.ma.average(ages_stack, axis=2, weights=hess_stack)

    def _build_hess(self):
        # Co-add synth hess diagrams, weighted by amplitude
        hess_stack = self._build_hess_stack()
        return np.sum(hess_stack, axis=2)

    def _build_hess_stack(self):
        synth_hess = self._read_synth_hess()
        hess_stack = np.dstack(synth_hess)
        A = np.swapaxes(np.atleast_3d(self._amps), 1, 2)
        hess_stack = A * hess_stack
        return hess_stack

    def _read_synth_hess(self):
        hesses = []
        for name in self._synth.lockfile.active_groups:
            synth_path = os.path.join(starfish_dir, name + self._plane.suffix)
            h = self._plane.read_hess(synth_path)
            hesses.append(h)
        return hesses
