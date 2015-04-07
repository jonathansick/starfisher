#!/usr/bin/env python
# encoding: utf-8
"""
Colour Plane definitions and masks.

2015-04-07 - Created by Jonathan Sick
"""

import os

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from astropy.table import Table

from starfisher.pathutils import starfish_dir


class ColorPlane(object):
    """Define a CMD or color-color plane for synth to build.

    Parameters
    ----------
    x_mag : int or tuple
        Indices (0-based) of bands to form the x-axis. If `x_mag` is a
        int, then the x-axis is that magnitude. If `x_mag` is a
        length-2 tuple, then the x-axis is the difference (colour) of
        those two magnitudes.
    y_mag : int or tuple
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
                 suffix=None, x_label="x", y_label="y", dpix=0.05):
        super(ColorPlane, self).__init__()
        if isinstance(y_mag, int):
            self._is_cmd = True
        else:
            self._is_cmd = False
        if not isinstance(x_mag, int):
            x_str = "-".join([str(i + 1) for i in x_mag])
        else:
            x_str = str(x_mag + 1)
        if not isinstance(y_mag, int):
            y_str = "-".join([str(i + 1) for i in y_mag])
        else:
            y_str = str(y_mag + 1)
        if suffix is None:
            suffix = "".join((x_str, y_str)).replace('-', '')
        self.suffix = suffix
        self.x_str = x_str
        self.y_str = y_str
        self.x_span = x_span
        self.y_span = y_span
        self.y_crowding_max = y_crowding_max
        self.x_label = x_label
        self.y_label = y_label
        self.dpix = dpix

        self._msk = self._init_mask()

    @property
    def is_cmd(self):
        return self._is_cmd

    @property
    def synth_config(self):
        lines = []
        lines.append(self.x_str)
        lines.append(self.y_str)
        lines.append("%.2f" % min(self.x_span))
        lines.append("%.2f" % max(self.x_span))
        lines.append("%.2f" % min(self.y_span))
        lines.append("%.2f" % self.y_crowding_max)
        lines.append("%.2f" % max(self.y_span))
        lines.append(self.suffix)
        return lines

    def _init_mask(self):
        """Create an empty color plane mask."""
        nx = int(np.ceil((max(self.x_span) - min(self.x_span)) / self.dpix))
        ny = int(np.ceil((max(self.y_span) - min(self.y_span)) / self.dpix))
        npix = nx * ny

        dt = [("icmd", np.int), ("ibox", np.int), ("maskflag", np.int),
              ("x", np.float), ("y", np.float)]
        msk = np.empty(npix, dtype=dt)

        # Fill in data
        msk['ibox'] = np.arange(1, npix + 1, dtype=np.int)
        msk['maskflag'][:] = 0

        # Produce a coordinate grid
        xgrid = np.linspace(min(self.x_span), max(self.x_span), nx)
        ygrid = np.linspace(min(self.y_span), max(self.y_span), ny)
        x, y = np.meshgrid(xgrid, ygrid, indexing='xy')  # FIXME xy or ij?
        msk['x'] = x.reshape((npix,), order='C')
        msk['y'] = y.reshape((npix,), order='C')
        return msk


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

    def plot_cmd_mask(self, index, output_path, xspan, yspan, dpix,
                      xlabel, ylabel, format="png", dpi=300,
                      figsize=(4, 4), flipx=False, flipy=False, aspect='auto'):
        """Plot a CMD mask.

        .. todo:: Refactor internals of this method and
           :meth:`synth._plot_hess`.
        """
        nx = int((max(xspan) - min(xspan)) / dpix)
        ny = int((max(yspan) - min(yspan)) / dpix)
        # print nx, ny
        mask_image = self._cmds[index]['maskflag'].reshape((ny, nx), order='C')
        # mask_image = self._cmds[index]['y'].reshape((ny, nx), order='C')

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

        fig = Figure(figsize=(4., 4.))
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(1, 1,
                               left=0.15, right=0.95, bottom=0.15, top=0.95,
                               wspace=None, hspace=None,
                               width_ratios=None, height_ratios=None)
        ax = fig.add_subplot(gs[0])
        ax.imshow(mask_image, cmap=mpl.cm.gray_r, norm=None,
                  aspect=aspect,
                  interpolation='none',
                  extent=extent, origin=origin,
                  alpha=None, vmin=None, vmax=None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)

        plot_dir = os.path.dirname(output_path)
        if plot_dir is not "" and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        canvas.print_figure(output_path + "." + format, format=format, dpi=dpi)
