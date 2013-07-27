#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `sfh`, the program for estimating the star formation
of a stellar population by optimizing the linear combination of eigen-CMDs.
"""

import os

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from astropy.table import Table


class Mask(object):
    """Create a mask file for regular CMD gridding."""
    def __init__(self, mask_path):
        super(Mask, self).__init__()
        self.mask_path = mask_path
        self._cmds = []  # Masks for each CMD
        self._current_cmd_index = 0

    def init_cmd(self, xspan, yspan, dpix):
        """Add a CMD to mask.

        CMDs must be added in the same order as specified for the `synth`
        program.

        .. todo:: Check the coordinate packaging against plotting.

        Parameters
        ----------

        xspan : tuple
            Tuple of min and max x-axis coordinates.
        yspan : tuple
            Tuple of min and max y-axis coordinates.
        dpix : float
            Size of CMD pixels
        """
        dt = [("icmd", np.int), ("ibox", np.int), ("maskflag", np.int),
                ("x", np.float), ("y", np.float)]
        nx = int((max(xspan) - min(xspan)) / dpix)
        ny = int((max(yspan) - min(yspan)) / dpix)
        npix = nx * ny
        msk = np.empty(npix, dtype=dt)
        # Fill in data
        msk['icmd'][:] = self._current_cmd_index
        msk['ibox'] = np.arange(1, npix + 1, dtype=np.int)
        msk['maskflag'][:] = 0
        # Produce a coordinate grid
        xgrid = np.linspace(min(xspan), max(xspan), nx)
        ygrid = np.linspace(min(yspan), max(yspan), ny)
        x, y = np.meshgrid(xgrid, ygrid, indexing='xy')  # TODO xy or ij
        msk['x'] = x.reshape((npix,), order='C')
        msk['y'] = y.reshape((npix,), order='C')
        self._cmds.append(msk)
        self._current_cmd_index += 1

    def write(self):
        """Write the mask file."""
        dirname = os.path.dirname(self.mask_path)
        if not os.path.exists(dirname): os.makedirs(dirname)
        if os.path.exists(self.mask_path): os.remove(self.mask_path)
        mskdata = np.concatenate(tuple(self._cmds))
        t = Table(mskdata)
        t.write(self.mask_path, format="ascii.fixed_width_no_header",
                delimiter=' ', bookend=False, delimiter_path=None,
                names=['icmd', 'ibox', 'maskflag'],
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
            wspace=None, hspace=None, width_ratios=None, height_ratios=None)
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


def main():
    msk = Mask("mask.txt")
    msk.init_cmd((0., 3.), (12., 20.), 0.05)
    msk.plot_cmd_mask(0, "test_mask", (0., 3.), (12., 20.), 0.05, "x", "y",
            flipy=True)


if __name__ == '__main__':
    main()
