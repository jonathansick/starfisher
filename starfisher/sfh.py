#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `sfh`, the program for estimating the star formation
of a stellar population by optimizing the linear combination of eigen-CMDs.
"""

import os
import subprocess
import math

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from astropy.table import Table, Column


class SFH(object):
    """Interface to the StarFISH ``sfh`` program.
    
    Parameters
    ----------

    data_root : str
        Root filename of the photometry data (the full path minus the suffix
        for each CMD plane).
    synth : :class:`synth.Synth` instance
        The instance of :class:`synth.Synth` used to prepare the synthetic
        CMDs.
    mask : :class:`sfh.Mask` instance 
        The instance of :class:`sfh.Mask` specifying how each CMD plane
        should be masked.
    input_dir : str
        Direcory where input files are stored for the StarFISH run.
    """
    def __init__(self, data_root, synth, mask, input_dir):
        super(SFH, self).__init__()
        self.data_root = data_root
        self.synth = synth
        self.mask = mask
        self.input_dir = input_dir
        self._sfh_config_path = os.path.join(self.input_dir, "sfh.dat")
        self._outfile_path = os.path.join(self.input_dir, "output.dat")
        self._hold_path = os.path.join(self.input_dir, "hold.dat")
        self._log_path = os.path.join(self.input_dir, "sfh.log")
        self._plg_path = os.path.join(self.input_dir, "plg.log")
        self._chi_path = os.path.join(self.input_dir, "chi.txt")

    def run_sfh(self):
        """Run the StarFISH `sfh` software."""
        self.cmd_path = os.path.join(self.input_dir, "cmd.txt")
        self.synth.lockfile.write_cmdfile(self.cmd_path)
        self.synth.lockfile.write_holdfile(self._hold_path)
        self.mask.write()
        self._write_sfh_input()
        subprocess.call("./sfh < %s" % self._sfh_config_path, shell=True)

    def _write_sfh_input(self):
        """Write the SFH input file."""
        if os.path.exists(self._sfh_config_path):
            os.remove(self._sfh_config_path)

        lines = []

        # Filenames
        lines.append(self.data_root)  # datpre
        lines.append(self.cmd_path)  # cmdfile
        lines.append(self.mask.mask_path)  # maskfile
        lines.append(self._hold_path)  # hold file (needs to be created)
        lines.append(self._outfile_path)  # output
        lines.append(self._log_path)  # log
        lines.append(self._plg_path)  # plg
        lines.append(self._chi_path)  # chi

        # Synth CMD parameters
        # number of independent isochrones
        # TODO modified by the holdfile?
        lines.append(str(self.synth.n_active_groups))
        lines.append(str(self.synth.n_cmd))
        lines.append("1")  # binning factor between synth and CMD pixels
        lines.append(str(self.synth.dpix))

        # Parameters for each CMD
        # TODO likely want a better accessor here for Synth's CMDs
        for cmd in self.synth._cmds:
            lines.append(cmd['suffix'])
            lines.append("%.2f" % min(cmd['x_span']))
            lines.append("%.2f" % max(cmd['x_span']))
            lines.append("%.2f" % min(cmd['y_span']))
            lines.append("%.2f" % max(cmd['y_span']))
            nx = int((max(cmd['x_span']) - min(cmd['x_span']))
                    / self.synth.dpix)
            ny = int((max(cmd['y_span']) - min(cmd['y_span']))
                    / self.synth.dpix)
            nbox = nx * ny
            lines.append(str(nbox))

        # Runtime parameters
        # TODO enable user customization here
        lines.append("256")  # seed
        lines.append("2")  # Use Poisson fit statistic
        lines.append("0")  # don't start from a logged position
        lines.append("0")  # don't generate plg file of all tested positions
        lines.append("0")  # uniform grid
        lines.append("3")  # verbosity
        lines.append("1000.00")  # lambda; initial simplex size
        lines.append("68.0")  # error bars are at 1 sigma confidence level
        lines.append("1.000")  # threshold delta-chi**2
        lines.append("10.00")  # required parameter tolerance
        lines.append("0.0000001")  # required fit_stat tolerance
        lines.append("10000")  # number of parameter directions to search
        lines.append("3")  # number of iterations for determining errorbars

        txt = "\n".join(lines)
        with open(self._sfh_config_path, 'w') as f:
            f.write(txt)

    def solution_table(self, avgmass=1.628):
        """Returns a `class`:astropy.table.Table of the derived star formation
        history.

        This is based on the ``sfh.sm`` script distributed with StarFISH.

        Parameters
        ----------

        avgmass : float
            Average mass of the stellar population; given the IMF. For a
            Salpeter IMF this is 1.628.
        """
        # read in time interval table (produced by lockfile)
        dt = self.synth.lockfile.group_dt()

        # read sfh output
        t = Table.read(self._outfile_path,
               format="ascii.no_header",
               names=['Z', 'log(age)',
                   'amp_nstars', 'amp_nstars_n', 'amp_nstars_p'])

        # Open a photometry file to count stars
        dataset_path = self.data_root + self.synth._cmds[0]['suffix']
        _catalog = np.loadtxt(dataset_path)
        nstars = _catalog.shape[0]

        # Renormalize to SFR (Msun/yr)
        # (Amps in the SFH file have units Nstars.)
        ep = (t['amp_nstars_p'] - t['amp_nstars']) * avgmass / dt
        en = (t['amp_nstars'] - t['amp_nstars_n']) * avgmass / dt
        sfr = t['amp_nstars'] * avgmass / dt

        # Include Poisson errors in errorbars
        snstars = np.sqrt(float(nstars))
        _foo = t['amp_nstars'] * np.sqrt((snstars / nstars) ** 2.)
        sap = ep + _foo
        san = en + _foo
        # Truncate error bars if they extend below zero
        san[san < 0.] = 0.

        csfr = Column(sfr, name='sfr', unit='M_solar/yr')
        csap = Column(sap, name='sfr_pos_err', unit='M_solar/yr')
        csan = Column(san, name='sfr_neg_err', unit='M_solar/yr')
        t.add_columns([csfr, csap, csan])

        return t


class Mask(object):
    """Create a mask file for regular CMD gridding."""
    def __init__(self, mask_path):
        super(Mask, self).__init__()
        self.mask_path = mask_path
        self._cmds = []  # Masks for each CMD
        self._current_cmd_index = 1

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
        nx = int(math.ceil((max(xspan) - min(xspan)) / dpix))
        ny = int(math.ceil((max(yspan) - min(yspan)) / dpix))
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
                delimiter=' ', bookend=False, delimiter_pad=None,
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
