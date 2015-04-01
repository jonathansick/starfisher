#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `synth`, the program for creating eigen-CMDs for each
isochrone and CMD plane.
"""

import os
import glob
import subprocess
import logging
import multiprocessing

import numpy as np
from astropy.table import Table

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from starfisher.pathutils import starfish_dir, EnterStarFishDirectory
from starfisher.plots import plot_synth_hess


class Synth(object):
    """Interface to StarFISH's `synth` command.

    :class:`ExtinctionDistribution` instances can be set for either (or both)
    young and old stars. If these extinction distributions are not set,
    then a zero extinction file will automatically be created. To have the
    same extinction for both young and old stars, set these to the same
    instance.

    Parameters
    ----------
    library_builder : :class:`isolibrary.LibraryBuilder` instance
        The instance of :class:`isolibrary.LibraryBuilder` used to prepare the
        isochrone library
    lockfile : :class:`Lockfile` instance
        A prepared :class:`Lockfile` instance.
    input_dir : str
        Directory where input files are stored for the StarFISH run, relative
        to the StarFISH directory. Typically this is `'input'`.
    rel_extinction : `ndarray`, `(n_bands, 1)`
        Sequence of relative extinction values for each band. The bands must be
        ordered as in the isochrones. One band *must* have a relative
        extinction of 1.0, which signifies the magnitude system that the
        :class:`ExtinctionDistribution` files are in.
    young_extinction : :class:`ExtinctionDistribution`
        An :class:`ExtinctionDistribution` instance for young stars (younger
        than log(age) = 7.0 by default).
    old_extinction : :class:`ExtinctionDistribution`
        An :class:`ExtinctionDistribution` instance for old stars (older
        than log(age) = 7.0 by default).
    dpix : float
        Size of CMD pixels (in magnitudes).
    nstars : int
        Number of stars in include in artificial population per isochone.
    verb : int
        Verbosity flag (higher N = more output)
    interp_err : bool
        Set to `True` to interpolate errors between bracketing crowding bins.
    seed : int
        Seed value for StarFISH's random number generator.
    mass_span : tuple, length 2
        Tuple giving the (minimum, maximum) mass range of the synthesized
        stars.
    fbinary : float
        Binary fraction.
    crowdfile :
        A crowding data instance,
        e.g., :class:`starfisher.crowd.MockNullCrowdingTable`.
    crowd_output_path : str
        Path where `synth` will write the error lookup table, relative
        to the StarFISH directory.
    planes : list
        Optional list of :class:`ColorPlane` instances. Can also set these
        with the :meth:`add_cmd` method.
    """
    def __init__(self, input_dir, library_builder, lockfile, crowdfile,
                 rel_extinction, young_extinction=None, old_extinction=None,
                 dpix=0.05, nstars=1000000, verb=3, interp_err=True,
                 seed=256, mass_span=(0.5, 100.), fbinary=0.5,
                 planes=None):
        super(Synth, self).__init__()
        self.library_builder = library_builder
        self.lockfile = lockfile
        self.input_dir = input_dir
        self.rel_extinction = rel_extinction
        self.young_extinction = young_extinction
        self.old_extinction = old_extinction
        self.dpix = dpix
        self.nstars = nstars
        self.verb = verb
        self.interp_err = interp_err
        self.seed = seed
        self.mass_span = mass_span
        self.fbinary = fbinary
        self.crowdfile = crowdfile
        self.crowding_output_path = os.path.join(input_dir, "crowd_lookup.dat")

        if planes is not None:
            self._cmds = planes
        else:
            self._cmds = []  # add_cmd() inserts data here

    @property
    def lock_path(self):
        return os.path.join(self.input_dir, "lock.dat")

    @property
    def full_lock_path(self):
        return os.path.join(starfish_dir, self.lock_path)

    @property
    def synth_config_path(self):
        return os.path.join(self.input_dir, "synth.dat")

    @property
    def full_synth_config_path(self):
        return os.path.join(starfish_dir, self.synth_config_path)

    @property
    def full_input_dir(self):
        return os.path.join(starfish_dir, self.input_dir)

    @property
    def young_av_path(self):
        return os.path.join(self.input_dir, "young.av")

    @property
    def full_young_av_path(self):
        return os.path.join(starfish_dir, self.young_av_path)

    @property
    def old_av_path(self):
        return os.path.join(self.input_dir, "old.av")

    @property
    def full_old_av_path(self):
        return os.path.join(starfish_dir, self.old_av_path)

    @property
    def n_cmd(self):
        """Number of CMD planes."""
        return len(self._cmds)

    @property
    def n_active_groups(self):
        """Number of isochrone groups that have been realized by synth``.

        Isochrones that generated errors will be excluded here. This value
        should be used as the input for ``sfh`` for the dimensionality
        of the optimizations.
        """
        return len(self.lockfile.active_groups)

    def add_cmd(self, cmd):
        """Add a CMD plane for synthesis."""
        self._cmds.append(cmd)

    def run_synth(self, n_cpu=1, include_unlocked=False):
        """Run the StarFISH `synth` code to create synthetic CMDs.
        
        Parameters
        ----------
        n_cpu : int
            Number of CPUs to run synth with. For `n_cpu` > 1, the lockfile
            is split up so that several `synth` commands can be run
            simultaneously.
        include_unlocked : bool
            Synthesize isochrones even if they were not explicity inluded in
            a group in the lockfile.
        """
        synth_paths = self._write(n_cpu, include_unlocked)
        self._clean()

        if not os.path.exists(self.lockfile.full_synth_dir):
            os.makedirs(self.lockfile.full_synth_dir)
        if n_cpu > 1:
            pool = multiprocessing.Pool(n_cpu)
            m = pool.map
        else:
            m = map
        # map synth
        m(synth_paths)
        if n_cpu > 1:
            pool.close()

    def _write(self, n_cpu, include_unlocked):
        """Write the `synth` input file."""
        if os.path.exists(self.full_synth_config_path):
            os.remove(self.full_synth_config_path)

        # Prep lock file and edited isofile
        self.lockfile.write(self.lock_path,
                            include_unlocked=include_unlocked)

        # Create each line of synth input
        lines = []

        lines.append(self.lockfile.synth_isofile_path)  # matches lockfile
        lines.append(self.lockfile.lock_path)

        self.young_extinction.write(self.young_av_path)
        lines.append(self.young_av_path)

        self.old_extinction.write(self.old_av_path)
        lines.append(self.old_av_path)

        lines.append(self.crowdfile.path)
        lines.append(self.crowding_output_path)

        lines.append(str(self.library_builder.nmag))
        lines.append(str(len(self._cmds)))
        lines.append(str(self.library_builder.mag0))

        lines.append(str(self.dpix))

        # CMD section
        for cmd in self._cmds:
            lines.extend(cmd.synth_config)

        # Crowding section
        lines.extend(self.crowdfile.config_section)

        for av_ratio in self.rel_extinction:
            lines.append("%.3f" % av_ratio)

        lines.append(str(self.verb))

        if self.interp_err:
            lines.append("1")
        else:
            lines.append("0")

        lines.append(str(self.crowdfile.error_method))

        lines.append(str(self.nstars))
        lines.append(str(self.seed))
        lines.append("%.2f" % min(self.mass_span))
        lines.append("%.2f" % max(self.mass_span))
        lines.append("%.2f" % self.library_builder.gamma)
        lines.append("%.2f" % self.library_builder.faint)
        lines.append("%.2f" % self.fbinary)

        txt = "\n".join(lines)
        with open(self.full_synth_config_path, 'w') as f:
            f.write(txt)

    def _clean(self):
        """Remove existing synthetic CMDs."""
        paths = glob.glob(os.path.join(self.lockfile.full_synth_dir, "z*"))
        for path in paths:
            logging.warning("Removing %s" % path)
            os.remove(path)

    def plot_all_hess(self, plotdir, **plot_args):
        """Plot Hess (binned CMD) diagrams of all synthetic CMD planes.

        Parameters
        ----------
        plotdir : str
            Directory where plots will be saved.
        format : str
            Format of the plot (typically ``png``, ``pdf`` or ``eps).
        dpi : int
            Resolution of the output.
        figsize : tuple
            Size of matplotlib axes.
        """
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        group_names = self.lockfile.active_groups
        for name in group_names:
            for cmd in self._cmds:
                synth_path = os.path.join(starfish_dir, name + cmd.suffix)
                basename = os.path.basename(synth_path)
                plot_path = os.path.join(plotdir, basename)
                if not os.path.exists(synth_path):
                    logging.warning("plot_all_hess: %s does not exist"
                                    % synth_path)
                    continue
                self._plot_hess(synth_path, plot_path, name, cmd,
                                log_age=self.lockfile.mean_age_for_group(name),
                                z=self.lockfile.mean_z_for_group(name),
                                **plot_args)

    def _plot_hess(self, synth_path, plot_path, name, cmd,
                   log_age=None, z=None,
                   format="png", dpi=300,
                   figsize=(4, 4), aspect='auto'):
        """Plot a Hess diagram for a single synthesized image."""
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(1, 1,
                               left=0.17, right=0.95, bottom=0.15, top=0.95,
                               wspace=None, hspace=None,
                               width_ratios=None, height_ratios=None)
        ax = fig.add_subplot(gs[0])
        plot_synth_hess(synth_path, ax, cmd, self.dpix,
                        log_age=log_age, z=z)
        gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
        canvas.print_figure(plot_path + "." + format, format=format, dpi=dpi)


def _run_synth(synth_config_path):
    with EnterStarFishDirectory():
        command = "./synth < {0}".format(synth_config_path)
        print(command)
        subprocess.call(command, shell=True)


class ExtinctionDistribution(object):
    """Create an extinction distribution file for :class:`Synth`.

    Synthesized stars will have extinction values drawn randomly from samples
    in the extintion distribution. Uniform extinction can be implemented by
    using only one extinction value.
    """
    def __init__(self):
        super(ExtinctionDistribution, self).__init__()
        self._extinction_array = None

    def set_samples(self, extinction_array):
        """Set a 1D array of extinction values.

        Parameters
        ----------
        extinction_array : ndarray, (n,)
            A 1D array of extinction sample values (in magnitudes).
        """
        # TODO check shape of extinction array
        self._extinction_array = extinction_array

    def set_uniform(self, extinction):
        """Set uniform extinction.

        Parameters
        ----------
        extinction : float
            The uniform extinction value (in magnitudes).
            I.e., set `extinction=0.` for no
            extinction.
        """
        self._extinction_array = np.array([extinction])

    def write(self, path):
        """Write the extinction file to `path`.

        Parameters
        ----------
        path : str
            Path where extinction file is written, relative to the StarFISH
            directory.
        """
        full_path = os.path.join(starfish_dir, path)
        dirname = os.path.dirname(full_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        t = Table([self._extinction_array], names=['A'])
        t.write(full_path, format='ascii.no_header', delimiter=' ')


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
    """
    def __init__(self, x_mag, y_mag, x_span, y_span, y_crowding_max,
                 suffix=None, x_label="x", y_label="y"):
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
