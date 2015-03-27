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

import numpy as np
from scipy.stats import mode
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from astropy.table import Table

from starfisher.pathutils import starfish_dir, EnterStarFishDirectory
from starfisher.hess import read_hess


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
    """
    def __init__(self, input_dir, library_builder, lockfile, crowdfile,
                 rel_extinction, young_extinction=None, old_extinction=None,
                 dpix=0.05, nstars=1000000, verb=3, interp_err=True,
                 seed=256, mass_span=(0.5, 100.), fbinary=0.5):
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

    def add_cmd(self, x_mag, y_mag, x_span, y_span, y_crowding_max, suffix,
                xlabel="x", ylabel="y"):
        """Add a CMD plane for synthesis.

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
        xlabel : str
            Optional label for x-axis of this CMD. Used by `starfisher`'s
            plotting methods to properly label axes. Can use matplotlib's
            latex formatting.
        ylabel : str
            Optional label for y-axis of this CMD. Used by `starfisher`'s
            plotting methods to properly label axes. Can use matplotlib's
            latex formatting.
        """
        if not isinstance(x_mag, int):
            x_str = "-".join([str(i + 1) for i in x_mag])
        else:
            x_str = str(x_mag + 1)
        if not isinstance(y_mag, int):
            y_str = "-".join([str(i + 1) for i in y_mag])
        else:
            y_str = str(y_mag + 1)
        cmd_def = {'x_mag': x_mag, 'y_mag': y_mag,
                   "x_str": x_str, "y_str": y_str,
                   "x_span": x_span, "y_span": y_span,
                   "y_crowding_max": y_crowding_max,
                   "suffix": suffix,
                   "x_label": xlabel, "y_label": ylabel}
        self._cmds.append(cmd_def)

    def run_synth(self, include_unlocked=False):
        """Run the StarFISH `synth` code to create synthetic CMDs."""
        self._write(include_unlocked=include_unlocked)
        self._clean()
        if not os.path.exists(self.lockfile.full_synth_dir):
            os.makedirs(self.lockfile.full_synth_dir)
        with EnterStarFishDirectory():
            command = "./synth < {0}".format(self.synth_config_path)
            print(command)
            subprocess.call(command, shell=True)

    def _write(self, include_unlocked=False):
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
            lines.append(cmd['x_str'])
            lines.append(cmd['y_str'])
            lines.append("%.2f" % min(cmd['x_span']))
            lines.append("%.2f" % max(cmd['x_span']))
            lines.append("%.2f" % min(cmd['y_span']))
            lines.append("%.2f" % cmd['y_crowding_max'])
            lines.append("%.2f" % max(cmd['y_span']))
            lines.append(cmd['suffix'])

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
        flipx : bool
            Reverse orientation of x-axis if ``True``.
        flipy : bool
            Reverse orientation of y-axis if ``True`` (e.g., for CMDs).
        aspect : str
            Controls aspect of the image axes. Set to ``auto`` for color-
            magnitude digrams. For color-color diagrams where pixels must
            have equal aspect, set to ``equal``.
        """
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        group_names = self.lockfile.active_groups
        print("group_names", group_names)
        for name in group_names:
            for cmd in self._cmds:
                print("suffix", cmd['suffix'])
                synth_path = os.path.join(starfish_dir, name + cmd['suffix'])
                basename = os.path.basename(synth_path)
                plot_path = os.path.join(plotdir, basename)
                if not os.path.exists(synth_path):
                    logging.warning("plot_all_hess: %s does not exist"
                                    % synth_path)
                    continue
                self._plot_hess(synth_path, plot_path, cmd, **plot_args)

    def _plot_hess(self, synth_path, plot_path, cmd, format="png", dpi=300,
                   figsize=(4, 4), flipx=False, flipy=False, aspect='auto'):
        """Plot a Hess diagram for a single synthesized image."""
        _ = read_hess(synth_path,
                      cmd['x_span'], cmd['y_span'], self.dpix, flipx=flipx,
                      flipy=flipy)
        hess, extent, origin = _

        # Get synthetic Z and logA from filename
        basename = os.path.splitext(os.path.basename(synth_path))[0][1:]
        zstr, logastr = basename.split("_")
        Z = float(zstr) / 10000.
        logA = float(logastr)

        ZZsol = np.log10(Z / 0.019)
        age_gyr = 10. ** (logA - 9.)
        z_str = r"$Z=%.4f$; $\log(Z/Z_\odot)=%.2f$" % (Z, ZZsol)
        if age_gyr >= 1.:
            age_str = r"$\log(A)=%.2f$; $%.1f$ Gyr" % (logA, age_gyr)
        else:
            age_str = r"$\log(A)=%.2f$; $%i$ Myr" % (logA, age_gyr * 10. ** 3.)

        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(1, 1,
                               left=0.17, right=0.95, bottom=0.15, top=0.95,
                               wspace=None, hspace=None,
                               width_ratios=None, height_ratios=None)
        ax = fig.add_subplot(gs[0])
        ax.imshow(hess, cmap=mpl.cm.gray_r, norm=None,
                  aspect=aspect,
                  interpolation='none',
                  extent=extent, origin=origin,
                  alpha=None, vmin=None, vmax=None)
        ax.set_xlabel(cmd['x_label'])
        ax.set_ylabel(cmd['y_label'])
        # title = synth_path
        # title = title.replace("_", "\_")
        ax.text(0.1, 0.9, age_str, ha='left', va='baseline',
                transform=ax.transAxes)
        ax.text(0.1, 0.8, z_str, ha='left', va='baseline',
                transform=ax.transAxes)
        gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
        canvas.print_figure(plot_path + "." + format, format=format, dpi=dpi)


class Lockfile(object):
    """Construct lockfiles for :class:`Synth`. Lockfiles tie several degenerate
    isochrones together, reducing the dimensionality of the star formation
    history space.

    Parameters
    ----------

    library_builder : :class:`isolibrary.LibraryBuilder` instance
        The instance of :class:`isolibrary.LibraryBuilder` used to prepare the
        isochrone library
    synth_dir : str
        Directory name of synthesized CMDs, relative to the StarFISH
        directory. E.g., `'synth'`.
    """
    def __init__(self, library_builder, synth_dir):
        super(Lockfile, self).__init__()
        self.library_builder = library_builder
        self.synth_dir = synth_dir
        self._index_isochrones()
        self._current_new_group_index = 1
        self._isoc_sel = []  # orders _index by isochrone group

    @property
    def full_synth_dir(self):
        return os.path.join(starfish_dir, self.synth_dir)

    def _index_isochrones(self):
        """Build an index of installated ischrones, noting filename, age,
        metallicity. The index includes an empty group index column.
        """
        # Read the isofile to get list of isochrones
        t = self.library_builder.read_isofile()
        paths = t['output_path']
        n_isoc = len(paths)
        # The _index lists isochrones and grouping info for lockfile
        dt = np.dtype([('age', np.float), ('Z', np.float), ('group', np.int),
                       ('path', 'S40'), ('name', 'S40'),
                       ('z_str', 'S4'), ('age_str', 'S5'), ('dt', np.float),
                       ('mean_group_age', np.float),
                       ('mean_group_z', np.float)])
        self._index = np.empty(n_isoc, dtype=dt)
        for i, p in enumerate(paths):
            z_str, age_str = os.path.basename(p)[1:].split('_')
            Z = float("0." + z_str)
            age = float(age_str)
            self._index['age'][i] = age
            self._index['Z'][i] = Z
            self._index['z_str'][i] = z_str
            self._index['age_str'][i] = age_str
            self._index['path'][i] = p
            self._index['name'][i] = " " * 40
            self._index['dt'][i] = np.nan
            self._index['mean_group_age'][i] = np.nan
            self._index['mean_group_z'][i] = np.nan
        self._index['group'][:] = 0

    @property
    def active_groups(self):
        """Returns a list of groups that have CMD planes prepared by synth."""
        active_groups = []
        names = np.unique(self._index['name'])
        for name in names:
            paths = glob.glob(os.path.join(starfish_dir, name + "*"))
            if len(paths) > 0:
                active_groups.append(name)
            else:
                logging.warning("Can't find %s" % name)
        return active_groups

    def lock_grid(self, age_grid, z_groups=None):
        """An easy-to-use method for locking isochrones according to an
        age and metallicity grid specified by the user.

        Parameters
        ----------
        age_grid : ndarray
            1D array of log(age) grid *edges*. The first age group spans
            from ``age_grid[0]`` to ``age_grid[1]``, while the last age group
            spans from ``age_grid[:2]`` to ``age_grid[:1]``. Thus ``age_grid``
            is 1-longer than the number of age bins.
        z_groups : list of tuples
            This allows metallicities to be locked together. Each metallicity
            group is a tuple in a list. The tuple consists of the ``z_code``
            for each metallicity (that is, the ``str`` ``XXXX``, giving the
            fractional part of the metallicity. Metallicities appearing
            alone are given as single-item tuples. Metallicites not included
            in the list are ommitted.

            If left as ``None``, then isochrones of distinct metallicities
            will not be locked together, and all metallicities will be used.
        """
        if z_groups is None:
            # Make a default listing of all Z groups, unbinned
            unique_z, unique_indices = np.unique(self._index['z_str'],
                                                 return_index=True)
            zvals = self._index['Z'][unique_indices]
            sort = np.argsort(zvals)
            unique_z = unique_z[sort]
            z_groups = [(zstr,) for zstr in unique_z]
        # print "z_groups", z_groups

        len_age_grid = len(age_grid)
        for z_group in z_groups:
            zsels = [np.where(self._index['z_str'] == zstr)[0]
                     for zstr in z_group]
            zsel = np.concatenate(zsels)
            if zsel.shape[0] == 0:
                logging.warning("No isochrones for z_group: %s" % z_group)
                continue
            ages = self._index['age'][zsel]
            # Bin ages in this metallicity group
            indices = np.digitize(ages, age_grid, right=False)
            # Unique bin values to iterate through
            unique_bin_vals = np.unique(indices)
            _all_indices = np.arange(len(self._index), dtype=np.int)
            for i in unique_bin_vals:
                # print "binval", binval
                if i == 0 or i == len_age_grid:
                    continue
                agesel = np.where(indices == i)[0]
                sel = np.copy(_all_indices[zsel][agesel])
                age_start = age_grid[i - 1]
                age_stop = age_grid[i]
                binages = self._index['age'][sel]
                binz = self._index['Z'][sel]
                mean_age = binages.mean()
                mean_z = binz.mean()
                dt = 10. ** age_stop - 10. ** age_start
                age_str = "%05.2f" % mean_age
                z_str = "%.4f" % mean_z
                z_str = z_str[2:]
                stemname = os.path.join(self.synth_dir,
                                        "z%s_%s" % (z_str, age_str))
                # print "stemname", stemname
                self._index['group'][sel] = self._current_new_group_index
                self._index['name'][sel] = stemname
                self._index['dt'][sel] = dt
                self._index['mean_group_age'][sel] = mean_age
                self._index['mean_group_z'][sel] = mean_z
                self._current_new_group_index += 1
                # Add these isochrones to the isochrone selector index
                for i in sel:
                    self._isoc_sel.append(i)
                # print self._index['dt'][sel]
                # print self._index['name'][sel]

    def lock_box(self, name, age_span, z_span, d_age=0.001, d_z=0.00001):
        """Lock together isochrones in a box in Age-Z space.

        Parameters
        ----------
        name : str
            Name of the isochrone group. By convension this is `zXXXX_YY.YY`
            where `XXXX` is the representative metallicity (fractional part)
            and `YY.YY` is the log(age). This name is for primarily for your
            record keeping/to help you identify synthesized CMDs.
        age_span : sequence, (2,)
            The (min, max) space of isochrone log(age) to include in group.
            Note that span will be broadened by +/-`d_age`, so actual grid
            values can be safely included. This age span will also be used to
            determine the timespan an isochrone group is effective for; this
            is used in star formation rate calculations.
        z_span : sequence, (2,)
            The (min, max) space of isochrone Z to include in group.
            Note that span will be broadened by +/-`d_z`, so actual grid
            values can be safely included.
        d_age : float
            Fuzz added to the box so that grid points at the edge of the age
            span are included.
        z_age : float
            Fuzz added to the box so that grid points at the edge of the Z
            span are included.
        """
        indices = np.where((self._index['age'] > min(age_span) - d_age)
                           & (self._index['age'] < max(age_span) + d_age)
                           & (self._index['Z'] > min(age_span) - d_z)
                           & (self._index['Z'] < max(age_span) + d_z))[0]
        self._index['group'][indices] = self._current_new_group_index
        stemname = os.path.join(self.synth_dir, name)
        self._index['name'][indices] = stemname
        binages = self._index['age'][indices]
        binz = self._index['Z'][indices]
        mean_age = binages.mean()
        mean_z = binz.mean()
        dt = 10. ** max(age_span) - 10. ** min(age_span)  # span in years
        self._index['dt'][indices] = dt
        self._index['mean_group_age'][indices] = mean_age
        self._index['mean_group_z'][indices] = mean_z
        self._current_new_group_index += 1
        # Add these isochrones to the isochrone selector index
        for i in indices:
            self._isoc_sel.append(i)

    def _include_unlocked_isochrones(self):
        """Creates single-isochrone groups for for isochrones that have
        not otherwise been grouped.
        """
        indices = np.where(self._index['group'] == 0)[0]
        grid_dt = self._estimate_age_grid()
        for idx in indices:
            name = "z%s_%s" % (self._index['z_str'][idx],
                               self._index['age_str'][idx])
            self._index['group'][idx] = self._current_new_group_index
            stemname = os.path.join(self.synth_dir, name)
            self._index['name'][idx] = stemname
            logage = self._index['age'][idx]
            dt = 10. ** (logage + grid_dt / 2.) \
                - 10. ** (logage - grid_dt / 2.)
            self._index['dt'][idx] = dt  # years
            self._index['mean_group_age'][idx] = self._index['age'][idx]
            self._index['mean_group_z'][idx] = self._index['Z'][idx]
            self._current_new_group_index += 1
            # Add these isochrones to the isochrone selector index
            self._isoc_sel.append(idx)

    def _estimate_age_grid(self):
        """Assuming that ischrones are sampled from a regular grid of log(age),
        this method finds the cell size of that log(age) grid.
        """
        unique_age_str, indices = np.unique(self._index['age_str'],
                                            return_index=True)
        age_grid = self._index['age'][indices]
        sort = np.argsort(age_grid)
        age_grid = age_grid[sort]
        diffs = np.diff(age_grid)
        dage = mode(diffs)[0][0]
        return float(dage)

    def write(self, path, include_unlocked=False):
        """Write the lockfile to path.

        Parameters
        ----------
        path : str
            Filename of lockfile, relative to StarFISH.
        include_unlocked : bool
            If `True` then any isochrones not formally included in a
            group will be automatically placed in singleton groups. If
            `False` then these isochrones are omitted from ``synth`` and
            other StarFISH computations.
        """
        self.lock_path = path
        if include_unlocked:
            self._include_unlocked_isochrones()
        self._write(path)

    def _write(self, path):
        """Write the lock file to `path`.

        Each row of the lockfile has the columns:

        - group id[int]
        - isoname[up to 40 characters]
        - synthfilestem [up to 40 characters]
        """
        full_path = os.path.join(starfish_dir, path)
        dirname = os.path.dirname(full_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Lockfile has just the isochrones needed in it; ordered by group
        sel = np.array(self._isoc_sel, dtype=np.int)
        lockdata = self._index[sel]

        t = Table(lockdata)
        t.write(full_path, format='ascii.fixed_width_no_header', delimiter=' ',
                bookend=False, delimiter_pad=None,
                include_names=['group', 'path', 'name'],
                formats={"group": "%03i", "path": "%s", "name": "%s"})

        # Also write the edited isofile
        self.synth_isofile_path = self.library_builder.isofile_path + ".synth"
        self.library_builder.write_edited_isofile(self.synth_isofile_path,
                                                  sel)

        # also make sure synth dir is ready
        full_synth_dir = os.path.join(starfish_dir, self.synth_dir)
        if not os.path.exists(full_synth_dir):
            os.makedirs(full_synth_dir)

    def write_cmdfile(self, path):
        """Create the ``cmdfile`` needed by the ``sfh`` program.

        Parameters
        ----------
        path : str
            Path where the ``cmdfile`` will be created.
        """
        active_groups = self.active_groups
        ndata = np.empty(len(active_groups),
                         dtype=np.dtype([('Z', np.float),
                                         ('log(age)', np.float),
                                         ('path', 'S40')]))
        for j, groupname in enumerate(active_groups):
            i = np.where(self._index['name'] == groupname)[0][0]
            ndata['Z'][j] = self._index['mean_group_z'][i]
            ndata['log(age)'][j] = self._index['mean_group_age'][i]
            ndata['path'][j] = self._index['name'][i]
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        t = Table(ndata)
        t.write(path, format="ascii.fixed_width_no_header", delimiter=' ',
                bookend=False, delimiter_pad=None,
                names=['Z', 'log(age)', 'path'],
                formats={"Z": "%6.4f", "log(age)": "%5.2f", "path": "%s"})

    def write_holdfile(self, path):
        """Write the ``holdfile`` needed by the ``sfh`` program.

        .. note:: Currently this hold file places no 'holds' on the star
           formation history optimization.
        """
        active_groups = self.active_groups
        dt = np.dtype([('amp', np.float), ('Z', np.float),
                       ('log(age)', np.float)])
        ndata = np.empty(len(active_groups), dtype=dt)
        for j, groupname in enumerate(active_groups):
            i = np.where(self._index['name'] == groupname)[0][0]
            ndata['amp'][j] = 0.
            ndata['Z'][j] = self._index['mean_group_z'][i]
            ndata['log(age)'][j] = self._index['mean_group_age'][i]
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        t = Table(ndata)
        t.write(path, format="ascii.fixed_width_no_header", delimiter=' ',
                bookend=False, delimiter_pad=None,
                names=['amp', 'Z', 'log(age)'],
                formats={"amp": "%9.7f", "Z": "%6.4f", "log(age)": "%5.2f"})

    def group_dt(self):
        """Return an array of time spans associated with each isochrone group,
        in the same order as isochrones appear in the lockfile.

        The time spans are in years, and are used to determine star formation
        rates.
        """
        active_groups = self.active_groups
        dt = np.zeros(len(active_groups))
        for i, groupname in enumerate(active_groups):
            idx = np.where(self._index['name'] == groupname)[0][0]
            dt[i] = self._index['dt'][idx]
        return dt


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
