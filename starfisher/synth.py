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
    crowding_path : str
        Path to the artificial star test file (in StarFISH format).
        A package such as `delphinus` can help make this file from `dolphot`
        photometry, for example.
    input_dir : str
        Directory where input files are stored for the StarFISH run.
        Typically this is `'input'`.
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
    """
    def __init__(self, library_builder, lockfile, input_dir,
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
        self.error_method = 1  # by default assume analytic errors
        self._cmds = []  # add_cmd() inserts data here

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
            Indices (1-based) of bands to form the x-axis. If `x_mag` is a
            float, then the x-axis is that magnitude. If `x_mag` is a
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
            x_str = "-".join([str(i) for i in x_mag])
        else:
            x_str = str(x_mag)
        if not isinstance(y_mag, int):
            y_str = "-".join([str(i) for i in y_mag])
        else:
            y_str = str(y_mag)
        cmd_def = {'x_mag': x_mag, 'y_mag': y_mag,
                "x_str": x_str, "y_str": y_str,
                "x_span": x_span, "y_span": y_span,
                "y_crowding_max": y_crowding_max,
                "suffix": suffix,
                "x_label": xlabel, "y_label": ylabel}
        self._cmds.append(cmd_def)

    def set_crowding_table(self, path, output_path, dbin, error_range,
            binsize, error_method=2):
        """Setup the artificial star test crowding table.

        Parameters
        ----------
        
        crowding_path : str
            Path to the artificial star test file (in StarFISH format).
            A package such as `delphinus` can help make this file from
            `dolphot` photometry, for example.
        output_path : str
            Path where `synth` where write the error lookup table.
        dbin : length-2 tuple
            Tuple of (x, y) size of crowding bins, in magnitudes.
        error_range : length-2 tuple
            Tuple of (error_min, error_max) span of acceptable magnitudes
            errors for an artificial star to be considered recovered.
        binsize : float
            Binsize of delta-magnitude histograms.
        error_method : int
            Flag specifying the method for applying errors to the synthetic
            CMD. Can be:

            - 0 for regular crowding table lookup
            - 2 for scatter crowding table lookup
        """
        self.error_method = error_method
        self.crowding_path = path
        self.crowding_output_path = output_path
        self._crowd_config = {
                "dbin": dbin,
                "error_range": error_range,
                "binsize": binsize}
        self.error_method = error_method

    def run_synth(self, include_unlocked=False):
        """Run the StarFISH `synth` code to create synthetic CMDs."""
        self._write(include_unlocked=include_unlocked)
        self._clean()
        subprocess.call("./synth < %s" % self._synth_config_path, shell=True)

    def _write(self, include_unlocked=False):
        """Write the `synth` input file."""
        self._synth_config_path = os.path.join(self.input_dir, "synth.dat")
        if os.path.exists(self._synth_config_path):
            os.remove(self._synth_config_path)

        lines = []
        lines.append(self.library_builder.isofile_path)

        self.lockfile.write(os.path.join(self.input_dir, "lock.dat"),
                include_unlocked=include_unlocked)
        lines.append(self.lockfile.lock_path)
        
        self.young_extinction.write(os.path.join(self.input_dir, "young.av"))
        lines.append(self.young_extinction.path)

        self.old_extinction.write(os.path.join(self.input_dir, "old.av"))
        lines.append(self.old_extinction.path)

        lines.append(self.crowding_path)
        lines.append(self.crowding_output_path)

        lines.append(str(self.library_builder.nmag))
        lines.append(str(len(self._cmds)))
        lines.append(str(self.library_builder.mag0))

        lines.append(str(self.dpix))

        for cmd in self._cmds:
            lines.append(cmd['x_str'])
            lines.append(cmd['y_str'])
            lines.append("%.2f" % min(cmd['x_span']))
            lines.append("%.2f" % max(cmd['x_span']))
            lines.append("%.2f" % min(cmd['y_span']))
            lines.append("%.2f" % cmd['y_crowding_max'])
            lines.append("%.2f" % max(cmd['y_span']))
            lines.append(cmd['suffix'])

        lines.append(str(min(self._crowd_config['dbin'])))
        lines.append(str(max(self._crowd_config['dbin'])))
        lines.append(str(min(self._crowd_config['error_range'])))
        lines.append(str(max(self._crowd_config['error_range'])))
        lines.append(str(self._crowd_config['binsize']))

        for av_ratio in self.rel_extinction:
            lines.append("%.3f" % av_ratio)

        lines.append(str(self.verb))

        if self.interp_err:
            lines.append("1")
        else:
            lines.append("0")

        lines.append(str(self.error_method))

        lines.append(str(self.nstars))
        lines.append(str(self.seed))
        lines.append("%.2f" % min(self.mass_span))
        lines.append("%.2f" % max(self.mass_span))
        lines.append("%.2f" % self.library_builder.gamma)
        lines.append("%.2f" % self.library_builder.faint)
        lines.append("%.2f" % self.fbinary)

        txt = "\n".join(lines)
        with open(self._synth_config_path, 'w') as f:
            f.write(txt)

    def _clean(self):
        """Remove existing synthetic CMDs."""
        synthdir = self.lockfile.synth_dir
        paths = glob.glob(os.path.join(synthdir, "z*"))
        for path in paths:
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
        t = Table.read(self.library_builder.isofile_path,
                format='ascii.no_header',
                names=['log(age)', 'path', 'output_path', 'msto'])
        for row in t:
            isocpath = row['output_path']
            baseisocpath = os.path.basename(isocpath)
            for cmd in self._cmds:
                synth_path = os.path.join(self.lockfile.synth_dir,
                        baseisocpath + cmd['suffix'])
                plot_path = os.path.join(plotdir,
                        baseisocpath + cmd['suffix'])
                if not os.path.exists(synth_path):
                    logging.error("%s does not exist" % synth_path)
                    continue
                self._plot_hess(synth_path, plot_path, cmd, **plot_args)

    def _plot_hess(self, synth_path, plot_path, cmd, format="png", dpi=300,
            figsize=(4, 4), flipx=False, flipy=False, aspect='auto'):
        """Plot a Hess diagram for a single synthesized image."""
        indata = np.loadtxt(synth_path)
        nx = int((max(cmd['x_span']) - min(cmd['x_span'])) / self.dpix)
        ny = int((max(cmd['y_span']) - min(cmd['y_span'])) / self.dpix)
        hess = indata.reshape((ny, nx), order='C')

        # extent format is (left, right, bottom, top)
        if flipx:
            extent = [max(cmd['x_span']), min(cmd['x_span'])]
        else:
            extent = [min(cmd['x_span']), max(cmd['x_span'])]
        if flipy:
            extent.extend([max(cmd['y_span']), min(cmd['y_span'])])
        else:
            extent.extend([min(cmd['y_span']), max(cmd['y_span'])])
        if flipy:
            origin = 'lower'
        else:
            origin = 'upper'

        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(1, 1,
            left=0.15, right=0.95, bottom=0.15, top=0.95,
            wspace=None, hspace=None, width_ratios=None, height_ratios=None)
        ax = fig.add_subplot(gs[0])
        ax.imshow(hess, cmap=mpl.cm.gray_r, norm=None,
                aspect=aspect,
                interpolation='none',
                extent=extent, origin=origin,
                alpha=None, vmin=None, vmax=None)
        ax.set_xlabel(cmd['x_label'])
        ax.set_ylabel(cmd['y_label'])
        title = synth_path
        title = title.replace("_", "\_")
        ax.text(0.1, 0.9, title, ha='left', va='baseline',
                transform=ax.transAxes)
        gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
        canvas.print_figure(plot_path + "." + format, format=format, dpi=dpi)


class Lockfile(object):
    """Construct lockfiles for :class:`Synth`. Lockfiles tie several degenerate
    isochrones together, reducing the dimensionality of the star formation
    history space.

    Parameters
    ----------

    synth_dir : str
        Directory name of synthesized CMDs. E.g., `'synth'`.
    isofile_path : str
        Path to the `isofile` created by :class:`isolibrary.LibraryBuilder`.
    lib_dir : str
        Directory where isochrones are installed by
        :class:`isolibrary.LibraryBuilder`.
    """
    def __init__(self, synth_dir, isofile_path, lib_dir):
        super(Lockfile, self).__init__()
        self.iso_dir = lib_dir  # directory with installed isochrones
        self.synth_dir = synth_dir
        self.isofile_path = isofile_path
        self._index_isochrones()
        self._current_new_group_index = 1

    def _index_isochrones(self):
        """Build an index of installated ischrones, noting filename, age,
        metallicity. The index includes an empty group index column.
        """
        # Read the isofile to get list of isochrones
        t = Table.read(self.isofile_path, format='ascii.no_header',
                names=['log(age)', 'path', 'output_path', 'msto'])
        paths = t['output_path']
        n_isoc = len(paths)
        # The _index lists isochrones and grouping info for lockfile
        dt = np.dtype([('age', np.float), ('Z', np.float), ('group', np.int),
            ('path', 'S40'), ('name', 'S40'),
            ('z_str', 'S4'), ('age_str', 'S5'), ('dt', np.float)])
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
        self._index['group'][:] = 0

    @property
    def active_groups(self):
        """Returns a list of groups that have CMD planes prepared by synth."""
        active_groups = []
        names = np.unique(self._index['name'])
        for name in names:
            paths = glob.glob(name + "*")
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
        if z_groups == None:
            # Make a default listing of all Z groups, unbinned
            unique_z, unique_indices = np.unique(self._index['z_str'],
                    return_index=True)
            zvals = self._index['Z'][unique_indices]
            sort = np.argsort(zvals)
            unique_z = unique_z[sort]
            z_groups = [(zstr,) for zstr in unique_z]
        # print "z_groups", z_groups

        for z_group in z_groups:
            zsels = [np.where(self._index['z_str'] == zstr)[0]
                for zstr in z_group]
            zsel = np.concatenate(zsels)
            ages = self._index['age'][zsel]
            # Bin ages in this metallicity group
            indices = np.digitize(ages, age_grid,right=False)
            # Unique bin values to iterate through
            unique_bin_vals, inverse_indices = np.unique(indices,
                    return_inverse=True)
            # print "z_group", z_group
            # print "ages", len(ages), ages
            # print "indices", len(indices), indices
            # print "unique_bin_vals", len(unique_bin_vals), unique_bin_vals
            # print "inverse_indices", len(inverse_indices), inverse_indices
            _all_indices = np.arange(len(self._index), dtype=np.int)
            for i, binval in enumerate(unique_bin_vals):
                # print "binval", binval
                agesel = np.where(indices == binval)[0]
                sel = np.copy(_all_indices[zsel][agesel])
                age_start = age_grid[i]
                age_stop = age_grid[i + 1]
                # print "agesel", agesel
                # print "age range", age_start, age_stop
                # print "ages", self._index['age_str'][sel]
                # print "metallicities", self._index['z_str'][sel]
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
                self._current_new_group_index += 1
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
        dt = 10. ** max(age_span) - 10. ** min(age_span)  # span in years
        self._index['dt'][indices] = dt
        self._current_new_group_index += 1

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
            dt = 10. ** (logage + grid_dt / 2.) - 10. ** (logage - grid_dt / 2.)
            self._index['dt'][idx] = dt  # years
            self._current_new_group_index += 1

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
            Filename of lockfile.
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
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)

        t = Table(self._index)
        t.write(path, format='ascii.fixed_width_no_header', delimiter=' ',
                bookend=False, delimiter_pad=None,
                include_names=['group', 'path', 'name'],
                formats={"group": "%03i", "path": "%s", "name": "%s"})

        # also make sure synth dir is ready
        if not os.path.exists(self.synth_dir):
            os.makedirs(self.synth_dir)

    def write_cmdfile(self, path):
        """Create the ``cmdfile`` needed by the ``sfh`` program.
        
        Parameters
        ----------

        path : str
            Path where the ``cmdfile`` will be created.
        """
        ngroups = self._index.shape[0]
        active_groups = self.active_groups
        ndata = np.empty(len(active_groups), dtype=np.dtype([('Z', np.float),
            ('log(age)', np.float), ('path', 'S40')]))
        j = 0
        for i in xrange(ngroups):
            if self._index['name'][i] in active_groups:
                ndata['Z'][j] = self._index['Z'][i]
                ndata['log(age)'][j] = self._index['age'][i]
                ndata['path'][j] = self._index['name'][i]
                j += 1
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)
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
        ngroups = self._index.shape[0]
        active_groups = self.active_groups()
        ndata = np.empty(len(active_groups), dtype=np.dtype([('amp', np.float),
            ('Z', np.float), ('log(age)', np.float)]))
        j = 0
        for i in xrange(ngroups):
            if self._index['name'][i] in active_groups:
                ndata['amp'][j] = 0.
                ndata['Z'][j] = self._index['Z'][i]
                ndata['log(age)'][j] = self._index['age'][i]
                j += 1
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)
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

    Attributes
    ----------

    path : str
        Path to the extinction file (available once :meth:`write` is called).
    """
    def __init__(self):
        super(ExtinctionDistribution, self).__init__()
        self.path = None
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
            Path where extinction file is written.
        """
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)

        t = Table([self._extinction_array], names=['A'])
        t.write(path, format='ascii.no_header', delimiter=' ')
        self.path = path


def main():
    pass


if __name__ == '__main__':
    main()
