#!/usr/bin/env python
# encoding: utf-8
"""
Data structures for lockfiles.
"""

import os
import glob
import logging
import abc

import numpy as np
from scipy.stats import mode
from astropy.table import Table

from starfisher.pathutils import starfish_dir


class BaseLockfile(object):
    """Docstring for BaseLockfile. """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseLockfile, self).__init__()

    @property
    def full_synth_dir(self):
        return os.path.join(starfish_dir, self.synth_dir)

    @property
    def lock_path(self):
        return os.path.join(self.synth_dir, "lock.dat")

    @property
    def full_lock_path(self):
        return os.path.join(starfish_dir, self.lock_path)

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

    def mean_age_for_group(self, name):
        i = np.where(self._index['name'] == name)[0][0]
        return self._index['mean_group_age'][i]

    def mean_z_for_group(self, name):
        i = np.where(self._index['name'] == name)[0][0]
        return self._index['mean_group_z'][i]

    def _include_unlocked_isochrones(self):
        """Creates single-isochrone groups for for isochrones that have
        not otherwise been grouped.
        """
        indices = np.where(self._index['group'] == 0)[0]
        grid_dt = self._estimate_age_grid()
        self._current_new_group_index = self._index['group'].max() + 1
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

    def _estimate_age_grid(self):
        """Assuming that ischrones are sampled from a regular grid of log(age),
        this method finds the cell size of that log(age) grid.
        """
        if len(self._index) < 2:
            return 0.
        else:
            unique_age_str, indices = np.unique(self._index['age_str'],
                                                return_index=True)
            age_grid = self._index['age'][indices]
            sort = np.argsort(age_grid)
            age_grid = age_grid[sort]
            diffs = np.diff(age_grid)
            dage = mode(diffs)[0][0]
            return float(dage)

    def write(self, include_unlocked=False):
        """Write the lockfile to path.

        Parameters
        ----------
        include_unlocked : bool
            If `True` then any isochrones not formally included in a
            group will be automatically placed in singleton groups. If
            `False` then these isochrones are omitted from ``synth`` and
            other StarFISH computations.
        """
        if include_unlocked:
            self._include_unlocked_isochrones()
        self._write()

    def _write(self):
        """Write the lock file to `full_lock_path`.

        Each row of the lockfile has the columns:

        - group id[int]
        - isoname[up to 40 characters]
        - synthfilestem [up to 40 characters]
        """
        dirname = os.path.dirname(self.full_lock_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Lockfile has just the isochrones needed in it; ordered by group
        sel = self._build_isochrone_selector()
        lockdata = self._index[sel]

        t = Table(lockdata)
        t.write(self.full_lock_path,
                format='ascii.fixed_width_no_header', delimiter=' ',
                bookend=False, delimiter_pad=None,
                include_names=['group', 'path', 'name'],
                formats={"group": "%03i", "path": "%s", "name": "%s"})

        # Also write the edited isofile
        # FIXME verify this
        self.synth_isofile_path = self.library_builder.isofile_path + ".synth"
        self.library_builder.write_edited_isofile(self.synth_isofile_path,
                                                  sel)

    def _build_isochrone_selector(self):
        sel = []
        for g in self.active_groups:
            for i in np.where(self._index['group'] == g)[0]:
                sel.append(i)
        return np.array(sel, dtype=int)

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

    @property
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

    @property
    def group_polygons(self):
        return self._polygons

    def split_lockfile(self, n_cpu, include_unlocked=False):
        """Create :class:`SplitLockfile` instances that contain subsets of
        the groups.
        """
        group_numbers = np.unique(self._index['group'])
        group_numbers.sort()

        lockfiles = []
        for i in range(n_cpu):
            # Group indices to include in this lockfile
            groups = group_numbers[i::n_cpu]
            sels = []
            split_polygons = {}
            for g in groups:
                g = int(g)
                sels.append(np.where(self._index['group'] == g)[0])
                split_polygons[g] = self._polygons[g]
            sel = np.concatenate(sels)
            slf = SplitLockfile(i, self.isofile[sel], self.synth_dir,
                                self._index[sel], split_polygons)
            lockfiles.append(slf)
        return lockfiles


class Lockfile(BaseLockfile):
    """Construct lockfiles for :class:`Synth`. Lockfiles tie several degenerate
    isochrones together, reducing the dimensionality of the star formation
    history space.

    Parameters
    ----------
    isofile : :class:`astropy.table.Table` instance
        The isofile table, made with `LibraryBuilder.read_isofile()`.
    synth_dir : str
        Directory name of synthesized CMDs, relative to the StarFISH
        directory. E.g., `'synth'`.
    """
    def __init__(self, isofile, synth_dir):
        super(Lockfile, self).__init__()
        self.isofile = isofile
        self.synth_dir = synth_dir
        self._index_isochrones()
        self._current_new_group_index = 1
        self._polygons = {}

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

            For example::

                [('0150',), ('0190', '0240')]

            will make a group for just the `0150` metallicity isochrones,
            while grouping `0190` and `0240` together.

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

        all_z_codes = np.unique(self._index['z_str'])
        sort = np.argsort(all_z_codes)

        # Iterate over the age grid first
        for age_i in xrange(len(age_grid) - 1):
            age_min = age_grid[age_i]
            age_max = age_grid[age_i + 1]

            for group in z_groups:
                sels = []
                print "group", group
                for zcode in group:
                    sels.append(np.where((self._index['age'] >= age_min) &
                                         (self._index['age'] < age_max) &
                                         (self._index['z_str'] == zcode))[0])
                sel = np.concatenate(sels)
                if len(sel) == 0:
                    logging.warning("No isochrones found in group "
                                    "{0:.4f} {1:.4f} {2}".format(age_min,
                                                                 age_max,
                                                                 zcode))
                    continue

                # Compute statistics in this group
                mean_age = self._index['age'][sel].mean()
                mean_z = self._index['Z'][sel].mean()

                dt = 10. ** age_max - 10. ** age_min
                z_str = "{0:.4f}".format(mean_z)[2:]
                stemname = os.path.join(
                    self.synth_dir,
                    "z{0}_{1:05.2f}".format(z_str, mean_age))

                # Persist group data with the isochrones
                self._index['group'][sel] = self._current_new_group_index
                self._index['name'][sel] = stemname
                self._index['dt'][sel] = dt
                self._index['mean_group_age'][sel] = mean_age
                self._index['mean_group_z'][sel] = mean_z

                # Create a (multi)polygon from this group
                lock_poly = LockPolygon()
                for z_group in self._make_contig_z_groups(group):
                    print "z_group", z_group
                    zsel = np.concatenate(
                        [np.where(self._index['z_str'] == z)[0]
                         for z in z_group])
                    print "zsel", zsel
                    z_min = self._index['Z'][zsel].min()
                    z_max = self._index['Z'][zsel].max()
                    lock_poly.add_poly_for_range(age_min, age_max,
                                                 z_min, z_max)
                self._polygons[self._current_new_group_index] = lock_poly

                self._current_new_group_index += 1

    def _make_contig_z_groups(self, group):
        z_indices = np.unique(self._index['z_str'])
        z_indices.sort()
        contig_groups = []
        last_group_i = None
        for i, zi in enumerate(z_indices):
            if zi in group:
                if last_group_i is not None and ((i - last_group_i) == 1):
                    contig_groups[-1].append(zi)
                else:
                    contig_groups.append([zi])
                last_group_i = i

        return contig_groups

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
        indices = np.where((self._index['age'] >= min(age_span) - d_age)
                           & (self._index['age'] < max(age_span) + d_age)
                           & (self._index['Z'] >= min(z_span) - d_z)
                           & (self._index['Z'] < max(z_span) + d_z))[0]
        stemname = os.path.join(self.synth_dir, name)
        binages = self._index['age'][indices]
        binz = self._index['Z'][indices]
        mean_age = binages.mean()
        mean_z = binz.mean()
        dt = 10. ** max(age_span) - 10. ** min(age_span)  # span in years
        self._index['dt'][indices] = dt
        self._index['name'][indices] = stemname
        self._index['mean_group_age'][indices] = mean_age
        self._index['mean_group_z'][indices] = mean_z
        self._index['group'][indices] = self._current_new_group_index
        poly = LockPolygon()
        poly.add_poly_for_range(min(age_span) - d_age,
                                max(age_span) + d_age,
                                min(z_span) - d_z,
                                max(z_span) + d_z)
        self._polygons[self._current_new_group_index] = poly
        self._current_new_group_index += 1

    def _index_isochrones(self):
        """Build an index of installated ischrones, noting filename, age,
        metallicity. The index includes an empty group index column.
        """
        paths = self.isofile['output_path']
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


class SplitLockfile(BaseLockfile):
    """Docstring for SplitLockfile."""
    def __init__(self, i, isofile, synth_dir, index, polygons):
        super(SplitLockfile, self).__init__()
        self._i = i
        self.isofile = isofile
        self.synth_dir = synth_dir
        self._index = index
        self._polygons = polygons

    @property
    def lock_path(self):
        return os.path.join(self.synth_dir, "lock.{0:d}.dat".format(self._i))


class LockPolygon(object):
    """Multi-polygons to visualize lock groups."""
    def __init__(self):
        super(LockPolygon, self).__init__()
        self._polygons = []

    def add_polgon(self, poly):
        """Add a polygon with vertices defined as (log(age), z).
        """
        self._polygons.append(poly)

    def add_poly_for_range(self, log_age_min, log_age_max,
                           z_min, z_max):
        """Add a rectangular polygon.

        Parameters
        ----------
        log_age_min : float
            Minimum log(age).
        log_age_max : float
            Maximum log(age)
        z_min : float
            Minimum fractional metallicity
        z_max : float
            Maximum fractional metallicity
        """
        p = [[log_age_min, z_min],
             [log_age_min, z_max],
             [log_age_max, z_max],
             [log_age_max, z_min]]
        self._polygons.append(np.array(p))

    @property
    def logage_logzsol_verts(self):
        for poly in self._polygons:
            p = np.array(poly)
            p[:, 1] = np.log10(p[:, 1] / 0.019)
            yield p
