#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `synth`, the program for creating eigen-CMDs for each
isochrone and CMD plane.
"""

import os
import subprocess

import numpy as np
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
    analytic_errs : bool
        Set to `True` to use analytic errors rather than artificial star tests.
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
            dpix=0.05, nstars=1000000, verb=3,
            interp_err=True, analytic_errs=False,
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
        self.analytic_errs = analytic_errs
        self.seed = seed
        self.mass_span = mass_span
        self.fbinary = fbinary
        self._cmds = []  # add_cmd() inserts data here

    def add_cmd(self, x_mag, y_mag, x_span, y_span, y_crowding_max, suffix):
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
                "suffix": suffix}
        self._cmds.append(cmd_def)

    def set_crowding_table(self, path, output_path, dbin, error_range,
            binsize):
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
        """
        self.crowding_path = path
        self.crowding_output_path = output_path
        self._crowd_config = {
                "dbin": dbin,
                "error_range": error_range,
                "binsize": binsize}
        self.analytic_errs = False

    def run_synth(self):
        """Run the StarFISH `synth` code to create synthetic CMDs."""
        self._write()
        subprocess.call("./synth < %s" % self._synth_config_path, shell=True)

    def _write(self):
        """Write the `synth` input file."""
        self._synth_config_path = os.path.join(self.input_dir, "synth.dat")
        if os.path.exists(self._synth_config_path):
            os.remove(self._synth_config_path)

        lines = []
        lines.append(self.library_builder.isofile_path)

        self.lockfile.write(os.path.join(self.input_dir, "lock.dat"))
        lines.append(self.lockfile.lock_path)
        
        self.young_extinction.write(os.path.join(self.input_dir, "young.av"))
        lines.append(self.young_extinction.path)

        self.old_extinction.write(os.path.join(self.input_dir, "old.av"))
        lines.append(self.old_extinction.path)

        lines.append(self.crowding_path)
        lines.append(self.crowding_output_path)

        lines.append(str(self.library_builder.nmag))
        lines.append(str(len(self._cmds)))

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

        if self.analytic_errs:
            lines.append("1")
        else:
            lines.append("0")

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
        self._current_new_group_index = 0

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
            ('z_str', 'S4'), ('age_str', 'S5')])
        self._index = np.empty(n_isoc, dtype=dt)
        for i, p in enumerate(paths):
            z_str, age_str = os.path.basename(p)[1:].split('_')
            Z = float("0." + z_str)
            age = float(age_str)
            print p, Z, age
            self._index['age'][i] = age
            self._index['Z'][i] = Z
            self._index['z_str'][i] = z_str
            self._index['age_str'][i] = age_str
            self._index['path'][i] = p
            self._index['name'][i] = " " * 40
        self._index['group'][:] = 0

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
            values can be safely included.
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
        self._current_new_group_index += 1

    def _include_unlocked_isochrones(self):
        """Creates single-isochrone groups for for isochrones that have
        not otherwise been grouped.
        """
        indices = np.where(self._index['group'] == 0)[0]
        for idx in indices:
            name = "z%s_%s" % (self._index['z_str'][idx],
                    self._index['age_str'][idx])
            self._index['group'][idx] = self._current_new_group_index
            stemname = os.path.join(self.synth_dir, name)
            self._index['name'][idx] = stemname
            self._current_new_group_index += 1

    def write(self, path):
        """Write the lockfile to path."""
        self.lock_path = path
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
        t.write(path, format='ascii.no_header', delimiter=' ',
                include_names=['group', 'path', 'name'])

        # also make sure synth dir is ready
        if not os.path.exists(self.synth_dir):
            os.makedirs(self.synth_dir)


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