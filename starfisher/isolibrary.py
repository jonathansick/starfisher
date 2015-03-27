#!/usr/bin/env python
# encoding: utf-8
"""
This module helps manage the isochrone library, from preprocessing the
Padova isochrones to running `mklib`.
"""

import glob
import logging
import os
import subprocess

from astropy.table import Table

from starfisher.pathutils import starfish_dir, EnterStarFishDirectory


class LibraryBuilder(object):
    """Setup an isochrone library.

    Parameters
    ----------
    input_dir : str
        Directory where input files are stored for the StarFISH run, relative
        to the StarFISH directory. Typically this is `'input'`.
    isoc_src_dir : str
        Name of the directory with raw isochrones, relative to the root of
        the StarFISH directory.
    lib_dir : float
        Directory where the isochrones will be installed by ``mklib``,
        relative to the root of the StarFISH directory.
    faint : float
        Faint magnitude limit for output isochrone library (according to
        filter at `mag0` index). Should be several mag fainter than the
        data's faint limit.
    dmag : float
        Photometric distance between adjacent interpolated points.
    dmod : float
        Distance modulus (magnitudes).
    gamma : float
        Logarithmic IMF slope (Salpeter = -1.35).
    nmag : int
        Number of bandpasses.
    mag0 : int
        Index (1-based) of reference magnitude for `faint` and `msto`.
    iverb : int
        Verbosity of `mklib`.
        - 0 = silent
        - 1 = screen messages
        - 2 = extra output files
    """
    def __init__(self, input_dir, isoc_src_dir, lib_dir,
                 faint=30., dmag=0.005, dmod=0.,
                 gamma=-1.35, nmag=2, mag0=1, iverb=0):
        super(LibraryBuilder, self).__init__()
        self.input_dir = input_dir
        self.isoc_src_dir = isoc_src_dir
        self._iso_dir = lib_dir
        self._isofile_path = os.path.join(self.input_dir, "isofile")
        self._libdat_path = os.path.join(self.input_dir, "lib.dat")
        self.faint = faint
        self.dmag = dmag
        self.dmod = dmod
        self.gamma = gamma
        self.nmag = nmag
        self.mag0 = mag0
        self.iverb = iverb
        for dirname in (self.input_dir, self.isoc_src_dir, self._iso_dir):
            full_path = os.path.join(starfish_dir, dirname)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

    @property
    def isofile_path(self):
        """StarFish-relative path to the ``isofile`` used by ``mklib``."""
        return self._isofile_path

    @property
    def full_isofile_path(self):
        """Absolute path to the ``isofile`` used by ``mklib``."""
        return os.path.join(starfish_dir, self._isofile_path)

    @property
    def library_dir(self):
        """Path to directory with isochrones built by `mklib`."""
        return self._iso_dir

    @property
    def full_library_dir(self):
        """Path to directory with isochrones built by `mklib`."""
        return os.path.join(starfish_dir, self._iso_dir)

    def install(self):
        """Runs `mklib` to install the parsed isochrones into the isochrone
        library directory.
        """
        tbl = self._build_isofile_table()
        self._write_isofile(tbl)
        self._build_libdat()
        self._clean_isodir()
        with EnterStarFishDirectory():
            command = './mklib < {0}'.format(self._libdat_path)
            print(command)
            subprocess.call(command, shell=True)
        self._check_library()

    def write_edited_isofile(self, path, sel):
        """Write a new version of the `isofile`, included only isochrones
        included in the `sel` index.

        This method is intended to by used be `synth` to create a version
        of the `isofile` that only has as many isochrones as are included
        in the `lockfile`.

        Parameters
        ----------
        path : str
            Filepath of the new isofile, relative to starfish
        sel : ndarray
            Numpy index array, selecting isochrones.
        """
        # Read the actual isofile because it has already been edited
        # for bad isochrones
        t = self.read_isofile()
        t2 = t[sel]
        t2.write(os.path.join(starfish_dir, path),
                 format='ascii.no_header',
                 delimiter=' ')

    def _build_isofile_table(self):
        """Build an isofile, specifying each isochrone file.

        Parsed isochrone files are named `z001_06.60`, for example, meaning
        Z=0.001 and log(age/yr)=6.6.

        The isocfile's format is specified from the StarFISH manual:
        Each line in the file should contain the following space-delimited
        fields (the column format does not matter)::

            log(age) [real number],
            input raw isochrone file [up to 40 chars],
            output isochrone file [up to 40 chars],
            msto mag [real number]

        msto mag is the absolute magnitude of each isochrone’s MSTO point.
        If you do not know the MSTOs for your isochrones, you’ll need to
        identify them. Use the fact that at the MSTO, the occupation
        probability changes dramatically.
        """
        t = Table(names=('log(age)', 'path', 'output_path', 'msto'),
                  dtype=('f4', 'S40', 'S40', 'f4'))

        isoc_paths = glob.glob(os.path.join(starfish_dir,
                                            self.isoc_src_dir,
                                            "z*"))
        for p in isoc_paths:
            # exact Z and Age from filename convention
            z_str, age_str = os.path.basename(p)[1:].split('_')
            rel_path = os.path.relpath(p, starfish_dir)
            basepath = os.path.basename(p)
            output_path = os.path.join(self._iso_dir, basepath)
            t.add_row((float(age_str), rel_path, output_path, 100.))

        return t

    def _write_isofile(self, tbl):
        """Write the isofile table to `self.isofile_path`."""
        if os.path.exists(self.full_isofile_path):
            os.remove(self.full_isofile_path)
        tbl.write(self.full_isofile_path,
                  format='ascii.no_header',
                  delimiter=' ')

    def read_isofile(self):
        t = Table.read(self.full_isofile_path,
                       format='ascii.no_header',
                       names=['log(age)', 'path', 'output_path', 'msto'])
        return t

    def _build_libdat(self):
        """Build the library data file, used by `mklib`.
        """
        template = "{path}\n{faint:.3f}\n{dmag:.3f}\n{dmod:.3f}\n{gamma:.3f}"\
                   "\n{nmag:d}\n{mag0:d}\n{iverb:d}"
        datstr = template.format(path=self._isofile_path,
                                 faint=self.faint,
                                 dmag=self.dmag,
                                 dmod=self.dmod,
                                 gamma=self.gamma,
                                 nmag=self.nmag,
                                 mag0=self.mag0,
                                 iverb=self.iverb)
        full_path = os.path.join(starfish_dir, self._libdat_path)
        if os.path.exists(full_path):
            os.remove(full_path)
        with open(full_path, 'w') as f:
            f.write(datstr)

    def _clean_isodir(self):
        """Remove pre-existing isochrones from the isochrone installation dir,
        `iso/`.
        """
        paths = glob.glob(os.path.join(self.full_library_dir, 'z*'))
        for p in paths:
            os.remove(p)

    def _check_library(self):
        """Verifies that all isochrones written by `mklib` are valid (not
        all NaN). Removes those isochrones from the isofile if necessary.
        """
        t = self.read_isofile()
        remove_indices = []
        for i, row in enumerate(t):
            isvalid = self._check_lib_isochrone(row['output_path'])
            if not isvalid:
                remove_indices.append(i)
                logging.warning("Installed isochrone contains NaNs: %s"
                                % row['output_path'])
        if len(remove_indices) > 0:
            t.remove_rows(remove_indices)
            self._write_isofile(t)

    def _check_lib_isochrone(self, isocpath):
        """Check if `mklib` output isochrone file contains NaNs."""
        with open(os.path.join(starfish_dir, isocpath), 'r') as f:
            for line in f:
                if "nan" in line:
                    return False
                elif "NaN" in line:
                    return False
        return True
