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


class LibraryBuilder(object):
    """Setup an ischrone library.
    
    .. note: Note that this class assumes it is being called from the root of
       the StarFISH directory.

    Parameters
    ----------

    input_dir : str
        Directory where input files are stored for the StarFISH run.
        Typically this is `'input'`.
    isoc_src_dir : str
        Name of the directory with raw isochrones, relative to the root of
        the StarFISH directory.
    """
    def __init__(self, input_dir, isoc_src_dir):
        super(LibraryBuilder, self).__init__()
        self.input_dir = input_dir
        self.isoc_src_dir = isoc_src_dir
        self._isofile_path = None
        self._iso_dir = None
        self._libdat_pat = None
        for dirname in (self.input_dir, self.isoc_src_dir):
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    @property
    def isofile_path(self):
        """Path to the ``isofile`` used by ``mklib``."""
        return self._isofile_path

    @property
    def library_dir(self):
        """Path to directory with isochrones built by `mklib`."""
        return self._iso_dir

    def _build_isofile(self):
        """Build an isofile, specifying each isochrone file.
        
        Parsed isochrone files are named `z001_06.60`, for example, meaning
        Z=0.001 and log(age/yr)=6.6.

        The isocfile's format is specified from the StarFISH manual:
        Each line in the file should contain the following space-delimited
        fields (the column format does not matter)::

            log(age) [real number], input raw isochrone file [up to 40 chars], output isochrone file [up to 40 chars], msto mag [real number]

        msto mag is the absolute magnitude of each isochrone’s MSTO point.
        If you do not know the MSTOs for your isochrones, you’ll need to
        identify them. Use the fact that at the MSTO, the occupation
        probability changes dramatically.
        """
        t = Table(names=('log(age)', 'path', 'output_path', 'msto'),
               dtypes=('f4', 'S40', 'S40', 'f4'))

        isoc_paths = glob.glob(os.path.join(self.isoc_src_dir, "z*"))
        for p in isoc_paths:
            # exact Z and Age from filename convention
            z_str, age_str = os.path.basename(p)[1:].split('_')
            basepath = os.path.basename(p)
            output_path = os.path.join(self._iso_dir, basepath)
            t.add_row((float(age_str), p, output_path, 100.))
        
        self._isofile_path = os.path.join(self.isoc_src_dir, "isofile")
        self._write_isofile(t)

    def _write_isofile(self, tbl):
        """Write the isofile table to `self.isofile_path`."""
        if os.path.exists(self._isofile_path): os.remove(self._isofile_path)
        tbl.write(self._isofile_path, format='ascii.no_header', delimiter=' ')

    def _build_libdat(self, faint=30., dmag=0.005, dmod=0.,
            gamma=-1.35, nmag=2, mag0=1, iverb=0):
        """Build the library data file, used by `mklib`.

        Parameters
        ----------

        faint : float
            Faint magnitude limit for output isochrone library (according to
            filter at `mag0` index). Should be several mag fainter than the
            data's faint limit.
        dmag : float
            Photometric distance between adjacent interpolated points.
        dmod : float
            Distance modulus (magnitudes).
        gamma : float
            Logarithmic IMF sloap (Salpeter = -1.35).
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
        self.faint = faint
        self.dmag = dmag
        self.dmod = dmod
        self.gamma = gamma
        self.nmag = nmag
        self.mag0 = mag0
        datstr = "%s\n%.3f\n%.3f\n%.3f\n%.3f\n%i\n%i\n%i" % \
                (self._isofile_path, faint, dmag, dmod, gamma,
                nmag, mag0, iverb)
        self._libdat_path = os.path.join(self.input_dir, "lib.dat")
        if os.path.exists(self._libdat_path): os.remove(self._libdat_path)
        with open(self._libdat_path, 'w') as f:
            f.write(datstr)

    def _clean_isodir(self):
        """Remove pre-existing isochrones from the isochrone installation dir,
        `iso/`.
        """
        paths = glob.glob(os.path.join(self._iso_dir, 'z*'))
        for p in paths:
            os.remove(p)

    def install(self, lib_dir='iso', **kwargs):
        """Runs `mklib` to install the parsed isochrones into the isochrone
        library directory.

        Parameters
        ----------

        lib_dir : float
            Directory where the isochrones will be installed by ``mklib``.
        faint : float
            Faint magnitude limit for output isochrone library (according to
            filter at `mag0` index). Should be several mag fainter than the
            data's faint limit.
        dmag : float
            Photometric distance between adjacent interpolated points.
        dmod : float
            Distance modulus (magnitudes).
        gamma : float
            Logarithmic IMF sloap (Salpeter = -1.35).
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
        self._iso_dir = lib_dir
        if not os.path.exists(self._iso_dir):
            os.makedirs(self._iso_dir)
        self._build_isofile()
        self._build_libdat(**kwargs)
        self._clean_isodir()
        subprocess.call('./mklib < %s' % self._libdat_path, shell=True)
        self._check_library()

    def _check_library(self):
        """Verifies that all isochrones written by `mklib` are valid (not
        all NaN). Removes those isochrones from the isofile if necessary.
        """
        t = Table.read(self.isofile_path, format='ascii.no_header',
                names=['log(age)', 'path', 'output_path', 'msto'])
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
        with open(isocpath, 'r') as f:
            for line in f:
                if "nan" in line:
                    return False
        return True



def main():
    pass


if __name__ == '__main__':
    main()
