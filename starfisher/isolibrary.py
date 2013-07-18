#!/usr/bin/env python
# encoding: utf-8
"""
This module helps manage the isochrone library, from preprocessing the
Padova isochrones to running `mklib`.
"""

import glob
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
        Name of the directory with raw isochrones, relative to the root of
        the StarFISH directory.
    """
    def __init__(self, input_dir):
        super(LibraryBuilder, self).__init__()
        self.input_dir = input_dir

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

        isoc_paths = glob.glob(os.path.join(self.input_dir, "z*"))
        for p in isoc_paths:
            # exact Z and Age from filename convention
            z_str, age_str = os.path.basename(p)[1:].split('_')
            basepath = os.path.basename(p)
            output_path = os.path.join("iso", basepath)
            t.add_row((float(age_str), p, output_path, 100.))
        
        writepath = os.path.join(self.input_dir, "isofile")
        if os.path.exists(writepath): os.remove(writepath)
        t.write(writepath, format='ascii.no_header', delimiter=' ')

    def install(self):
        """Runs `mklib` to install the parsed isochrones into StarFISH's
        `iso/` directory.
        """
        self._build_isofile()
        subprocess.call('./mklib < input/lib.dat', shell=True)




def main():
    pass


if __name__ == '__main__':
    main()
