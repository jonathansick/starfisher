#!/usr/bin/env python
# encoding: utf-8
"""
Handle photometric crowding definitions.
"""
import abc
import os

import numpy as np

from starfisher.pathutils import starfish_dir


class BaseCrowdingTable(object):
    """Base class for crowding specification tables (used with synth).

    Parameters
    ----------
    path : str
        Path relative to starfish.
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
    __metaclass__ = abc.ABCMeta

    def __init__(self, path,
                 dbin=(0.5, 0.5), error_range=(-1, 1.),
                 binsize=0.1, error_method=2):
        super(BaseCrowdingTable, self).__init__()
        self._path = path
        self._error_method = error_method
        self._dbin = dbin
        self._error_range = error_range
        self._binsize = binsize

    @property
    def path(self):
        return self._path

    @property
    def full_path(self):
        return os.path.join(starfish_dir, self.path)

    @property
    def config_section(self):
        lines = []
        lines.append(str(min(self._dbin)))
        lines.append(str(max(self._dbin)))
        lines.append(str(min(self._error_range)))
        lines.append(str(max(self._error_range)))
        lines.append(str(self._binsize))
        return lines

    @property
    def error_method(self):
        return str(self._error_method)


class ExtantCrowdingTable(BaseCrowdingTable):
    """Crowding table wrapper for a pre-built crowdfile.

    Parameters
    ----------
    path : str
        Path relative to starfish.
    **args : dict
        Arguments for :class:`BaseCrowdingTable`.
    """
    def __init__(self, path, **args):
        super(ExtantCrowdingTable, self).__init__(path, **args)


class MockNullCrowdingTable(BaseCrowdingTable):
    """Make a mock crowding table where stars have no errors."""
    def __init__(self, path, n_bands, mag_range=(10., 35.), n_stars=100000,
                 **kwargs):
        super(MockNullCrowdingTable, self).__init__(path, **kwargs)
        self._n_bands = n_bands
        self._n_stars = n_stars
        self._range = mag_range
        self._write()

    def _write(self):
        dt = [('ra', float), ('dec', float)]
        for i in xrange(self._n_bands):
            dt.append(('mag{0:d}'.format(i), float))
            dt.append(('dmag{0:d}'.format(i), float))
        data = np.zeros(self._n_stars, dtype=np.dtype(dt))
        for i in xrange(self._n_bands):
            mag_label = "mag{0:d}".format(i)
            data[mag_label][:] = (max(self._range) - min(self._range)) \
                * np.random.random_sample(self._n_stars) + min(self._range)
        np.savetxt(self.full_path, data, fmt='%.5e', delimiter=' ')
