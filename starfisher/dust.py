#!/usr/bin/env python
# encoding: utf-8
"""
Classes for configuring extinction.

2015-04-07 - Created by Jonathan Sick
"""

import os

from pkg_resources import resource_stream, resource_exists
import numpy as np
from astropy.table import Table

from starfisher.pathutils import starfish_dir


class ExtinctionDistribution(object):
    """Create an extinction distribution file for :class:`Synth`.

    Synthesized stars will have extinction values drawn randomly from samples
    in the extintion distribution. Uniform extinction can be implemented by
    using only one extinction value.
    """
    def __init__(self):
        super(ExtinctionDistribution, self).__init__()
        self._extinction_array = None

    @property
    def samples(self):
        return self._extinction_array

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


class SF11ExtinctionCurve(object):
    """Extintion laws based on """
    def __init__(self):
        super(SF11ExtinctionCurve, self).__init__()
        path = "data/schlafly_finkbeiner_2011_table6.txt"
        assert resource_exists(__name__, path)
        self.data = Table.read(resource_stream(__name__, path),
                               delimiter='\t',
                               guess=False,
                               quotechar="'",
                               format="ascii.commented_header")

    def __getitem__(self, key):
        i = np.where(self.data['bandpass'] == key)[0][0]
        return self.data['R_V_3.1'][i]

    def extinction_ratio(self, band, ref_band='Landolt V'):
        return self[band] / self[ref_band]
