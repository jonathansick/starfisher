#!/usr/bin/env python
# encoding: utf-8
"""
Classes for configuring extinction.

2015-04-07 - Created by Jonathan Sick
"""

import os

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
