#!/usr/bin/env python
# encoding: utf-8
"""
Simulate Hess diagrams for arbitrary star formation histories.

2015-04-08 - Created by Jonathan Sick
"""

import os
import numpy as np

from starfisher.pathutils import starfish_dir
from starfisher.hess import read_hess


class SimHess(object):
    """Builds Hess diagrams from synthetic SSP hess diagrams made by `synth`.

    Parameters
    ----------
    synth : :class:`starfisher.synth.Synth`
        A `Synth` instance where simulated Hess diagrams have been pre-built.
    colorplane : :class:`starfisher.plane.ColorPlane`
        The `ColorPlane` that will be simulated. This `ColorPlane` must be
        in `Synth`.
    amplitudes : ndarray
        Star formation amplitudes for each isochrone (group).
    """
    def __init__(self, synth, colorplane, amplitudes):
        super(SimHess, self).__init__()
        self._synth = synth
        self._plane = colorplane
        self._amps = amplitudes

        assert self._plane in self._synth._cmds
        assert len(self._amps) == len(self._synth.lockfile.active_groups)

        # build hess diagram
        self._h = self._build_hess()

    @property
    def hess(self):
        """The Hess diagram as numpy array."""
        return self._h

    @property
    def origin(self):
        return 'lower'

    @property
    def extent(self):
        return self._plane.extent

    def _build_hess(self):
        # Co-add synth hess diagrams, weighted by amplitude
        synth_hess = self._read_synth_hess()
        hess_stack = np.dstack(synth_hess)
        A = np.swapaxes(np.atleast_3d(self._amps), 1, 2)
        hess_stack = A * hess_stack
        return np.sum(hess_stack, axis=2)

    def _read_synth_hess(self):
        if self._plane.is_cmd:
            flipy = True
        else:
            flipy = False

        hesses = []
        for name in self._synth.lockfile.active_groups:
            synth_path = os.path.join(starfish_dir, name + self._plane.suffix)
            h, _, _ = read_hess(synth_path,
                                self._plane.x_span, self._plane.y_span,
                                self._plane.dpix,
                                flipy=flipy)
            hesses.append(h)
        return hesses