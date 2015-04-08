#!/usr/bin/env python
# encoding: utf-8
"""
Simulate Hess diagrams for arbitrary star formation histories.

2015-04-08 - Created by Jonathan Sick
"""


class SimHess(object):
    """Builds Hess diagrams from synthetic SSP hess diagrams made by `synth`.

    Parameters
    ----------
    synth : :class:`starfisher.synth.Synth`
        A `Synth` instance where simulated Hess diagrams have been pre-built.
    colorplane : :class:`starfisher.plane.ColorPlane`
        The `ColorPlane` that will be simulated. This `ColorPlane` must be
        in `Synth`.
    """
    def __init__(self, synth, colorplane):
        super(SimHess, self).__init__()
        self._synth = synth
        self._plane = colorplane

        assert self._plane in self._synth.cmds
