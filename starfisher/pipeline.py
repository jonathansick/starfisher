#!/usr/bin/env python
# encoding: utf-8
"""
Pipeline for StarFISH runs as a class with multiple inheritance of
computational components.
"""

import os
import abc
from collections import OrderedDict
from glob import glob

import numpy as np
from astropy.coordinates import Distance
import astropy.units as u

from starfisher import LibraryBuilder
from starfisher import SimHess
from starfisher import Synth
from starfisher import ExtinctionDistribution
from starfisher import MockNullCrowdingTable
from starfisher.plots import plot_hess

STARFISH = os.getenv("STARFISH")


class PipelineBase(object):
    """Abstract baseclass for running StarFISH pipelines."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.root_dir = kwargs.pop('root_dir')

        # StarFISH product directories
        self.isoc_dir = os.path.join(self.root_dir, 'isoc')
        self.lib_dir = os.path.join(self.root_dir, 'lib')
        self.synth_dir = os.path.join(self.root_dir, 'synth')

        # result caches
        self.fits = OrderedDict()
        self._solution_tables = {}

        print "PipelineBase", kwargs
        if len(kwargs) > 0:
            print "Uncaught arguments:", kwargs
        super(PipelineBase, self).__init__()

        dirs = (self.isoc_dir, self.lib_dir, self.synth_dir)
        for d in dirs:
            if not os.path.exists(os.path.join(STARFISH, d)):
                os.makedirs(os.path.join(STARFISH, d))

        print self.isoc_dir, self.lib_dir, self.synth_dir

        # Now run the pipeline
        self.setup_isochrones()
        print "self.builder.full_isofile_path", self.builder.full_isofile_path
        self.build_lockfile()
        self.build_crowding()
        self.build_extinction()
        self.run_synth()

    def run_synth(self):
        full_synth_dir = os.path.join(STARFISH, self.synth_dir)

        self.synth = Synth(self.synth_dir,
                           self.builder,
                           self.lockfile,
                           self.crowd,
                           self.rel_extinction,
                           young_extinction=self.young_av,
                           old_extinction=self.old_av,
                           planes=self.all_planes,
                           mass_span=(0.08, 150.),
                           nstars=10000000)
        existing_synth = len(glob(
            os.path.join(full_synth_dir, "z*"))) == 0
        if existing_synth:
            self.synth.run_synth(n_cpu=4, clean=False)

    def plot_sim_hess(self, ax, plane_key):
        plane = self.planes[plane_key]
        sim = self.get_sim_hess(plane_key)
        plot_hess(ax, sim.hess, plane, sim.origin,
                  imshow_args=None)


class IsochroneSetBase():
    """Abstract baseclass for pipeline components that obtain isochrones."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "IsochroneSetBase", kwargs
        super(IsochroneSetBase, self).__init__(**kwargs)

    @abc.abstractproperty
    def bands(self):
        """Key names of bandpasses."""
        pass

    @property
    def n_bands(self):
        return len(self.bands)

    @abc.abstractproperty
    def distance(self):
        return Distance(0. * u.kpc)

    @abc.abstractmethod
    def setup_isochrones(self):
        self.builder = LibraryBuilder(self.isoc_dir,
                                      self.lib_dir,
                                      nmag=self.n_bands,
                                      dmod=self.distance.distmod.value,
                                      iverb=3)
        if not os.path.exists(self.builder.full_isofile_path):
            self.builder.install()


class DatasetBase(object):
    """Abstract baseclass for pipeline components that write observational
    data.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "DatasetBase", kwargs
        super(DatasetBase, self).__init__(**kwargs)

    @abc.abstractmethod
    def write_phot(self, x_band, y_band, data_root, suffix):
        pass


class PlaneBase(object):
    """Abstract baseclass for defining ColorPlane sets in a pipeline."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self._sim_hess_planes = {}  # cache
        print "PlaneBase", kwargs
        super(PlaneBase, self).__init__(**kwargs)

    @abc.abstractproperty
    def planes(self):
        pass

    def get_sim_hess(self, key):
        if key not in self._sim_hess_planes:
            sh = SimHess(self.synth, self.planes[key],
                         np.ones(len(self.lockfile.active_groups)))
            self._sim_hess_planes[key] = sh
        return self._sim_hess_planes[key]

    @property
    def all_planes(self):
        return [p for k, p in self.planes.iteritems()]


class LockBase(object):
    """Abstract baseclass for lockfile management."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "LockBase", kwargs
        super(LockBase, self).__init__(**kwargs)

    @abc.abstractmethod
    def build_lockfile(self):
        pass


class CrowdingBase(object):
    """Abstract baseclass for managing crowding files."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(CrowdingBase, self).__init__(**kwargs)

    @abc.abstractmethod
    def build_crowding(self):
        # NOTE null crowding by default
        path = os.path.join(self.synth_dir, "crowding.dat")
        self.crowd = MockNullCrowdingTable(path, self.n_bands)


class ExtinctionBase(object):
    """Abstract baseclass for managing extinction files."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ExtinctionBase, self).__init__(**kwargs)

    @abc.abstractmethod
    def build_extinction(self):
        # NOTE null crowding by default
        self.young_av = ExtinctionDistribution()
        self.old_av = ExtinctionDistribution()
        self.rel_extinction = np.ones(self.n_bands, dtype=float)
        for av in (self.young_av, self.old_av):
            av.set_uniform(0.)
