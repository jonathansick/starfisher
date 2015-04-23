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

# import palettable
import cubehelix

from starfisher import LibraryBuilder
from starfisher import SimHess
from starfisher import Synth
from starfisher.plane import StarCatalogHess
from starfisher import SFH
from starfisher import ExtinctionDistribution
from starfisher import MockNullCrowdingTable
from starfisher.plots import plot_hess
from starfisher.plots import plot_lock_polygons
from starfisher.plots import plot_isochrone_logage_logzsol
from starfisher.sfhplot import ChiTriptykPlot
from starfisher.sfhplot import LinearSFHCirclePlot, SFHCirclePlot

STARFISH = os.getenv("STARFISH")


class DatasetBase(object):
    """Abstract baseclass for pipeline components that write observational
    data.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "DatasetBase", kwargs
        super(DatasetBase, self).__init__(**kwargs)

    @abc.abstractmethod
    def get_phot(self, band):
        pass

    @abc.abstractmethod
    def write_phot(self, x_band, y_band, data_root, suffix):
        pass


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
                           self.bands,
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

    @property
    def hold_template(self):
        return self.lockfile.empty_hold

    def fit(self, fit_key, plane_keys, dataset, redo=False, hold=None):
        fit_dir = os.path.join(self.root_dir, fit_key)
        data_root = os.path.join(fit_dir, "phot.")
        planes = []
        for plane_key in plane_keys:
            plane = self.planes[plane_key]
            planes.append(plane)
            dataset.write_phot(plane.x_mag, plane.y_mag,
                               data_root, plane.suffix)
        sfh = SFH(data_root, self.synth, fit_dir, planes=planes)
        if (not os.path.exists(sfh.full_outfile_path)) or redo:
            sfh.run_sfh(hold=hold)
        self.fits[fit_key] = sfh

    def plot_sim_hess(self, ax, plane_key):
        plane = self.planes[plane_key]
        sim = self.get_sim_hess(plane_key)
        plot_hess(ax, sim.hess, plane, sim.origin,
                  imshow_args=None)

    def plot_fit_hess(self, ax, fit_key, plane_key):
        plane = self.planes[plane_key]
        fit_hess = SimHess.from_sfh_solution(self.fits[fit_key], plane)
        plot_hess(ax, fit_hess.hess, plane, fit_hess.origin,
                  imshow_args=None)

    def plot_obs_hess(self, ax, dataset, plane_key):
        plane = self.planes[plane_key]
        x = dataset.get_phot(plane.x_mag)
        y = dataset.get_phot(plane.y_mag)
        obs_hess = StarCatalogHess(x, y, plane)
        plot_hess(ax, obs_hess.hess, plane, obs_hess.origin,
                  imshow_args=None)

    def plot_lockfile(self, ax,
                      logage_lim=(6.2, 10.2),
                      logzzsol_lim=(-0.2, 0.2)):
        plot_isochrone_logage_logzsol(ax, self.builder, c='k', s=8)
        plot_lock_polygons(ax, self.lockfile, facecolor='None', edgecolor='r')
        ax.set_xlim(*logage_lim)
        ax.set_ylim(*logzzsol_lim)
        ax.set_xlabel(r"$\log(A)$")
        ax.set_ylabel(r"$\log(Z/Z_\odot)$")

    def plot_triptyk(self, fig, ax_obs, ax_model, ax_chi, fit_key, plane_key,
                     xtick=1., xfmt="%.0f"):
        cmapper = lambda: cubehelix.cmap(startHue=240, endHue=-300, minSat=1,
                                         maxSat=2.5, minLight=.3,
                                         maxLight=.8, gamma=.9)
        fit = self.fits[fit_key]
        plane = self.planes[plane_key]
        ctp = ChiTriptykPlot(fit, plane)
        ctp.setup_axes(fig, ax_obs=ax_obs, ax_mod=ax_model, ax_chi=ax_chi,
                       major_x=xtick, major_x_fmt=xfmt)
        ctp.plot_obs_in_ax(ax_obs, cmap=cmapper())
        ctp.plot_mod_in_ax(ax_model, cmap=cmapper())
        ctp.plot_chi_in_ax(ax_chi, cmap=cubehelix.cmap())
        ax_obs.text(0.0, 1.01, "Observed",
                    transform=ax_obs.transAxes, size=8, ha='left')
        ax_model.text(0.0, 1.01, "Model",
                      transform=ax_model.transAxes, size=8, ha='left')
        ax_chi.text(0.0, 1.01, r"$\log \chi^2$",
                    transform=ax_chi.transAxes, size=8, ha='left')

    def plot_linear_sfh_circles(self, ax, fit_key, ylim=(-0.2, 0.2)):
        sfh = self.fits[fit_key]
        cp = LinearSFHCirclePlot(sfh.solution_table())
        cp.plot_in_ax(ax, max_area=800)
        for tl in ax.get_ymajorticklabels():
            tl.set_visible(False)
        ax.set_ylim(*ylim)

    def plot_log_sfh_circles(self, ax, fit_key, ylim=(-0.2, 0.2)):
        sfh = self.fits[fit_key]
        cp = SFHCirclePlot(sfh.solution_table())
        cp.plot_in_ax(ax, max_area=800)
        for logage in np.log10(np.arange(1, 13, 1) * 1e9):
            ax.axvline(logage, c='0.8', zorder=-1)
        ax.set_ylim(*ylim)


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
