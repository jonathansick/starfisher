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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from palettable.cubehelix import perceptual_rainbow_16
from palettable.colorbrewer.diverging import RdBu_11

from starfisher import LibraryBuilder
from starfisher import SimHess
from starfisher import Synth
from starfisher.plane import Hess
from starfisher.plane import StarCatalogHess
from starfisher import SFH
from starfisher import ExtinctionDistribution
from starfisher import MockNullCrowdingTable
from starfisher.plots import plot_hess, setup_hess_axes
from starfisher.plots import plot_lock_polygons
from starfisher.plots import plot_isochrone_logage_logzsol
from starfisher.sfhplot import plot_sfh_line
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
        self.n_synth_cpu = kwargs.pop('n_synth_cpu', 1)

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
        self.mask_planes()  # mask planes based on completeness cuts
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
            os.path.join(full_synth_dir, "z*"))) > 0
        if not existing_synth:
            self.synth.run_synth(n_cpu=self.n_synth_cpu, clean=False)

    @abc.abstractmethod
    def mask_planes(self):
        pass

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

    def make_fit_diff_hess(self, dataset, fit_key, plane_key):
        obs_hess = self.make_obs_hess(dataset, plane_key)
        fit_hess = self.make_fit_hess(fit_key, plane_key)
        return Hess(obs_hess.hess - fit_hess.hess,
                    self.planes[plane_key])

    def make_chisq_hess(self, dataset, fit_key, plane_key):
        obs_hess = self.make_obs_hess(dataset, plane_key)
        fit_hess = self.make_fit_hess(fit_key, plane_key)
        sigma = np.sqrt(obs_hess.hess)
        chi = ((obs_hess.hess - fit_hess.hess) / sigma) ** 2.
        return Hess(chi, self.planes[plane_key])

    def compute_fit_chi(self, dataset, fit_key, plane_key, chi_hess=None):
        """Compute the reduced chi-sq for the plane with the given fit.

        Returns both the sum of chi-sq and the total number of pixels in
        in the plane that were not masked.
        """
        if chi_hess is None:
            chi_hess = self.make_chisq_hess(dataset, fit_key, plane_key)
        g = np.where(np.isfinite(chi_hess.masked_hess))
        n_pix = len(g[0])
        chi_sum = chi_hess.masked_hess[g].sum()
        n_amp = len(self.lockfile.active_groups)
        return chi_sum / (n_pix - n_amp)

    def make_sim_hess(self, plane_key):
        return self.get_sim_hess(plane_key)

    def make_fit_hess(self, fit_key, plane_key):
        plane = self.planes[plane_key]
        return SimHess.from_sfh_solution(self.fits[fit_key], plane)

    def make_obs_hess(self, dataset, plane_key):
        plane = self.planes[plane_key]
        x = dataset.get_phot(plane.x_mag)
        y = dataset.get_phot(plane.y_mag)
        return StarCatalogHess(x, y, plane)

    def init_plane_axes(self, ax, plane_key):
        plane = self.planes[plane_key]
        setup_hess_axes(ax, plane, 'lower')

    def plot_sim_hess(self, ax, plane_key, imshow=None):
        plane = self.planes[plane_key]
        sim = self.get_sim_hess(plane_key)
        return plot_hess(ax, sim.hess, plane, sim.origin,
                         imshow_args=imshow)

    def plot_fit_hess(self, ax, fit_key, plane_key, imshow=None):
        plane = self.planes[plane_key]
        fit_hess = SimHess.from_sfh_solution(self.fits[fit_key], plane)
        return plot_hess(ax, fit_hess.hess, plane, fit_hess.origin,
                         imshow_args=imshow)

    def plot_obs_hess(self, ax, dataset, plane_key, imshow=None):
        plane = self.planes[plane_key]
        x = dataset.get_phot(plane.x_mag)
        y = dataset.get_phot(plane.y_mag)
        obs_hess = StarCatalogHess(x, y, plane)
        return plot_hess(ax, obs_hess.hess, plane, obs_hess.origin,
                         imshow_args=imshow)

    def plot_hess_array(self, ax, hess, plane_key, imshow=None, log=True):
        plane = self.planes[plane_key]
        return plot_hess(ax, hess, plane, plane.origin,
                         imshow_args=imshow, log=log)

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
        fit = self.fits[fit_key]
        plane = self.planes[plane_key]
        ctp = ChiTriptykPlot(fit, plane)
        ctp.setup_axes(fig, ax_obs=ax_obs, ax_mod=ax_model, ax_chi=ax_chi,
                       major_x=xtick, major_x_fmt=xfmt)
        ctp.plot_obs_in_ax(ax_obs, cmap=perceptual_rainbow_16.mpl_colormap)
        ctp.plot_mod_in_ax(ax_model, cmap=perceptual_rainbow_16.mpl_colormap)
        ctp.plot_chi_in_ax(ax_chi, cmap=perceptual_rainbow_16.mpl_colormap)
        ax_obs.text(0.0, 1.01, "Observed",
                    transform=ax_obs.transAxes, size=8, ha='left')
        ax_model.text(0.0, 1.01, "Model",
                      transform=ax_model.transAxes, size=8, ha='left')
        ax_chi.text(0.0, 1.01, r"$\log \chi^2$",
                    transform=ax_chi.transAxes, size=8, ha='left')

    def plot_linear_sfh_circles(self, ax, fit_key, ylim=(-0.2, 0.2),
                                amp_key='sfr'):
        sfh = self.fits[fit_key]
        cp = LinearSFHCirclePlot(sfh.solution_table())
        cp.plot_in_ax(ax, max_area=800, amp_key=amp_key)
        for tl in ax.get_ymajorticklabels():
            tl.set_visible(False)
        ax.set_ylim(*ylim)

    def plot_log_sfh_circles(self, ax, fit_key, ylim=(-0.2, 0.2),
                             amp_key='sfr'):
        sfh = self.fits[fit_key]
        cp = SFHCirclePlot(sfh.solution_table())
        cp.plot_in_ax(ax, max_area=800, amp_key=amp_key)
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


def show_fit(pipeline, dataset, fit_key, plane_key):
    cube_map = perceptual_rainbow_16.mpl_colormap

    obs_hess = pipeline.make_obs_hess(dataset, plane_key)
    fit_hess = pipeline.make_fit_hess(fit_key, plane_key)
    sigma = np.sqrt(obs_hess.hess)
    chi = ((obs_hess.hess - fit_hess.hess) / sigma) ** 2.
    diff = obs_hess.hess - fit_hess.hess

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 4, wspace=0.4, bottom=0.2, right=0.95,
                  width_ratios=(1, 1, 1, 1), height_ratios=(0.1, 1))
    ax_obs = fig.add_subplot(gs[1, 0])
    ax_model = fig.add_subplot(gs[1, 1])
    ax_chi = fig.add_subplot(gs[1, 2])
    ax_diff = fig.add_subplot(gs[1, 3])
    ax_obs_cb = fig.add_subplot(gs[0, 0])
    ax_model_cb = fig.add_subplot(gs[0, 1])
    ax_chi_cb = fig.add_subplot(gs[0, 2])
    ax_diff_cb = fig.add_subplot(gs[0, 3])

    fit_map = pipeline.plot_fit_hess(ax_model, fit_key, plane_key,
                                     imshow=dict(vmin=0, vmax=3.,
                                                 cmap=cube_map))
    fit_cb = plt.colorbar(fit_map, cax=ax_model_cb, orientation='horizontal')
    fit_cb.set_label(r"$\log(N_*)$ Model")
    fit_cb.ax.xaxis.set_ticks_position('top')
    fit_cb.locator = mpl.ticker.MultipleLocator(0.5)
    fit_cb.update_ticks()

    obs_map = pipeline.plot_obs_hess(ax_obs, dataset, plane_key,
                                     imshow=dict(vmin=0, vmax=3.,
                                                 cmap=cube_map))
    obs_cb = plt.colorbar(obs_map, cax=ax_obs_cb, orientation='horizontal')
    obs_cb.set_label(r"$\log(N_*)$ Obs.")
    obs_cb.ax.xaxis.set_ticks_position('top')
    obs_cb.locator = mpl.ticker.MultipleLocator(0.5)
    obs_cb.update_ticks()

    chi_map = pipeline.plot_hess_array(ax_chi, chi, plane_key, log=False,
                                       imshow=dict(vmax=20, cmap=cube_map))
    chi_cb = plt.colorbar(chi_map, cax=ax_chi_cb, orientation='horizontal',)
    chi_cb.set_label(r"$\chi^2$")
    chi_cb.ax.xaxis.set_ticks_position('top')
    chi_cb.locator = mpl.ticker.MultipleLocator(5)
    chi_cb.update_ticks()

    diff_map = pipeline.plot_hess_array(ax_diff, diff, plane_key, log=False,
                                        imshow=dict(vmin=-50, vmax=50,
                                                    cmap=RdBu_11.mpl_colormap))
    diff_cb = plt.colorbar(diff_map, cax=ax_diff_cb, orientation='horizontal')
    diff_cb.set_label(r"$\Delta_\mathrm{obs-model}$ ($N_*$)")
    diff_cb.ax.xaxis.set_ticks_position('top')
    diff_cb.locator = mpl.ticker.MultipleLocator(20)
    diff_cb.update_ticks()

    fig.show()


def show_sfh(pipeline, fit_key, ylim=(-0.2, 0.2)):
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 2, wspace=0.1, hspace=0.2, bottom=0.2, right=0.95)
    ax_sfr_log = fig.add_subplot(gs[0, 0])
    ax_sfr_lin = fig.add_subplot(gs[0, 1])
    ax_n_log = fig.add_subplot(gs[1, 0])
    ax_n_lin = fig.add_subplot(gs[1, 1])

    ax_sfr_log.text(0.1, 0.9, "SFR(t)",
                    transform=ax_sfr_log.transAxes, va='top')
    ax_sfr_lin.text(0.1, 0.9, "SFR(t)",
                    transform=ax_sfr_lin.transAxes, va='top')
    ax_n_log.text(0.1, 0.9, r"$N_\star$",
                  transform=ax_n_log.transAxes, va='top')
    ax_n_lin.text(0.1, 0.9, r"$N_\star$",
                  transform=ax_n_lin.transAxes, va='top')

    pipeline.plot_log_sfh_circles(ax_sfr_log, fit_key, ylim=ylim)
    pipeline.plot_linear_sfh_circles(ax_sfr_lin, fit_key, ylim=ylim)
    pipeline.plot_log_sfh_circles(ax_n_log, fit_key, ylim=ylim,
                                  amp_key='amp_nstars')
    pipeline.plot_linear_sfh_circles(ax_n_lin, fit_key, ylim=ylim,
                                     amp_key='amp_nstars')
    ax_sfr_lin.set_xlabel('')
    ax_sfr_lin.set_ylabel('')
    ax_sfr_log.set_xlabel('')
    ax_n_lin.set_ylabel('')
    fig.show()


def show_sfh_line(pipeline, fit_key):
    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 2, wspace=0.1, hspace=0.1, bottom=0.2, right=0.95)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    tbl = pipeline.fits[fit_key].solution_table()

    plot_sfh_line(ax1, tbl, amp_key='sfr',
                  log_age=True, legend=False, x_label=False)
    plot_sfh_line(ax2, tbl, amp_key='sfr',
                  log_age=False, legend=True, x_label=False, y_label=False)
    plot_sfh_line(ax3, tbl, amp_key='amp_nstars',
                  log_age=True, legend=False)
    plot_sfh_line(ax4, tbl, amp_key='amp_nstars',
                  log_age=False, legend=False, y_label=False)

    for logage in np.log10(np.arange(1, 14, 1) * 1e9):
        ax1.axvline(logage, c='0.8', zorder=-1)
        ax3.axvline(logage, c='0.8', zorder=-1)

    fig.show()
