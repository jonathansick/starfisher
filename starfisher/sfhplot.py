#!/usr/bin/env python
# encoding: utf-8
"""
Tools for plotting star formation histories.
"""

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from starfisher.hess import read_chi


class LinearSFHCirclePlot(object):
    """Plot SFH amplitudes as circular areas in a linear age vs metallicity

    Parameters
    ----------
    sfh_table : :class:`astropy.table.Table` instance
        The SFH solution table generated via :meth:`sfh.SFH.solution_table`.
        This is an Astropy :class:`Table` instance.
    """
    def __init__(self, sfh_table):
        super(LinearSFHCirclePlot, self).__init__()
        self._table = sfh_table

    @staticmethod
    def z_tick_formatter():
        """Formatter for metallicity axis."""
        return mpl.ticker.FormatStrFormatter("%.2f")

    @staticmethod
    def age_tick_formatter():
        """Formatter for log(age) axis."""
        return mpl.ticker.FormatStrFormatter("%4.1f")

    def plot_in_ax(self, ax, max_area=200.):
        """Plot the SFH in the given axes.

        Parameters
        ----------
        ax : matplotlib `Axes` instance
            The axes to plot into.
        max_area : float
            Area in square-points of the largest circle, corresponding
            to the highest star formation rate component of the CMD.
            Tweak this value to make a plot that neither oversaturates,
            nor produces points too small to see.
        """
        scaled_area = self._table['sfr'] / self._table['sfr'].max() * max_area
        ZZsol = np.log10(self._table['Z'] / 0.019)
        ax.scatter(10. ** (self._table['log(age)'] - 9.), ZZsol,
                   s=scaled_area,
                   c='k', marker='o', linewidths=0.)
        ax.set_xlabel(r"$A$ (Gyr)")
        ax.set_ylabel(r"$\log(Z/Z_\odot)$")
        ax.xaxis.set_major_formatter(self.age_tick_formatter())
        ax.yaxis.set_major_formatter(self.z_tick_formatter())
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.5))
        ax.set_ylim(-2.0, 0.5)
        ax.set_xlim(0., 14.)
        return ax


class SFHCirclePlot(object):
    """Plot SFH amplitudes as circular areas in an age vs metallicity plot.

    Parameters
    ----------
    sfh_table : :class:`astropy.table.Table` instance
        The SFH solution table generated via :meth:`sfh.SFH.solution_table`.
        This is an Astropy :class:`Table` instance.
    """
    def __init__(self, sfh_table):
        super(SFHCirclePlot, self).__init__()
        self._table = sfh_table

    @staticmethod
    def z_tick_formatter():
        """Formatter for metallicity axis."""
        return mpl.ticker.FormatStrFormatter("%.2f")

    @staticmethod
    def age_tick_formatter():
        """Formatter for log(age) axis."""
        return mpl.ticker.FormatStrFormatter("%4.1f")

    def plot_in_ax(self, ax, max_area=200.):
        """Plot the SFH in the given axes.

        Parameters
        ----------
        ax : matplotlib `Axes` instance
            The axes to plot into.
        max_area : float
            Area in square-points of the largest circle, corresponding
            to the highest star formation rate component of the CMD.
            Tweak this value to make a plot that neither oversaturates,
            nor produces points too small to see.
        """
        scaled_area = self._table['sfr'] / self._table['sfr'].max() * max_area
        ZZsol = np.log10(self._table['Z'] / 0.019)
        ax.scatter(self._table['log(age)'], ZZsol, s=scaled_area,
                   c='k', marker='o', linewidths=0.)
        ax.set_xlabel(r"$\log(A)$")
        ax.set_ylabel(r"$\log(Z/Z_\odot)$")
        ax.xaxis.set_major_formatter(self.age_tick_formatter())
        ax.yaxis.set_major_formatter(self.z_tick_formatter())
        ax.set_ylim(-2.0, 0.5)
        ax.set_xlim(6.45, 10.55)
        return ax

    def plot(self, path, figsize=(4.5, 3.5), format='pdf', plotargs={}):
        """Construct and save SFH plot to ``path``.

        Parameters
        ----------
        path : str
            Path where the plot will be saved. Do not include the format
            suffix.
        figsize : tuple
            The ``(width, height)`` size of the figure, in inches.
        format : str
            Suffix of the format. E.g. ``pdf`` or ``png``.
        plotargs : dict
            Dictionary of keyword arguments passed to :meth:`plot_in_ax`.
        """
        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        gs = gridspec.GridSpec(1, 1, left=0.18, right=0.95, bottom=0.15,
                               top=0.95, wspace=None, hspace=None,
                               width_ratios=None, height_ratios=None)
        ax = fig.add_subplot(gs[0])
        self.plot_in_ax(ax, **plotargs)
        gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
        canvas.print_figure(path + "." + format, format=format)


class ChiTriptykPlot(object):
    """Three-panel plots of model Hess diagram, observed Hess diagram
    and chi-square Hess diagram.

    Parameters
    ----------
    chipath : str
        Path to the 'chi' output file from SFH.
    icmd : int
        Index of the CMD. The first CMD has an index of ``1``.
    xspan : tuple
        Tuple of ``(x_min, x_max)`` extent of the x-axis, in units of
        magnitudes.
    yspan : tuple
        Tuple of ``(y_min, y_max)`` extent of the y-axis, in units of
        magnitudes.
    dpix : float
        Size of each CMD pixel (``dpix`` by ``dpix`` in area) in mags.
    flipx : bool
        Reverse orientation of x-axis if ``True``.
    flipy : bool
        Reverse orientation of y-axis if ``True`` (e.g., for CMDs).
    """
    def __init__(self, chipath, icmd, xspan, yspan, dpix, xlabel, ylabel,
                 flipx=False, flipy=False):
        super(ChiTriptykPlot, self).__init__()
        _ = read_chi(chipath, icmd, xspan, yspan, dpix,
                     flipx=flipx, flipy=flipy)
        self.mod_hess = _[0]
        self.obs_hess = _[1]
        self.chi_hess = _[2]
        self.extent = _[3]
        self.origin = _[4]
        self.chipath = chipath
        self.icmd = icmd
        self.xspan = xspan
        self.yspan = yspan
        self.dpix = dpix
        self.flipx = flipx
        self.flipy = flipy
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot_mod_in_ax(self, ax, **args):
        """Plot the model Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.extent,
                 origin=self.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        print a
        ax.imshow(np.log10(self.mod_hess), **a)
        return ax

    def plot_obs_in_ax(self, ax, **args):
        """Plot the observed Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.extent,
                 origin=self.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        print a
        ax.imshow(np.log10(self.obs_hess), **a)
        return ax

    def plot_chi_in_ax(self, ax, **args):
        """Plot the chi Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.extent,
                 origin=self.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        print a
        ax.imshow(np.log10(self.chi_hess), **a)
        return ax

    def setup_axes(self, fig):
        gs = gridspec.GridSpec(1, 3, left=0.1, right=0.95,
                               bottom=0.15, top=0.95,
                               wspace=None, hspace=None,
                               width_ratios=None, height_ratios=None)
        ax_obs = fig.add_subplot(gs[0])
        ax_mod = fig.add_subplot(gs[1])
        ax_chi = fig.add_subplot(gs[2])
        for ax in [ax_obs, ax_mod, ax_chi]:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1.))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax_mod.set_xlabel(self.xlabel)
        ax_obs.set_ylabel(self.ylabel)
        for tl in ax_mod.get_ymajorticklabels():
            tl.set_visible(False)
        for tl in ax_chi.get_ymajorticklabels():
            tl.set_visible(False)
        return ax_obs, ax_mod, ax_chi

    def plot_triptyke(self, plotpath, format="pdf"):
        """Make a plot triptype and save to disk."""
        fig = Figure(figsize=(6.5, 3.5))
        canvas = FigureCanvas(fig)
        ax_obs, ax_mod, ax_chi = self.setup_axes(fig)
        self.plot_obs_in_ax(ax_obs)
        self.plot_mod_in_ax(ax_mod)
        self.plot_chi_in_ax(ax_chi)
        # gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
        canvas.print_figure(plotpath + "." + format, format=format, dpi=300)
