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

import palettable

from .sfh import marginalize_sfh_metallicity


def plot_sfh_line(ax, sfh_table,
                  z_formatter=mpl.ticker.FormatStrFormatter("%.2f"),
                  age_formatter=mpl.ticker.FormatStrFormatter("%4.1f"),
                  age_lim=(1e-3, 14.),
                  amp_key='sfr',
                  log_amp=True,
                  log_age=True,
                  legend=True,
                  x_label=True,
                  y_label=True,
                  z_colors=palettable.tableau.ColorBlind_10.mpl_colors):
    amp = sfh_table[amp_key]
    if log_amp:
        amp = np.log10(amp)

    if not log_age:
        age = 10. ** (sfh_table['log(age)'] - 9.)  # Gyr
    else:
        age = sfh_table['log(age)']

    ZZsol = np.log10(sfh_table['Z'] / 0.019)
    z_vals = np.unique(ZZsol)
    srt = np.argsort(z_vals)

    # print "mpl colors", colors
    ax.set_color_cycle(z_colors)
    for i, z in enumerate(z_vals[srt]):
        s = np.where(ZZsol == z)[0]
        ax.plot(age[s], amp[s], ls='-',
                label=r"$\log Z/Z_\odot={0:.3f}$".format(np.round(z,
                                                                  decimals=3)))

    if log_age:
        ax.set_xlabel(r"$\log(A~\mathrm{yr}^{-1})$")
    else:
        ax.set_xlabel(r"$A$ (Gyr)")

    if amp_key == 'sfr' and not log_amp:
        ax.set_ylabel(r"$\mathrm{M}_\odot \mathrm{yr}^{-1}$")
    elif amp_key == 'sfr' and log_amp:
        ax.set_ylabel(r"$\log M_\odot \mathrm{yr}^{-1}$")
    elif amp_key == 'amp_nstars' and not log_amp:
        ax.set_ylabel(r"$N_\star$")
    elif amp_key == 'amp_nstars' and log_amp:
        ax.set_ylabel(r"$\log N_\star$")

    ax.xaxis.set_major_formatter(age_formatter)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.5))

    if log_age:
        ax.set_xlim(np.log10(age_lim[0] * 1e9), np.log10(age_lim[-1] * 1e9))
    else:
        ax.set_xlim(*age_lim)

    if not x_label:
        for tl in ax.get_xmajorticklabels():
            tl.set_visible(False)
        ax.set_xlabel('')
    if not y_label:
        for tl in ax.get_ymajorticklabels():
            tl.set_visible(False)
        ax.set_ylabel('')

    if legend:
        ax.legend(frameon=False)


def plot_single_sfh_line(
        ax, sfh_table,
        z_formatter=mpl.ticker.FormatStrFormatter("%.2f"),
        age_formatter=mpl.ticker.FormatStrFormatter("%4.1f"),
        age_lim=(1e-3, 14.),
        amp_key='sfr',
        log_amp=True,
        log_age=True,
        x_label=True,
        y_label=True,
        color='dodgerblue',
        label=None,
        plot_errors=False,
        hatch_errors=None,
        drawstyle='default',
        log_error_floor=-10):
    t = marginalize_sfh_metallicity(sfh_table)

    amp = t[amp_key]
    if log_amp:
        amp = np.log10(amp)

    if not log_age:
        age = 10. ** (t['log(age)'] - 9.)  # Gyr
    else:
        age = t['log(age)']

    s = np.argsort(age)

    ax.plot(age[s], amp[s], ls='-', lw=2.5,
            c=color, drawstyle=drawstyle, label=label)

    if plot_errors and amp_key == 'sfr':
        pos_err = t['sfr_pos_err'][s]
        neg_err = t['sfr_neg_err'][s]
        if log_amp:
            pos_ci = np.log10(t['sfr'][s] + pos_err)
            neg_ci = np.log10(t['sfr'][s] - neg_err)
            neg_ci[~np.isfinite(neg_ci)] = log_error_floor  # HACK for -inf

        error_args = {'zorder': -10}
        hatch_error_args = {'facecolor': 'None', 'edgecolor': color,
                            'lw': 0., 'hatch': hatch_errors}
        solid_args = {'facecolor': color,
                      'alpha': 0.2,
                      'edgecolor': 'None'}
        if hatch_errors is not None:
            error_args.update(hatch_error_args)
        else:
            error_args.update(solid_args)
        ax.fill_between(age[s], pos_ci, y2=neg_ci, **error_args)

    if log_age:
        ax.set_xlabel(r"$\log(A~\mathrm{yr}^{-1})$")
    else:
        ax.set_xlabel(r"$A$ (Gyr)")

    if amp_key == 'sfr' and not log_amp:
        ax.set_ylabel(r"$\mathrm{M}_\odot \mathrm{yr}^{-1}$")
    elif amp_key == 'sfr' and log_amp:
        ax.set_ylabel(r"$\log M_\odot \mathrm{yr}^{-1}$")
    elif amp_key == 'amp_nstars' and not log_amp:
        ax.set_ylabel(r"$N_\star$")
    elif amp_key == 'amp_nstars' and log_amp:
        ax.set_ylabel(r"$\log N_\star$")

    ax.xaxis.set_major_formatter(age_formatter)
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.5))

    if log_age:
        ax.set_xlim(np.log10(age_lim[0] * 1e9), np.log10(age_lim[-1] * 1e9))
    else:
        ax.set_xlim(*age_lim)

    if not x_label:
        for tl in ax.get_xmajorticklabels():
            tl.set_visible(False)
        ax.set_xlabel('')
    if not y_label:
        for tl in ax.get_ymajorticklabels():
            tl.set_visible(False)
        ax.set_ylabel('')


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

    def plot_in_ax(self, ax, max_area=200., amp_key='sfr'):
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
        amp = self._table[amp_key]
        scaled_area = amp / amp.max() * max_area
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

    def plot_in_ax(self, ax, max_area=200., amp_key='sfr'):
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
        amp = self._table[amp_key]
        scaled_area = amp / amp.max() * max_area
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
    shf : :class:`starfisher.sfh.SFH`
        The :class:`SFH` instance with results.
    plane : :class:`starfisher.plane.ColorPlane`
        The :class:`ColorPlane` fitted by `sfh` to be plotted.
    """
    def __init__(self, sfh, plane):
        super(ChiTriptykPlot, self).__init__()
        _ = sfh.read_chi(plane)
        self.mod_hess = _[0]
        self.obs_hess = _[1]
        self.chi_hess = _[2]
        self.plane = plane

    def plot_mod_in_ax(self, ax, **args):
        """Plot the model Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.plane.extent,
                 origin=self.plane.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        ax.imshow(np.log10(self.mod_hess), **a)
        return ax

    def plot_obs_in_ax(self, ax, **args):
        """Plot the observed Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.plane.extent,
                 origin=self.plane.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        ax.imshow(np.log10(self.obs_hess), **a)
        return ax

    def plot_chi_in_ax(self, ax, **args):
        """Plot the chi Hess diagram in the axis."""
        a = dict(cmap=mpl.cm.gray_r, extent=self.plane.extent,
                 origin=self.plane.origin, aspect='auto', interpolation='none')
        if args is not None:
            a.update(args)
        ax.imshow(np.log10(self.chi_hess), **a)
        return ax

    def setup_axes(self, fig, ax_obs=None, ax_mod=None, ax_chi=None,
                   major_y=1., major_x=0.5, major_x_fmt="%.1f"):
        if (ax_obs is None) or (ax_mod is None) or (ax_chi is None):
            gs = gridspec.GridSpec(1, 3, left=0.1, right=0.95,
                                   bottom=0.15, top=0.95,
                                   wspace=None, hspace=None,
                                   width_ratios=None, height_ratios=None)
            ax_obs = fig.add_subplot(gs[0])
            ax_mod = fig.add_subplot(gs[1])
            ax_chi = fig.add_subplot(gs[2])

        for ax in [ax_obs, ax_mod, ax_chi]:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_y))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(
                major_x_fmt))
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(major_x))
        ax_mod.set_xlabel(self.plane.x_label)
        ax_obs.set_ylabel(self.plane.y_label)
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
