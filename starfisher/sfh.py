#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `sfh`, the program for estimating the star formation
of a stellar population by optimizing the linear combination of eigen-CMDs.
"""

import os
from collections import OrderedDict
import subprocess

import numpy as np

from astropy.table import Table, Column

from starfisher.pathutils import starfish_dir, EnterStarFishDirectory
from starfisher.plane import Mask


class SFH(object):
    """Interface to the StarFISH ``sfh`` program.

    Parameters
    ----------
    data_root : str
        Root filename of the photometry data (the full path minus the suffix
        for each CMD plane).
    synth : :class:`synth.Synth` instance
        The instance of :class:`synth.Synth` used to prepare the synthetic
        CMDs.
    fit_dir : str
        Direcory where input files are stored for the StarFISH run.
    planes : list
        List of CMD planes, made by :class:`Synth` to use. By default all
        of the planes built by :class:`Synth` will be used.
    """
    def __init__(self, data_root, synth, fit_dir, planes=None):
        super(SFH, self).__init__()
        self.data_root = data_root
        self.synth = synth
        self.fit_dir = fit_dir
        if planes is not None:
            self._planes = planes
        else:
            self._planes = self.synth._cmds
        self.mask = Mask(self._planes)
        self._sfh_config_path = os.path.join(self.fit_dir, "sfh.dat")
        self._cmd_path = os.path.join(self.fit_dir, "cmd.txt")
        self._outfile_path = os.path.join(self.fit_dir, "output.dat")
        self._hold_path = os.path.join(self.fit_dir, "hold.dat")
        self._mask_path = os.path.join(self.fit_dir, "mask.dat")
        self._log_path = os.path.join(self.fit_dir, "sfh.log")
        self._plg_path = os.path.join(self.fit_dir, "plg.log")
        self._chi_path = os.path.join(self.fit_dir, "chi.txt")

    @property
    def outfile_path(self):
        return self._outfile_path

    @property
    def full_outfile_path(self):
        return os.path.join(starfish_dir, self.outfile_path)

    @property
    def chi_path(self):
        return self._chi_path

    @property
    def full_chi_path(self):
        return os.path.join(starfish_dir, self.chi_path)

    def run_sfh(self, hold=None):
        """Run the StarFISH `sfh` software."""
        self.synth.lockfile.write_cmdfile(self._cmd_path)
        self.synth.lockfile.write_holdfile(self._hold_path, hold=hold)
        self.mask.write(self._mask_path)
        self._write_sfh_input()
        with EnterStarFishDirectory():
            subprocess.call("./sfh < %s" % self._sfh_config_path, shell=True)

    def _write_sfh_input(self):
        """Write the SFH input file."""
        if os.path.exists(self._sfh_config_path):
            os.remove(self._sfh_config_path)

        lines = []

        # Filenames
        lines.append(self.data_root)  # datpre
        lines.append(self._cmd_path)  # cmdfile
        lines.append(self.mask.mask_path)  # maskfile
        lines.append(self._hold_path)  # hold file (needs to be created)
        lines.append(self._outfile_path)  # output
        lines.append(self._log_path)  # log
        lines.append(self._plg_path)  # plg
        lines.append(self._chi_path)  # chi

        # Synth CMD parameters
        # number of independent isochrones
        # TODO modified by the holdfile?
        lines.append(str(self.synth.n_active_groups))
        lines.append(str(len(self._planes)))

        lines.append("1")  # binning factor between synth and CMD pixels
        lines.append(str(self.synth.dpix))

        # Parameters for each CMD
        for cmd in self._planes:
            lines.append(cmd.suffix)
            lines.append("%.2f" % min(cmd.x_span))
            lines.append("%.2f" % max(cmd.x_span))
            lines.append("%.2f" % min(cmd.y_span))
            lines.append("%.2f" % max(cmd.y_span))
            nx = int((max(cmd.x_span) - min(cmd.x_span)) / self.synth.dpix)
            ny = int((max(cmd.y_span) - min(cmd.y_span)) / self.synth.dpix)
            nbox = nx * ny
            lines.append(str(nbox))

        # Runtime parameters
        # TODO enable user customization here
        lines.append("256")  # seed
        lines.append("2")  # Use Poisson fit statistic
        lines.append("0")  # don't start from a logged position
        lines.append("0")  # don't generate plg file of all tested positions
        lines.append("0")  # uniform grid
        lines.append("3")  # verbosity
        lines.append("1000.00")  # lambda; initial simplex size
        lines.append("0.68")  # error bars are at 1 sigma confidence level
        lines.append("1.000")  # threshold delta-chi**2
        lines.append("10.00")  # required parameter tolerance
        lines.append("0.0000001")  # required fit_stat tolerance
        lines.append("10000")  # number of parameter directions to search
        lines.append("3")  # number of iterations for determining errorbars

        txt = "\n".join(lines)
        with open(os.path.join(starfish_dir, self._sfh_config_path), 'w') as f:
            f.write(txt)

    def solution_table(self, avgmass=1.628,
                       marginalize_z=False, split_z=False):
        """Returns a `class`:astropy.table.Table of the derived star formation
        history.

        This is based on the ``sfh.sm`` script distributed with StarFISH.

        Parameters
        ----------
        avgmass : float
            Average mass of the stellar population; given the IMF. For a
            Salpeter IMF this is 1.628.
        marginalize_z : bool
            If ``True``, the SFH at a given time but for different
            metallicities will be coadded, resulting in a table with only
            an age dimension. This can be useful for plotting overall SFH.
        split_z : bool
            If ``True``, the return SFH will be a dictionary of tables
            corresponding to each metallicity track. Keys are logZ/Zsol strings
        """
        # read in time interval table (produced by lockfile)
        dt = self.synth.lockfile.group_dt
        print "sum of dt (Gyr)", dt.sum() / 1e9

        # TODO refactor out to its own class?
        assert marginalize_z & split_z is False

        # read sfh output
        t = Table.read(self.full_outfile_path,
                       format="ascii.no_header",
                       names=['Z', 'log(age)',
                              'amp_nstars', 'amp_nstars_n', 'amp_nstars_p'])

        # Open a photometry file to count stars
        dataset_path = os.path.join(
            starfish_dir,
            self.data_root + self._planes[0].suffix)
        _catalog = np.loadtxt(dataset_path)
        nstars = _catalog.shape[0]

        if not split_z:
            t = self._make_sfh_table(t, dt, nstars,
                                     avgmass=avgmass,
                                     marginalize_z=marginalize_z)
            return t
        else:
            tables = OrderedDict()
            z_vals = np.unique(t['Z'])
            s = np.argsort(z_vals)
            z_vals = z_vals[s]
            z_strs = ["{0:.3f}".format(np.log10(z / 0.019)) for z in z_vals]
            for z_str, z in zip(z_strs, z_vals):
                sel = np.where(t['Z'] == z)[0]
                tables[z_str] = self._make_sfh_table(t[sel], dt[sel],
                                                     nstars,
                                                     avgmass=avgmass)
            return tables

    def _make_sfh_table(self, t, dt, nstars,
                        avgmass=1.628, marginalize_z=False):
        # Renormalize to SFR (Msun/yr)
        # (Amps in the SFH file have units Nstars.)
        print len(t['amp_nstars_p'])
        print len(t['amp_nstars'])
        print len(dt)
        ep = (t['amp_nstars_p'] - t['amp_nstars']) * avgmass / dt
        en = (t['amp_nstars'] - t['amp_nstars_n']) * avgmass / dt
        sfr = t['amp_nstars'] * avgmass / dt
        mass = t['amp_nstars'] * avgmass  # solar masses produced in bin
        mass_err_neg = (t['amp_nstars'] - t['amp_nstars_n']) * avgmass
        mass_err_pos = (t['amp_nstars_p'] - t['amp_nstars']) * avgmass

        # Include Poisson errors in errorbars
        poisson_sigma = sfr / np.sqrt(nstars)
        sap = ep + poisson_sigma
        san = en + poisson_sigma
        # Truncate error bars if they extend below zero
        # so that the negative confidence region bottoms out at zero
        s = np.where((sfr - san) < 0.)[0]
        san[s] = sfr[s]

        s = np.where((mass - mass_err_neg) < 0.)[0]
        mass_err_neg[s] = mass[s]

        cmass = Column(mass, name='mass', unit='M_solar')
        cmass_neg_err = Column(mass, name='mass_neg_err', unit='M_solar')
        cmass_pos_err = Column(mass, name='mass_pos_err', unit='M_solar')
        csfr = Column(sfr, name='sfr', unit='M_solar/yr')
        csap = Column(sap, name='sfr_pos_err', unit='M_solar/yr')
        csan = Column(san, name='sfr_neg_err', unit='M_solar/yr')
        dt = Column(dt, name='dt', unit='yr')
        t.add_columns([csfr, csap, csan, cmass, cmass_pos_err, cmass_neg_err,
                       dt])

        if marginalize_z:
            t = self._marginalize_z(t)

        return t

    def _marginalize_z(self, t):
        """Marginalize SFH table across metallicities."""
        # Uniqueness/comparisons are made against rounded integer myr ages
        rounded_ages = np.empty(len(t), dtype=np.int)
        np.around(10. ** (t['log(age)'] - 6.),
                  decimals=0, out=rounded_ages)
        unique_rounded_ages = np.unique(rounded_ages)
        s = np.argsort(unique_rounded_ages)
        unique_rounded_ages = unique_rounded_ages[s]

        binned_t = Table(names=t.colnames)
        for i, age_token in enumerate(unique_rounded_ages):
            tt = t[rounded_ages == age_token]
            # error propagation
            sfr_sigma_pos = np.sqrt(np.sum(tt['sfr_pos_err'] ** 2.))
            sfr_sigma_neg = np.sqrt(np.sum(tt['sfr_neg_err'] ** 2.))
            amp_sigma_pos = np.sqrt(np.sum(tt['amp_nstars_p'] ** 2.))
            amp_sigma_neg = np.sqrt(np.sum(tt['amp_nstars'] ** 2.))
            mass_err_pos = np.sqrt(np.sum(tt['sfr_pos_err'] ** 2.))
            mass_err_neg = np.sqrt(np.sum(tt['sfr_neg_err'] ** 2.))
            binned_t.add_row((np.mean(tt['Z']),
                              tt['log(age)'][0],
                              np.sum(tt['amp_nstars']),
                              amp_sigma_neg,
                              amp_sigma_pos,
                              np.sum(tt['sfr']),
                              sfr_sigma_pos,
                              sfr_sigma_neg,
                              np.sum(tt['mass']),
                              mass_err_pos,
                              mass_err_neg,
                              np.mean(tt['dt']),
                              ))
        return binned_t

    @property
    def mean_log_age(self):
        """Mean age of a fit, in log(age)."""
        t = self.solution_table(marginalize_z=True)
        m = np.interp(50.,
                      np.cumsum(t['mass']) / t['mass'].sum() * 100.,
                      t['logage'])
        # estimate mean uncertainty from positive and negative error lim
        sigma = (t['mass_pos_err'] + t['mass_neg_err']) / 2.
        # Use resampling to estimate uncertainty of mean
        n_boot = 1000
        boot_means = np.empty(n_boot, dtype=np.float)
        n_ages = len(t)
        for i in xrange(n_boot):
            resamp = sigma * np.random.randn(n_ages) + t['mass']
            mi = np.interp(50.,
                           np.cumsum(resamp) / resamp.sum() * 100.,
                           t['log(age)'])
            boot_means[i] = mi
        sigma_mean = np.std(boot_means)
        print "mean_log_age", m, sigma_mean
        return m, sigma_mean

    @property
    def mean_age(self):
        t = self.solution_table(marginalize_z=True)
        age_gyr = 10. ** t['log(age)'] / 1e9
        m = np.interp(50.,
                      np.cumsum(t['mass']) / t['mass'].sum() * 100.,
                      age_gyr)
        sigma = (t['mass_pos_err'] + t['mass_neg_err']) / 2.
        n_boot = 1000
        boot_means = np.empty(n_boot, dtype=np.float)
        n_ages = len(t)
        for i in xrange(n_boot):
            resamp = sigma * np.random.randn(n_ages) + t['mass']
            mi = np.interp(50.,
                           np.cumsum(resamp) / resamp.sum() * 100.,
                           age_gyr)
            boot_means[i] = mi
        sigma_mean = np.std(boot_means)
        print "mean_age", m, sigma_mean
        return m, sigma_mean

    @property
    def mean_age_by_z(self):
        sfh_tables = self.solution_table(split_z=True)
        mean_ages = OrderedDict()
        mean_age_sigmas = OrderedDict()
        for z, t in sfh_tables.iteritems():
            age_gyr = 10. ** t['log(age)'] / 1e9
            m = np.interp(50.,
                          np.cumsum(t['mass']) / t['mass'].sum() * 100.,
                          age_gyr)
            sigma = (t['mass_pos_err'] + t['mass_neg_err']) / 2.
            n_boot = 1000
            boot_means = np.empty(n_boot, dtype=np.float)
            n_ages = len(t)
            for i in xrange(n_boot):
                resamp = sigma * np.random.randn(n_ages) + t['mass']
                mi = np.interp(50.,
                               np.cumsum(resamp) / resamp.sum() * 100.,
                               age_gyr)
                boot_means[i] = mi
            sigma_mean = np.std(boot_means)
            mean_ages[z] = m
            mean_age_sigmas[z] = sigma_mean
        return mean_ages, mean_age_sigmas

    def plane_index(self, plane):
        """Index of a color plane in the SFH system.

        Parameters
        ----------
        plane : :class:`starfisher.plane.ColorPlane`
            The `ColorPlane` instance to get the SFH index of.

        Returns
        -------
        index : int
            Index of the color Plane.
        """
        return self._planes.index(plane) + 1

    def read_chi(self, plane):
        """Chi-sq Hess diagram for the given plane.

        Parameters
        ----------
        plane : :class:`starfisher.plane.ColorPlane`
            The `ColorPlane` instance to get the chi-sq Hess diagram of.
        """
        data = plane.read_chi(self.full_chi_path, self.plane_index(plane))
        return data
