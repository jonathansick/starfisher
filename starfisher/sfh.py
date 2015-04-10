#!/usr/bin/env python
# encoding: utf-8
"""
This module handles `sfh`, the program for estimating the star formation
of a stellar population by optimizing the linear combination of eigen-CMDs.
"""

import os
import subprocess

import numpy as np

from astropy.table import Table, Column

from starfisher.pathutils import starfish_dir, EnterStarFishDirectory
from starfisher.hess import read_chi
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

    def run_sfh(self):
        """Run the StarFISH `sfh` software."""
        self.synth.lockfile.write_cmdfile(self._cmd_path)
        self.synth.lockfile.write_holdfile(self._hold_path)
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

    def solution_table(self, avgmass=1.628):
        """Returns a `class`:astropy.table.Table of the derived star formation
        history.

        This is based on the ``sfh.sm`` script distributed with StarFISH.

        Parameters
        ----------
        avgmass : float
            Average mass of the stellar population; given the IMF. For a
            Salpeter IMF this is 1.628.
        """
        # TODO refactor out to its own class?
        # read in time interval table (produced by lockfile)
        dt = self.synth.lockfile.group_dt

        # read sfh output
        t = Table.read(self.full_outfile_path,
                       format="ascii.no_header",
                       names=['Z', 'log(age)',
                              'amp_nstars', 'amp_nstars_n', 'amp_nstars_p'])

        # Open a photometry file to count stars
        dataset_path = os.path.join(
            starfish_dir,
            self.data_root + self.synth._cmds[0].suffix)
        _catalog = np.loadtxt(dataset_path)
        nstars = _catalog.shape[0]

        # Renormalize to SFR (Msun/yr)
        # (Amps in the SFH file have units Nstars.)
        ep = (t['amp_nstars_p'] - t['amp_nstars']) * avgmass / dt
        en = (t['amp_nstars'] - t['amp_nstars_n']) * avgmass / dt
        sfr = t['amp_nstars'] * avgmass / dt

        # Include Poisson errors in errorbars
        snstars = np.sqrt(float(nstars))
        _foo = t['amp_nstars'] * np.sqrt((snstars / nstars) ** 2.)
        sap = ep + _foo
        san = en + _foo
        # Truncate error bars if they extend below zero
        san[san < 0.] = 0.

        csfr = Column(sfr, name='sfr', unit='M_solar/yr')
        csap = Column(sap, name='sfr_pos_err', unit='M_solar/yr')
        csan = Column(san, name='sfr_neg_err', unit='M_solar/yr')
        t.add_columns([csfr, csap, csan])

        return t

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

    def chi_hess(self, plane):
        """Chi-sq Hess diagram for the given plane.

        Parameters
        ----------
        plane : :class:`starfisher.plane.ColorPlane`
            The `ColorPlane` instance to get the chi-sq Hess diagram of.
        """
        data = read_chi(self.full_chi_path, self.plane_index(plane),
                        plane.x_span, plane.y_span, plane.dpix,
                        flipx=False, flipy=True)
        return data
