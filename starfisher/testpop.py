# encoding: utf-8
"""
Create artifical stellar catalogs with ``testpop``.
"""

import os
import subprocess

from .pathutils import starfish_dir, EnterStarFishDirectory


class TestPop(object):
    """Interface to the StarFISH ``testpop`` program.

    Note that testpop should not be used by multiple processes; it depends on
    a single testpop/synth.dat file that cannot be changed from one process
    to another.

    Parameters
    ----------
    name : str
        A prefix for the output photometry files (up to 8 characters).
    synth : :class:`starfisher.synth.Synth`
        A synth instance that defines the CMD planes.
    sfh_amps : :class:`numpy.ndarray`
        An array of star formation amplitudes that corresponds to isochrones
        or isochrone groups in the lockfile.
    use_lockfile : bool
        Use locked amplitudes according a lockfile, if True.
    n_stars : int
        Number of stars per unit amplitude.
    dmod : float
        Delta-distance modulus.
    fext : float
        Addtional extinction multiplier.
    gamma : float
        IMF slope. Default is Salpeter; -1.35.
    fbinary : float
        Binary fraction.
    n_star_amp : float
        Amplitudes are expressed as N_stars per bin. Otherwise amplitudes
        are expressed as solar masses per year (or Gyr; docs are unclear).
    """
    def __init__(self, name, synth, sfh_amps,
                 use_lockfile=True, delta_dmod=0.,
                 n_stars=10000, fext=1., gamma=-1.35, fbinary=0.5,
                 n_star_amp=True):
        super(TestPop, self).__init__()
        self.name = name
        assert len(self.name) <= 8
        self.synth = synth
        self.sfh_amps = sfh_amps
        self.use_lockfile = use_lockfile
        self.n_stars = n_stars
        self.delta_dmod = delta_dmod
        self.fext = fext
        self.gamma = gamma
        self.fbinary = fbinary
        self.n_star_amp = n_star_amp

    def run(self):
        # Setup synth
        # This synth path is hard-coded into testpop
        testpop_synth_path = 'testpop/synth.dat'
        self.synth._write(n_cpu=1, include_unlocked=False,
                          synth_input_path=testpop_synth_path)

        # Setup the test pop config
        config_path = 'testpop/{0}.txt'.format(self.name)
        self._write_config(config_path)

        # Run testpop
        cmd = "./testpop/testpop < %s" % config_path
        with EnterStarFishDirectory():
            subprocess.call(cmd, shell=True)

    def _write_config(self, path):
        lines = []

        lines.append(self.name)
        lines.append(str(int(self.use_lockfile)))
        lines.append(str(int(self.n_stars)))
        lines.append('{0:.2f}'.format(self.delta_dmod))
        lines.append('{0:.2f}'.format(self.gamma))
        lines.append('{0:.2f}'.format(self.fbinary))
        if self.n_star_amp:
            lines.append('0')
        else:
            lines.append('1')

        # The values for gamma and fbinary supercede the values in synth.dat.
        # Immediately following these 8 parameters, the input file should
        # contain N lines with the following columns:
        #
        # amp (f7.4) z_metal (f5.3)  log_age (f5.2), niso (i2)
        #
        # You should have one line per isochrone in the isofile, if lockflag=0,
        # or one for each independent isochrone group (if lockflag=1).  amp is
        # the desired amplitude for each isochrone/isochrone group.
        # If sfrflag=1, then the amps will be taken as star formation rates,
        # with units Msun/Gyr. Otherwise, they are simply Nstars per age bin.
        # niso is the number of isochrones in the group (if lockflag=0, then
        # niso is not read).

        fmt = '{amp:7.4f} {z_metal:5.3f} {log_age:5.2} {niso:d}'
        group_metallicities = self.synth.lockfile.group_metallicities
        group_logages = self.synth.lockfile.group_logages
        group_nisoc = self.synth.lockfile.group_isochone_count
        for i in xrange(self.sfh_table.shape[0]):
            l = fmt.format(amp=self.sfh_amps[i],
                           z_metal=group_metallicities[i],
                           log_age=group_logages[i],
                           niso=group_nisoc[i])
            lines.append(l)

        txt = 'n'.join(lines)
        config_path = os.path.join(self.synth_dir, path)
        with open(os.path.join(starfish_dir, config_path), 'w') as f:
            f.write(txt)
