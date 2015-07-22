# encoding: utf-8
"""
Create artifical stellar catalogs with ``testpop``.
"""

import os
import subprocess

import numpy as np

from .pathutils import starfish_dir, EnterStarFishDirectory
from .pipeline import DatasetBase


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
        print "configpath is {0}".format(config_path)
        cmd = "./testpop < %s" % os.path.basename(config_path)
        with EnterStarFishDirectory(dirname='testpop'):
            print cmd
            print 'cwd:', os.getcwd()
            subprocess.call(cmd, shell=True)

        # Create a dataset instance with the testpop catalogs
        self.dataset = TestPopDataset(self.name, self.synth)

    def _write_config(self, config_path):
        lines = []

        lines.append(self.name)
        lines.append(str(int(self.use_lockfile)))
        lines.append(str(int(self.n_stars)))
        lines.append('{0:.2f}'.format(self.delta_dmod))
        lines.append('{0:.2f}'.format(self.fext))
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

        fmt = '{amp:7.4f} {z_metal:5.3f} {log_age:5.2f} {niso:d}'
        group_metallicities = self.synth.lockfile.group_metallicities
        group_logages = self.synth.lockfile.group_logages
        group_nisoc = self.synth.lockfile.group_isochrone_count
        for i in xrange(self.sfh_amps.shape[0]):
            l = fmt.format(amp=self.sfh_amps[i],
                           z_metal=group_metallicities[i],
                           log_age=group_logages[i],
                           niso=group_nisoc[i])
            lines.append(l)

        txt = '\n'.join(lines) + '\n'
        with open(os.path.join(starfish_dir, config_path), 'w') as f:
            f.write(txt)

    @property
    def sfh_table(self):
        """Numpy record array of the model star formation history."""
        n = len(self.sfh_amps)
        dtype = [('log(age)', np.float), ('Z', np.float), ('sfr', np.float),
                 ('dt', np.float), ('sfr_msolar_yr', np.float),
                 ('mass', np.float)]
        a = np.empty(n, dtype=np.dtype(dtype))
        a['log(age)'][:] = self.synth.lockfile.group_logages
        a['Z'][:] = self.synth.lockfile.group_metallicities
        a['sfr'][:] = self.sfh_amps
        # FIXME generalize for other IMF slopes
        a['dt'][:] = self.synth.lockfile.group_dt
        a['sfr_msolar_yr'][:] = a['sfr'] * self.n_stars * 1.628 / a['dt']
        a['mass'][:] = a['sfr'] * self.n_stars * 1.628  # M_sun born in bin
        return a

    @property
    def sfh_table_marginalized(self):
        sfh_table = self.sfh_table
        # t = np.empty(len(sfh_table), dtype=sfh_table.dtype)
        # sfh_table.read_direct(t, source_sel=None, dest_sel=None)
        age_vals = np.unique(sfh_table['log(age)'])
        s = np.argsort(age_vals)
        age_vals = age_vals[s]
        A = []
        sfr = []
        sfr_msolar_yr = []
        dt = []
        mass = []
        for i, age_val in enumerate(age_vals):
            tt = sfh_table[sfh_table['log(age)'] == age_val]
            bin_sfr = np.sum(tt['sfr'])
            bin_sfr_msolar_yr = np.sum(tt['sfr_msolar_yr'])
            bin_mass = np.sum(tt['mass'])
            A.append(age_val)
            sfr.append(bin_sfr)
            dt.append(tt['dt'][0])  # assume all metallicity tracks same dt
            sfr_msolar_yr.append(bin_sfr_msolar_yr)
            mass.append(bin_mass)
        srt = np.argsort(A)
        A = np.array(A)
        sfr = np.array(sfr)
        sfr_msolar_yr = np.array(sfr_msolar_yr)
        mass = np.array(mass)
        dt = np.array(dt)
        A = A[srt]
        sfr = sfr[srt]
        dt = dt[srt]
        mass = mass[srt]
        new_sfh_table = np.empty(len(A),
                                 dtype=np.dtype([('log(age)', float),
                                                 ('sfr', float),
                                                 ('dt', float),
                                                 ('sfr_msolar_yr', float),
                                                 ('mass', float)]))
        new_sfh_table['log(age)'][:] = A
        new_sfh_table['sfr'][:] = sfr
        new_sfh_table['sfr_msolar_yr'][:] = sfr_msolar_yr
        new_sfh_table['dt'][:] = dt
        new_sfh_table['mass'][:] = mass

        return new_sfh_table


class TestPopDataset(DatasetBase):
    """A Dataset for testpop-derived catalogs."""
    def __init__(self, prefix, synth):
        self._datasets = {}
        for plane in synth._cmds:
            dataset = self._load_catalog(prefix, plane.suffix)
            record = {'dataset': dataset, 'x_mag': plane.x_mag,
                      'y_mag': plane.y_mag}
            self._datasets[plane.suffix] = record
        super(TestPopDataset, self).__init__()

    def _load_catalog(self, prefix, suffix):
        path = os.path.join(starfish_dir, 'testpop',
                            ''.join((prefix, suffix)))
        # Read the dataset
        dtype = [('mag_x', float), ('mag_y', float),
                 ('delta_x', float), ('delta_y', float)]
        data = np.loadtxt(path, dtype=np.dtype(dtype))
        return data

    def write_phot(self, x_mag, y_mag, data_root, suffix):
        """write_phot overrides the normal execution to write
        out the pre-populated dataset from testpop.
        """
        data = self._datasets[suffix]['dataset']
        phot_path = os.path.join(starfish_dir,
                                 ''.join((data_root, suffix)))
        phot_dir = os.path.dirname(phot_path)
        if not os.path.exists(phot_dir):
            os.makedirs(phot_dir)
        np.savetxt(phot_path,
                   data,
                   fmt='%.4f')

    def get_phot(self, band):
        for suffix, rec in self._datasets.iteritems():
            if band == rec['x_mag']:
                return rec['dataset']['mag_x']
            elif band == rec['y_mag']:
                return rec['dataset']['mag_y']
        print "Could not find photometry for", band
