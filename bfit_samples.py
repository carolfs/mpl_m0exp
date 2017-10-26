# Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Functions to manipulate MPL posterior samples from Stan."""

import sys
import os
import random
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, chi2
from scipy.special import expit
from scipy.misc import logsumexp
from mpl_stan import SFN, KMAX
from bdata import bdata, NTRIALS
from mpl import mpl_logl

def get_sample_files():
    """Returns Stan sample files."""
    fns = []
    ininame = SFN.format(0)[:-4]
    for filename in os.listdir():
        if filename[:len(ininame)] == ininame and filename[-4:] == '.csv' and\
                filename[-8:] != '_all.csv':
            fns.append(filename)
    return fns

def get_samples():
    """Get all MPL samples as a Pandas dataframe."""
    fnall = 'mpl_samples_all.csv'
    if not os.path.exists(fnall):
        sample_files = get_sample_files()
        for i, filename in enumerate(sample_files):
            warmup = int(filename[:-4].split('_')[2])
            assert warmup > 0
            samples = pd.read_csv(filename, comment='#')
            print('Loading {} samples from file {} of {} ...'.format(
                len(samples), i + 1, len(sample_files)))
            sys.stdout.flush()
            samples = samples.iloc[warmup:]
            if i == 0:
                all_df = samples
            else:
                all_df = pd.concat((all_df, samples), ignore_index=True)
            del samples
        all_df.to_csv(fnall)
        return all_df
    else:
        samples = pd.read_csv(fnall)
        return samples

def get_ks_hparams(samples):
    """Return k hyperparameters from samples as a list of Pandas series."""
    probk = [[] for k in range(KMAX+1)]
    for _, sample in samples.iterrows():
        for k in range(KMAX+1):
            probk[k].append(sample['probk.{}'.format(k + 1)])
    return [pd.Series(p) for p in probk]

def get_k(probsk):
    """Get random k from a list of probabilities."""
    prob = random.random()
    total_prob = 0
    for k, probk in enumerate(probsk):
        total_prob += probk
        if prob < total_prob:
            return k
    return len(probsk) - 1

def get_subject_meank(samples, part_num):
    """Calcules a participant's mean k from samples and participant number."""
    x, y = bdata[part_num]
    probsk = [0 for k in range(KMAX + 1)]
    mean_k = 0
    for _, sample in samples.iterrows():
        A = sample['A.{}'.format(part_num + 1)]
        rho = sample['rho.{}'.format(part_num + 1)]
        theta = sample['theta.{}'.format(part_num + 1)]
        for k in range(KMAX + 1):
            probsk[k] = np.log(sample['probk.{}'.format(k + 1)]) +\
                mpl_logl(x, y, 0, NTRIALS, k, A, rho, theta)
        sum_probsk_log = logsumexp(probsk)
        mean_k += sum(
            [k*np.exp(p - sum_probsk_log) for k, p in enumerate(probsk)])
    return mean_k / len(samples)

def multivariate_t(nu, mu, sigma):
    """
    Density of the multivariate t distribution with nu degress of freedom.

    Keyword parameters:
    nu -- degress of fredom
    mu -- location
    sigma -- scale
    """
    d = len(mu)
    Y = multivariate_normal.rvs(mean=np.zeros(d), cov=sigma)
    U = chi2.rvs(nu)
    return mu + Y*np.sqrt(nu/U)

def get_random_params(sample, size=1):
    """
    Get random MPL parameters from a sample.
    
    Keyword parameters:
    sample -- MPL sample
    size -- number of parameter sets to return (default: 1).

    Returns a generator of (k, A, rho, theta) tuples.
    """
    probsk = [float(sample['probk.{}'.format(k + 1)]) for k in range(KMAX + 1)]
    mu = [float(sample['mu.{}'.format(i)]) for i in range(1, 4)]
    nu = float(sample['nu'])
    sigma = [[float(sample['sigma.{}.{}'.format(i, j)]) for j in range(1, 4)]\
        for i in range(1, 4)]
    for _ in range(size):
        k = get_k(probsk)
        A, rho, theta = multivariate_t(nu, mu, sigma)
        A, rho, theta = expit(A), expit(rho), np.exp(theta)
        if theta > 1e6: # To avoid numeric errors. This is a huge theta anyway.
            theta = 1e6
        yield k, A, rho, theta
