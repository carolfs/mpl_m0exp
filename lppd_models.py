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

"""
Calculates the CV score of a model from model samples.

The samples should have already been obtained by running cross_validation.py
Run:
$ python3 lppd_model.py PVL/WSLS/MPL seed
where seed is the number that initialized the random number generator for
data partition.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.misc import logsumexp
from scipy.special import expit
from mpl import mpl_logl
from bfit_samples import multivariate_t, KMAX
from bdata import N, bdata, NTRIALS
from cross_validation import K, SFN, PVL_WARMUP, MPL_WARMUP, WSLS_WARMUP

def calc_lppd(samples, modelf, subjs):
    """Calculates a model's lppd from samples and subject numbers."""
    if len(samples) > 1000:
        samples = samples.sample(1000)
    lppd = 0
    for subjindex, subj in enumerate(subjs):
        sample_lppds = []
        for _, sample in samples.iterrows():
            sample_lppds.append(modelf(sample, subj, subjindex))
        lppd += logsumexp(sample_lppds) - np.log(len(samples))
    return lppd

def mplf(sample, subj, _):
    """Calculates the MPL lppd for a sample and subject number in bdata."""
    x, y = bdata[subj]
    probk = [sample['probk.{}'.format(k + 1)] for k in range(KMAX + 1)]
    mu = [sample['mu.{}'.format(i + 1)] for i in range(3)]
    nu = sample['nu']
    sigma = [[sample['sigma.{}.{}'.format(i + 1, j + 1)] for j in range(3)]\
        for i in range(3)]
    pks = [0 for k in range(KMAX + 1)]
    A, rho, theta = multivariate_t(nu, mu, sigma)
    A, rho, theta = expit(A), expit(rho), np.exp(theta)
    if theta > 1e6:
        theta = 1e6
    for k in range(KMAX + 1):
        pks[k] = np.log(probk[k]) +\
            mpl_logl(x, y, 0, NTRIALS, k, A, rho, theta)
    return logsumexp(pks)

def pvlf(sample, subj, _):
    """Calculates the PVL lppd for a sample and subject number in bdata."""
    x, y = bdata[subj]
    mu = [sample['mu.{}'.format(i + 1)] for i in range(2)]
    nu = sample['nu']
    sigma = [[sample['sigma.{}.{}'.format(i + 1, j + 1)] for j in range(2)]\
        for i in range(2)]
    A, theta = multivariate_t(nu, mu, sigma)
    A, theta = expit(A), np.exp(theta)
    if theta > 1e6:
        theta = 1e6
    return mpl_logl(x, y, 0, NTRIALS, 0, A, 1, theta)

def wslsf(sample, _, subjindex):
    """Calculates the WSLS lppd for a sample and subject number (from 0)."""
    return sample['log_lik.{}'.format(subjindex + 1)]

def main():
    """Prints the CV score of the selected model."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Computes lppd for '
        'comparison of the PVL, WSLS, and MPL models.')
    parser.add_argument('model', help='model (PVL, WSLS, or MPL)', type=str)
    parser.add_argument('seed', help='seed for data set partition', type=int)
    args = parser.parse_args()
    model = args.model
    if model not in ('PVL', 'MPL', 'WSLS'):
        print('Invalid model.')
        sys.exit(0)
    lppd = 0
    for cvgroup in range(K):
        grouplen = N // K
        excluded = list(range(cvgroup*grouplen, (cvgroup + 1)*grouplen))
        sfn = SFN.format(args.seed, model.lower(), cvgroup)
        if not os.path.exists(sfn):
            print('Error: sample file {} does not exist'.format(sfn))
            sys.exit(0)
        if model == 'PVL':
            warmup = PVL_WARMUP
            modelf = pvlf
        elif model == 'MPL':
            warmup = MPL_WARMUP
            modelf = mplf
        else:
            warmup = WSLS_WARMUP
            modelf = wslsf
        samples = pd.read_csv(sfn, comment='#')
        samples = samples.iloc[warmup:]
        lppd += calc_lppd(samples, modelf, excluded)
        print(lppd)
        sys.stdout.flush()
    print(-2*lppd)

if __name__ == '__main__':
    main()
