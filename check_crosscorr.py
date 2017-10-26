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

"""Calculate cross-correlation of data and compare with predicted values."""

import random
import numpy as np
import pandas as pd
from mpl import mpl
from bfit_samples import get_samples, get_random_params
from bdata import bdata, N
from bayesian import hpd

def cross_correlation(x, y):
    """Calculates the cross-correlation of the sequences x, y."""
    return np.mean([(2*xx - 1)*(2*yy - 1) for xx, yy in zip(x[-101:-1], y[-100:])])

def exp_cross_correlation(samples):
    """Calculates the expected cross-correlation from MPL samples."""
    ccs = []
    REPS = 100000
    for _, s in samples.sample(REPS).iterrows():
        c = np.mean(
            [cross_correlation(x, [int(random.random() < p1) for p1 in mpl(x, k, A, rho, t)])\
            for (x, _), (k, A, rho, t) in zip(bdata, get_random_params(s, N))])
        ccs.append(c)
    return pd.Series(ccs)

def main():
    """Prints observed and expected cross-correlation values."""
    obs_cc = []
    for x, y in bdata:
        obs_cc.append(cross_correlation(x, y))
    mean_cc = np.mean(obs_cc)
    print('Observed (mean, stdev):', mean_cc, np.std(obs_cc))
    samples = get_samples()
    ecc = exp_cross_correlation(samples)
    print(
        'Expected (mean, credible inteval, HPDI, prob[expected < observed]):',
        ecc.mean(), (ecc.quantile(0.025), ecc.quantile(0.975)),
        hpd(ecc), np.mean([int(d < mean_cc) for d in ecc]))

if __name__ == '__main__':
    main()
