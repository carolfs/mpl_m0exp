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

"""Calculates the predictive mean response for a participant or sample."""

import pickle
import random
import numpy as np
from mpl import mpl
from bfit_samples import get_samples, get_random_params
from bdata import N, bdata, NTRIALS
import pandas as pd
from hyperparams_estimates import f2s
from bayesian import hpd

def get_mr(ys):
    "Calculates the mean response in the last 100 trials."
    return np.mean(ys[-100:])

def get_mpl_mr(xs, k, A, rho, theta):
    "Calculates the mean response of an MPL agent with the given parameters."
    ys = [int(random.random() < p1) for p1 in mpl(xs, k, A, rho, theta)]
    return get_mr(ys)

KMAX_MEAN_RESP = 3
REPS = 10000

def main():
    "Calculates the predicted mean responses."
    truemr = np.mean([get_mr(ys) for xs, ys in bdata])
    samples = get_samples()

    # Predicted mean response for a new participant with a specific k
    with open('mr_dist_by_k.pickle', 'wb') as outf:
        for k in range(KMAX_MEAN_RESP + 1):
            mrs = []

            for _, sample in samples.iterrows():
                for _, A, rho, theta in get_random_params(sample):
                    xs = list((np.random.random(NTRIALS) < 0.7).astype(int))
                    mean_resp = get_mpl_mr(xs, k, A, rho, theta)
                    mrs.append(mean_resp)
            mrs = pd.Series(mrs)
            print(
                'k =', k, f2s(
                    mrs.mean(), mrs.std(), mrs.quantile(0.025),
                    mrs.quantile(0.975), *hpd(mrs)))
            pickle.dump(mrs, outf)

    # Predicted mean response for a new participant
    mrs = []

    for _, sample in samples.iterrows():
        for k, A, rho, theta in get_random_params(sample):
            xs = list((np.random.random(NTRIALS) < 0.7).astype(int))
            mean_resp = get_mpl_mr(xs, k, A, rho, theta)
            mrs.append(mean_resp)

    mrs = pd.Series(mrs)
    with open('mr_dist.pickle', 'wb') as outf:
        pickle.dump(mrs, outf)

    print(
        f2s(
            mrs.mean(), mrs.std(), mrs.quantile(0.025), mrs.quantile(0.25),
            mrs.quantile(0.75), mrs.quantile(0.975), *hpd(mrs)))

    # Predicted mean response for 84 new participants
    mrs1 = []
    for _, sample in samples.sample(REPS).iterrows():
        mr1 = 0
        for (x, _), (k, A, rho, theta) in zip(bdata, get_random_params(sample, N)):
            mean_resp = get_mpl_mr(x, k, A, rho, theta)
            mr1 += mean_resp
        mr1 /= N
        mrs1.append(mr1)
    mrs1 = pd.Series(mrs1)
    print(
        f2s(
            mrs1.mean(), mrs1.std(), mrs1.quantile(0.025), mrs1.quantile(0.975),
            *hpd(mrs1), np.mean([int(mean_resp > truemr) for mean_resp in mrs1])))

if __name__ == '__main__':
    main()
