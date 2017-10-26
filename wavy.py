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

"""Calculates how the mean response changes with the last minority outcome."""

from bdata import bdata, N, NTRIALS
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from bfit_samples import get_samples, get_random_params
from mpl import mpl
import random
from bayesian import hpd
import pandas as pd
import sys
from mpl_stan import get_stan_model

L = 7
REPS = 100000

def calc_wavy_last(x, y, l):
    "Calculates the effect in the last 100 trials of the task."
    c = [[] for i in range(l)]
    n = 1
    for t in range(0, NTRIALS):
        if t >= 200:
            c[n].append(y[t])
            c[0].append(y[t])
        if x[t] == 0:
            n = 1
        else:
            n = min(l - 1, n + 1)
    return c

def calc_wavy_first(x, y, l):
    "Calculates the effect in the first 100 trials of the task."
    c = [[] for i in range(l)]
    for m, o in enumerate(x):
        if o == 0:
            break
    n = 1
    assert x[m] == 0
    for t in range(m + 1, NTRIALS):
        if t < 100:
            c[n].append(y[t])
            c[0].append(y[t])
        if x[t] == 0:
            n = 1
        else:
            n = min(l - 1, n + 1)
    return c

def main():
    # Participants
    if not os.path.exists('part_wavy.pickle'):
        # Run statistical analysis
        sm = get_stan_model('model-binomial.stan', 'model-binomial')
        f1 = []
        f2 = []
        with open('wavy_stanfit.txt', 'w') as of:
            for f, fun in ((f1, calc_wavy_first), (f2, calc_wavy_last)):
                dat = [[] for i in range(L)]
                for x, y in bdata:
                    for l1, l2 in zip(dat, fun(x, y, L)):
                        l1.append(l2)

                for i, l in enumerate(dat):
                    of.write('{} {}\n'.format(str(fun), i))
                    model_dat = {
                        'M': N,
                        'N': [len(x) for x in l],
                        'k': [sum(x) for x in l],
                    }
                    fit = sm.sampling(data=model_dat, iter=30000, warmup=25000, chains=4)
                    of.write(str(fit) + '\n')
                    samples = fit.extract()
                    s = pd.Series(samples['mean_p'])
                    f.append(s)
        with open('part_wavy.pickle', 'wb') as outf:
            pickle.dump(f1, outf)
            pickle.dump(f2, outf)
    else:
        with open('part_wavy.pickle', 'rb') as inpf:
            f1 = pickle.load(inpf)
            f2 = pickle.load(inpf)

    # Simulation
    

    if not os.path.exists('sim_wavy.pickle'):
        samples = get_samples()
        repf1 = [[] for i in range(L)]
        repf2 = [[] for i in range(L)]
        for n, s in samples.sample(REPS).iterrows():
            f1 = [0 for i in range(L)]
            f2 = [0 for i in range(L)]
            for (x, _), (k, A, rho, t) in zip(bdata, get_random_params(s, N)):
                y = [int(random.random() < p1) for p1 in mpl(x, k, A, rho, t)]
                c = [np.mean(i) for i in calc_wavy_first(x, y, L)]
                f1 = [fi + ci for fi, ci in zip(f1, c)]
                c = [np.mean(i) for i in calc_wavy_last(x, y, L)]
                f2 = [fi + ci for fi, ci in zip(f2, c)]
            f1 = [i/N for i in f1]
            f2 = [i/N for i in f2]

            for l, i in zip(repf1, f1):
                l.append(i)
            for l, i in zip(repf2, f2):
                l.append(i)
        with open('sim_wavy.pickle', 'wb') as outf:
            pickle.dump(repf1, outf)
            pickle.dump(repf2, outf)
    else:
        with open('sim_wavy.pickle', 'rb') as inpf:
            repf1 = pickle.load(inpf)
            repf2 = pickle.load(inpf)

if __name__ == '__main__':
    main()