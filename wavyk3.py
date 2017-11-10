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

"""Simulates MPL k = 3 and calculates how the mean resp. changes after a 0."""

import random
import pickle
from mpl import mpl
import numpy as np
from bdata import NTRIALS
from wavy import calc_wavy_first, calc_wavy_last, L

A = 1
RHO = 1
THETA = 1e6
K = 3

N = 100000

def main():
    "Runs the simulation and analysis, prints and saves the results."
    results = []
    for _ in range(N):
        x = list((np.random.rand(NTRIALS) < 0.7).astype(int))
        y = [random.random() < p1 for p1 in mpl(x, K, A, RHO, THETA)]
        cwf_results = [np.mean(i) for i in calc_wavy_first(x, y, L)]
        cwl_results = [np.mean(i) for i in calc_wavy_last(x, y, L)]
        results.append((y, cwf_results, cwl_results))

    print(np.mean([np.mean(y[-100:]) for y, cwf_results, cwl_results in results]))

    with open('wavyk3.pickle', 'wb') as outf:
        pickle.dump(results, outf)

if __name__ == '__main__':
    main()
    