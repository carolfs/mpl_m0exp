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

"""Simulates MPL agents searching for patterns."""

import random
import pickle
import numpy as np
from mpl import mpl

PATTERNS = [
    '01',
    '0011',
    '110010',
]
A = 1
RHO = 1
THETA = 1e6
REPS = 10000
RESULTS_FN = 'mpl_demo.pickle'
KMAXP1 = 4

def main():
    """Runs pattern search simulations."""
    with open(RESULTS_FN, 'wb') as outf:
        for pat in PATTERNS:
            x = [int(i) for i in pat*(300//len(pat))]
            assert len(x) == 300
            for k in range(KMAXP1):
                acc = 0
                for _ in range(REPS):
                    y = [int(random.random() < p1) for p1 in mpl(x, k, A, RHO, THETA)]
                    acc += np.mean([int(xx == yy) for xx, yy in zip(x[100:], y[100:])])
                acc /= REPS
                print(pat, k, acc)
                pickle.dump((k, acc), outf)

if __name__ == '__main__':
    main()
