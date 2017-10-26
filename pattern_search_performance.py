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
Calculates the mean response curves for different MPL parameter sets.

This illustrates how k, A, rho, and theta affect performance differently.
"""

import pickle
from mpl import mpl
import numpy as np

REPS = 1000000
KMAX_PATSEARCH = 5
NTRIALS = 1000
PARAM_SETS = (
    (1, 1, 1e10),
    (1, 1, 0.3),
    (0.95, 1, 1e10),
    (1, 0.8, 1e10),
)
PATSEARCH_CURVES_FN = 'mpl_pattern_search_curves.pickle'

def main():
    "Calculates and saves the mean response curves."
    with open(PATSEARCH_CURVES_FN, 'wb') as outf:
        for A, rho, theta in PARAM_SETS:
            for k in range(KMAX_PATSEARCH + 1):
                mean_resp = [0 for i in range(NTRIALS)]
                for _ in range(REPS):
                    x = list((np.random.rand(NTRIALS) < 0.7).astype(int))
                    for i, p1 in enumerate(mpl(x, k, A, rho, theta)):
                        mean_resp[i] += p1
                mean_resp = [r/REPS for r in mean_resp]
                pickle.dump(mean_resp, outf)

if __name__ == '__main__':
    main()
