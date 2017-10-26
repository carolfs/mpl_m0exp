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

"Calculates the mean response curve for different MPL parameter sets."

import pickle
import numpy as np
from bfit_samples import get_samples, get_random_params
from mpl import mpl
from bdata import NTRIALS
from predictive_mean_resp import get_mr

TESTS = (
    lambda k, A, rho, theta: (k, A, rho, theta),
    lambda k, A, rho, theta: (0, A, rho, theta),
    lambda k, A, rho, theta: (k, A, 1, theta),
    lambda k, A, rho, theta: (k, A, rho, 1e6),
    lambda k, A, rho, theta: (0, A, 1, theta),
    lambda k, A, rho, theta: (0, A, rho, 1e6),
    lambda k, A, rho, theta: (k, A, 1, 1e6),
    lambda k, A, rho, theta: (0, A, 1, 1e6),
)
PARAMS_PERFORMANCE_FN = 'params_performance.pickle'
REPS = 1000000

def main():
    """Calculates and saves the mean response curves."""
    samples = get_samples()
    with open(PARAMS_PERFORMANCE_FN, 'wb') as outf:
        for _, test in enumerate(TESTS):
            mean_resp = np.zeros(NTRIALS)
            for _, sample in samples.sample(REPS).iterrows():
                for k, A, rho, theta in get_random_params(sample):
                    xs = list((np.random.random(NTRIALS) < 0.7).astype(int))
                    mean_resp += np.array(mpl(xs, *test(k, A, rho, theta)))
            mean_resp /= REPS
            pickle.dump(mean_resp, outf)
            print(get_mr(mean_resp))

if __name__ == '__main__':
    main()
