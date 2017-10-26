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

"Calculates the mean response for different probs of the majority outcome."

import pickle
import os
import numpy as np
from mpl import mpl
from bfit_samples import get_samples, get_random_params

NTRIALS = 1000
PROBS = (0.5, 0.6, 0.7, 0.8, 0.9, 1)
OUTCOME_PROBS_FN = 'outcome_probs.pickle'

def main():
    "Calculates the mean response curves and mean responses for each p"
    samples = get_samples()
    reps = 1000000
    if not os.path.exists(OUTCOME_PROBS_FN):
        with open(OUTCOME_PROBS_FN, 'wb') as outf:
            for prob in PROBS:
                # Curves for various ps
                mean_resp = [0 for i in range(NTRIALS)]
                for _, sample in samples.sample(reps).iterrows():
                    for k, A, rho, theta in get_random_params(sample):
                        x = list((np.random.rand(NTRIALS) < prob).astype(int))
                        for i, p1 in enumerate(mpl(x, k, A, rho, theta)):
                            mean_resp[i] += p1
                mean_resp = [r/reps for r in mean_resp]
                pickle.dump(mean_resp, outf)
    reps = 100000
    for prob in PROBS:
        mean_resp = 0
        for _, sample in samples.sample(reps).iterrows():
            for k, A, rho, theta in get_random_params(sample):
                x = list((np.random.rand(NTRIALS) < prob).astype(int))
                mean_resp += list(mpl(x, k, A, rho, theta))[-1]
        mean_resp /= reps
        print(prob, mean_resp)

if __name__ == '__main__':
    main()
