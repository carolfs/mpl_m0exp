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

"Calculates the predicted mean response curve for any k and for k = 0..3"

import pickle
import numpy as np
from bdata import NTRIALS
from mpl import mpl
from bfit_samples import get_samples, get_random_params

REPS = 1000000
KMAXCURVE = 3
MEAN_RESP_CURVE_FN = 'mpl_meanresp_curve.pickle'
KSPEED_CURVES_FN = 'mpl_kspeed_curves.pickle'

def main():
    """Creates the mean response curves and saves them to a file."""
    samples = get_samples()
    mean_resp = [0 for i in range(NTRIALS)]
    for _, sample in samples.sample(REPS).iterrows():
        for k, A, rho, theta in get_random_params(sample):
            x = list((np.random.rand(NTRIALS) < 0.7).astype(int))
            for i, p1 in enumerate(mpl(x, k, A, rho, theta)):
                assert not np.isnan(p1)
                mean_resp[i] += p1
    mean_resp = [r/REPS for r in mean_resp]
    with open(MEAN_RESP_CURVE_FN, 'wb') as outf:
        pickle.dump(mean_resp, outf)
    # Curves for various ks
    mean_resps = [[0 for i in range(NTRIALS)] for k in range(KMAXCURVE + 1)]
    for k, mean_resp in enumerate(mean_resps):
        for _, sample in samples.sample(REPS).iterrows():
            for _, A, rho, theta in get_random_params(sample):
                x = list((np.random.rand(NTRIALS) < 0.7).astype(int))
                for i, p1 in enumerate(mpl(x, k, A, rho, theta)):
                    assert not np.isnan(p1)
                    mean_resp[i] += p1
    with open(KSPEED_CURVES_FN, 'wb') as outf:
        for mean_resp in mean_resps:
            mean_resp = [r/REPS for r in mean_resp]
            pickle.dump(mean_resp, outf)

if __name__ == '__main__':
    main()
