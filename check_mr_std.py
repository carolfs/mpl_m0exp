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

"""Check stdev of observed mean response with regard to MPL prediction."""

import sys
import numpy as np
from bfit_samples import get_samples, get_random_params
from bdata import N, bdata
import pandas as pd
from predictive_mean_resp import get_mpl_mr
from bayesian import hpd

def main():
    """Prints observed and predicted stdev of mean response."""
    samples = get_samples()

    # Predicted stdev of mean response for 84 new participants
    stds = []

    for i, (_, sample) in enumerate(samples.iterrows()):
        print('Calculating predicted stdev of mean response...')
        if i % 100 == 0:
            print('Rep {} of {}...'.format(i + 1, len(samples)))
        sys.stdout.flush()
        mrs = []
        for (x, _), (k, A, rho, t) in zip(bdata, get_random_params(sample, N)):
            mrs.append(get_mpl_mr(x, k, A, rho, t))
        stds.append(np.std(mrs, ddof=1))
    stds = pd.Series(stds)

    truestd = np.std([np.mean(y[-100:]) for x, y in bdata], ddof=1)
    print(
        "Mean predicted stdev, predicted stdev credital interval, stdev HPDI,",
        "observed stdev, prob[expected > observed]:",
        stds.mean(), (stds.quantile(0.025), stds.quantile(0.975)), hpd(stds),
        truestd, np.mean([int(std > truestd) for std in stds]))

if __name__ == '__main__':
    main()
    