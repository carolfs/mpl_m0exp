# Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>
#
# This program is free software: you can redistribute iters and/or modify
# iters under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that iters will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Runs cross-validation for the PVL, MPL, and WSLS models.
To get the LPPD, run lppd_models.py
"""

import sys
import random
import os
import argparse
from bdata import N, bdata, NTRIALS
from mpl_stan import KMAX, get_stan_model


K = 12
assert N % K == 0
SFN = 'cv-{}-{}-{:02d}.csv'
PVL_WARMUP = 1000
MPL_WARMUP = 10000
WSLS_WARMUP = 2500

def main():
    """Performs cross-validation of the model."""
    parser = argparse.ArgumentParser(
        description='Runs cross validation for '
        'comparison of the PVL and MPL models.')
    parser.add_argument('model', help='model (PVL, MPL or WSLS)', type=str)
    parser.add_argument(
        'set', help='participant set to exclude (0 to {})'.format(K - 1),
        type=int)
    parser.add_argument('seed', help='seed for data set partition', type=int)
    args = parser.parse_args()
    model = args.model
    if model not in ('PVL', 'MPL', 'WSLS'):
        print('Invalid model.')
        sys.exit(0)
    random.seed(args.seed)
    random.shuffle(bdata)
    exc_set = args.set
    set_len = N // K
    excluded = bdata[exc_set*set_len:(exc_set + 1)*set_len]
    included = bdata[:exc_set*set_len] + bdata[(exc_set + 1)*set_len:]
    assert len(excluded) + len(included) == len(bdata)

    sfn = SFN.format(args.seed, model.lower(), exc_set)
    sfn = os.path.join(os.getcwd(), sfn)
    if not os.path.exists(sfn):
        model_dat = {
            'T': NTRIALS,
            'N': len(included),
            'x': [x for x, y in included],
            'y': [y for x, y in included],
        }
        if model == 'PVL':
            stan_model = get_stan_model('model-pvl.stan', 'model-pvl')
            iters = 2*PVL_WARMUP
            warmup = PVL_WARMUP
        elif model == 'MPL':
            stan_model = get_stan_model('model-mpl.stan', 'model-mpl')
            iters = 2*MPL_WARMUP
            warmup = MPL_WARMUP
            model_dat['kmaxp1'] = KMAX + 1
        else:
            assert model == 'WSLS'
            stan_model = get_stan_model('model-wsls.stan', 'model-wsls')
            iters = 2*WSLS_WARMUP
            warmup = WSLS_WARMUP
            model_dat['N_'] = len(excluded)
            model_dat['x_'] = [x for x, y in excluded]
            model_dat['y_'] = [y for x, y in excluded]

        fit = stan_model.sampling(
            data=model_dat, iter=iters, warmup=warmup, chains=1, sample_file=sfn)
        with open(sfn[:-3] + 'txt', 'w') as outf:
            outf.write(str(fit))

if __name__ == '__main__':
    main()
