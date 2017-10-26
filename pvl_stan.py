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

"""Fits the PVL model to the experimental data."""

import os
import sys
import argparse
from bdata import bdata, N, NTRIALS
from mpl_stan import get_stan_model

SFN = 'pvl_samples_{:04d}.csv'

def fit_pvl_stan():
    """Fits the PVL model to the experimental data."""
    parser = argparse.ArgumentParser(
        description='Fits the PVL(2) model to data using Stan.')
    parser.add_argument('chains', help='number of chains', type=int)
    parser.add_argument(
        '--iter', help='number of iterations (default 11000)', type=int,
        default=11000)
    parser.add_argument(
        '--warmup', help='number of warmup samples (default 1000)', type=int,
        default=1000)
    args = parser.parse_args()
    chains = args.chains
    if chains < 1:
        print('Invalid number of chains.')
        sys.exit(0)

    # Sample
    stan_model = get_stan_model('model-pvl.stan', 'model-pvl')
    model_dat = {
        'T': NTRIALS,
        'N': N,
        'x': [x for x, y in bdata],
        'y': [y for x, y in bdata],
    }
    sample_file_name = SFN.format(args.warmup)
    sample_file_name = os.path.join(os.getcwd(), sample_file_name)
    fit = stan_model.sampling(
        data=model_dat, iter=args.iter, warmup=args.warmup, chains=chains,
        refresh=10, sample_file=sample_file_name)
    print(fit)

if __name__ == '__main__':
    fit_pvl_stan()
