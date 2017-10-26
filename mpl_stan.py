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

"Fits MPL model to behavioral data."

import os
import pickle
import sys
import argparse
import pystan
from bdata import bdata, N, NTRIALS

SFN = 'mpl_samples_{:04d}'
KMAX = 5

def get_stan_model(textfn, binfn):
    "Compiles a stan model (textfn) and saves it to binary file (binfn)."
    assert textfn != binfn
    if os.path.exists(binfn) and \
        os.path.getmtime(binfn) > os.path.getmtime(textfn):
        with open(binfn, 'rb') as arq:
            stan_model = pickle.load(arq)
    else:
        stan_model = pystan.StanModel(textfn)
        with open(binfn, 'wb') as arq:
            pickle.dump(stan_model, arq)
    return stan_model

def main():
    "Fits the MPL model to the data using Stan."
    parser = argparse.ArgumentParser(
        description='Fits the MPL model to data using Stan.')
    parser.add_argument(
        'chains', help='number of chains', type=int)
    parser.add_argument(
        '--iter', help='number of iterations (default 60000)', type=int,
        default=60000)
    parser.add_argument(
        '--warmup', help='number of warmup samples (default 10000)', type=int,
        default=10000)
    parser.add_argument(
        '--thin', help='the period for saving samples (default 1)', type=int,
        default=1)
    args = parser.parse_args()
    chains = args.chains
    if chains < 1:
        print('Invalid number of chains.')
        sys.exit(0)

    # Rename existing files so that Stan will not to overwrite them
    sfn = SFN.format(args.warmup // args.thin)
    sfn = os.path.join(os.getcwd(), sfn)
    if os.path.exists(sfn + '.csv'):
        file_num = 0
        while os.path.exists(sfn + '_{}.csv'.format(file_num)):
            file_num += 1
        os.rename(sfn + '.csv', sfn + '_{}.csv'.format(file_num))
    for chain in range(chains):
        if os.path.exists(sfn + '_{}.csv'.format(chain)):
            file_num = chains
            while os.path.exists(sfn + '_{}.csv'.format(file_num)):
                file_num += 1
            os.rename(sfn + '_{}.csv'.format(chain), sfn + '_{}.csv'.format(file_num))
    sfn = sfn + '.csv'
    # Sample
    mpl_stan = get_stan_model('model-mpl.stan', 'model-mpl')
    model_dat = {
        'kmaxp1': KMAX + 1,
        'T': NTRIALS,
        'N': N,
        'x': [x for x, y in bdata],
        'y': [y for x, y in bdata],
    }
    fit = mpl_stan.sampling(
        data=model_dat, iter=args.iter, warmup=args.warmup, chains=chains,
        thin=args.thin, refresh=10, sample_file=sfn)
    print(fit)

if __name__ == '__main__':
    main()
