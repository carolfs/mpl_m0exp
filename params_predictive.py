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

"Calculates the predictive distribution of MPL parameters."

import pickle
from bfit_samples import get_samples, get_random_params, get_ks_hparams

PARAMS_PREDICTIVE_FN = 'params_predictive.pickle'

def main():
    "Generates and saves random MPL parameters from the posterior distribution."
    samples = get_samples()
    As, rhos, thetas = [], [], []
    for _, sample in samples.sample(100000).iterrows():
        for _, A, rho, theta in get_random_params(sample):
            As.append(float(A))
            rhos.append(float(rho))
            thetas.append(float(theta))
    with open('params_predictive.pickle', 'wb') as outf:
        probk = get_ks_hparams(samples)
        pickle.dump([p.mean() for p in probk], outf)
        pickle.dump(As, outf)
        pickle.dump(rhos, outf)
        pickle.dump(thetas, outf)

if __name__ == '__main__':
    main()
