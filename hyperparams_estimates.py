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

"""Calculates hyperparameter estimates from MPL samples."""

import collections
from bfit_samples import get_samples, get_ks_hparams, KMAX
from scipy.special import expit
from numpy import exp
import pandas as pd
from bayesian import hpd


def f2s(*xs):
    """Converts sequence of numbers to string with two decimal points."""
    rts = []
    for x in xs:
        if isinstance(x, collections.Sequence):
            rts.append(f2s(*x))
        else:
            rts.append('{:.2f}'.format(x))
    return '\t'.join(rts)

def main():
    """Prints hyperparameter estimates."""
    samples = get_samples()
    kprobs = get_ks_hparams(samples)
    print('Mean, credible interval, HPDI')
    for k in range(KMAX + 1):
        print(
            'Pr(k = {}) ='.format(k),
            f2s(
                kprobs[k].mean(),
                kprobs[k].quantile(0.025), kprobs[k].quantile(0.975),
                hpd(kprobs[k])))
    q = kprobs[1] + kprobs[2]
    print(
        'Pr(k = 1 or k = 2) =',
        f2s(q.mean(), q.quantile(0.025), q.quantile(0.975), hpd(q)))
    q = sum(kprobs[1:])
    print(
        'Pr(k >= 1) =',
        f2s(q.mean(), q.quantile(0.025), q.quantile(0.975), hpd(q)))
    q = sum(kprobs[3:])
    print(
        'Pr(k >= 3) =',
        f2s(q.mean(), q.quantile(0.025), q.quantile(0.975), hpd(q)))

    # Medians, not means!
    A, rho, theta = expit(samples['mu.1']), expit(samples['mu.2']),\
        exp(samples['mu.3'])

    for param, medians in zip(('A', 'rho', 'theta'), (A, rho, theta)):
        medians = pd.Series(medians)
        print(
            param,
            f2s(
                medians.mean(),
                medians.quantile(0.025), medians.quantile(0.975),
                hpd(medians)))
    # Correlations between parameters
    # Correlation is obtained by dividing the covariance by the product of stdev.
    cor = []
    for _, sample in samples.iterrows():
        scor = []
        for i in range(2):
            for j in range(i + 1, 3):
                scor.append(
                    sample['sigma.{}.{}'.format(i + 1, j + 1)]/\
                    (sample['scale.{}'.format(i + 1)]*sample['scale.{}'.format(j + 1)]))
        cor.append(scor)
    cols = ('corArho', 'corAtheta', 'corrhotheta')
    cor = pd.DataFrame(cor, columns=cols)
    for col in cols:
        print(
            col, f2s(
                cor[col].mean(), cor[col].quantile(0.025),
                cor[col].quantile(0.975), hpd(cor[col])))

if __name__ == '__main__':
    main()
