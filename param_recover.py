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

"Parameter recovery for individual simulated agents."

import random
import numpy as np
from scipy import stats
from scipy import interpolate
from mpl import mpl
from bdata import NTRIALS
from mpl_stan import get_stan_model, KMAX

PARAM_RECOVERY_FN = 'param_recover_results.csv'

def cont_kldivergence(data, prior):
    """
    Calculates the Kullback–Leibler divergence for a continuous distribution.

    data: samples from the posterior distribution
    prior: pdf of the prior distribution
    """
    x, p = np.histogram(data, bins='auto')
    x = np.array(x)/np.sum(x)
    h = len(x)
    item = 0
    zeroitem = False
    x1 = []
    p1 = []
    for i in range(h):
        if x[i] > 0 and not zeroitem:
            x1.append(x[i] / (p[1] - p[0]))
            p1.append(np.mean((p[i], p[i + 1])))
            item += 1
        if x[i] == 0 and not zeroitem:
            zeroitem = True
            initial = i
        if x[i] > 0 and zeroitem:
            zeroitem = False
            x1.append(x[i] / (p[i + 1] - p[initial]))
            p1.append(np.mean(p[initial:i + 1]))
            item += 1
    x1 = np.array(x1)
    q1 = prior(p1)
    x2 = x1*np.log(x1/q1)
    tck = interpolate.splrep(p1, x2)
    return interpolate.splint(min(p1), max(p1), tck)

def k_kldivergence(samples, prior):
    """Calculates the Kullback–Leibler divergence for the k distribution."""
    counts = [0 for i in range(KMAX + 1)]
    for i in samples:
        k = i.astype('int')
        counts[k] += 1
    posterior = np.array(counts)/sum(counts)
    return stats.entropy(posterior, prior)

def get_uniform_prior(a, b):
    """Returns a function for a uniform prior in [a, b]."""
    return lambda x: 1/(b - a)

def product_uniform_prior(x):
    """Pdf of the product of two variables with uniform distribution in [0, 1]."""
    return -np.log(x)

def main():
    """Performs parameter recovery for simulated MPL agents and saves result to CSV file."""
    mpl_stan = get_stan_model('model-mpl-ind.stan', 'model-mpl-ind')
    k_prior = [1/(KMAX + 1) for k in range(KMAX + 1)]
    A_prior = get_uniform_prior(0, 1)
    rho_prior = get_uniform_prior(0, 1)
    theta_prior = get_uniform_prior(0, 5)
    Axrho_prior = product_uniform_prior
    with open(PARAM_RECOVERY_FN, 'w') as outf:
        outf.write('k,A,rho,theta,kld_k,kld_A,kld_rho,kld_theta,'\
            'kld_Axrho\n')
        for _ in range(10000):
            k = random.randint(0, KMAX)
            A = random.random()
            rho = random.random()
            theta = random.uniform(0, 5)
            x = list((np.random.random(NTRIALS) < 0.7).astype(int))
            y = [int(random.random() < p1) for p1 in mpl(x, k, A, rho, theta)]
            model_dat = {
                'kmaxp1': KMAX + 1,
                'T': NTRIALS,
                'x': x,
                'y': y,
            }
            while True:
                try:
                    fit = mpl_stan.sampling(data=model_dat, iter=3500, warmup=1000,
                        chains=4)
                except RuntimeError:
                    # Fail
                    continue
                else:
                    # Success
                    break
            samples = fit.extract()
            line = '{},{},{},{},{},{},{},{},{}\n'
            line = line.format(
                k, A, rho, theta,
                k_kldivergence(samples['k'], k_prior),
                cont_kldivergence(samples['A'], A_prior),
                cont_kldivergence(samples['rho'], rho_prior),
                cont_kldivergence(samples['theta'], theta_prior),
                cont_kldivergence(samples['Axrho'], Axrho_prior),
            )
            print(k, A, rho, A*rho, theta)
            print(fit)
            outf.write(line)

if __name__ == '__main__':
    main()
