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

"MPL model implementation in Python."

import random
from numpy import exp

# MPL
def mpl_slow(xs, k, A, rho, theta):
    "Slower but easier to understand MPL implementation"
    num_mem = (1 << k)
    mem = [(0, 0) for i in range(num_mem)]
    eta = 0
    for t, x in enumerate(xs):
        if t < k:
            eta = (eta << 1) + x
            yield 0.5
        else:
            E0, E1 = mem[eta]
            f = theta * (E1 - E0)
            if f > 40: # Double precision
                yield 1
            elif f < -40:
                yield 0
            else:
                m0 = exp(f)
                yield m0/(m0 + 1)
            for i, (e0, e1) in enumerate(mem):
                mem[i] = (e0*A, e1*A)
            E0, E1 = mem[eta]
            E0 = rho * E0 + (1 if x == 0 else 0)
            E1 = rho * E1 + (1 if x == 1 else 0)
            mem[eta] = E0, E1
            if k > 0:
                eta = ((eta << 1) + x) % num_mem

def mpl(xs, k, A, rho, theta):
    "Faster MPL implementation"
    if k == 0:
        mem = 0
        for x in xs:
            f = theta*mem
            if f > 40:
                yield 1
            else:
                e = exp(f)
                yield e/(1 + e)
            mem = rho*A*mem + (2*x - 1)
    else:
        num_mem = (1 << k)
        mem = [0 for i in range(num_mem)]
        eta = 0
        for x in xs[:k]:
            eta = (eta << 1) + x
            yield 0.5
        for x in xs[k:]:
            f = theta*mem[eta]
            if f > 40:
                yield 1
            else:
                e = exp(f)
                yield e/(1 + e)
            for i, m in enumerate(mem):
                mem[i] = A*m
            mem[eta] = rho*mem[eta] + (2*x - 1)
            eta = ((eta << 1) + x) % num_mem

def test():
    "Tests the MPL implementations against each other"
    for _ in range(1000):
        xs = [int(random.random() < 0.7) for i in range(300)]
        k, A, rho, theta = random.randint(0, 3), random.random(), random.random(), random.uniform(0, 5)
        for _, (r1, r2) in enumerate(zip(mpl_slow(xs, k, A, rho, theta), mpl(xs, k, A, rho, theta))):
            if abs(r1 - r2) >= 1e-10:
                print(r1, r2, xs, k, A, rho, theta)
                raise Exception()

if __name__ == '__main__':
    test()
