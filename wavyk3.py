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

from mpl import mpl
import numpy as np
from bdata import NTRIALS
import random
from wavy import calc_wavy_first, calc_wavy_last, L
import pickle

A = 1
rho = 1
theta = 1e6
k = 3

N = 100000

results = []
for i in range(N):
    x = list((np.random.rand(NTRIALS) < 0.7).astype(int))
    y = [random.random() < p1 for p1 in mpl(x, k, A, rho, theta)]
    cf = [np.mean(i) for i in calc_wavy_first(x, y, L)]
    cl = [np.mean(i) for i in calc_wavy_last(x, y, L)]
    results.append((y, cf, cl))

print(np.mean([np.mean(y[-100:]) for y, cf, cl in results]))

with open('wavyk3.pickle', 'wb') as f:
    pickle.dump(results, f)
