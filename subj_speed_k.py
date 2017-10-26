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

"Calculates how a participant's estimated k influences learning speed."

import pickle
import os
import statsmodels.api as sm
from scipy.special import logit
import numpy as np
from bfit_samples import get_samples, get_subject_meank
from bdata import bdata, N, NTRIALS
from mpl_meanresp_curve import KMAXCURVE, KSPEED_CURVES_FN

def main():
    "Calculates the mean k and mean responses for each participant."
    with open(KSPEED_CURVES_FN, 'rb') as inpf:
        mpl_kcurves = [pickle.load(inpf) for k in range(KMAXCURVE + 1)]
    # Determine interval with maximum difference in mean response
    ini, end = None, None
    dif = 0
    for i in range(NTRIALS - 100):
        j = i + 100
        this_dif = 0
        for trial in range(i, j):
            this_dif += mpl_kcurves[0][trial] - mpl_kcurves[2][trial]
        this_dif /= 100
        if this_dif > dif:
            dif = this_dif
            ini, end = i, j
    print(dif, ini, end)
    if not os.path.exists('mean_k.pickle'):
        samples = get_samples()
        samples = samples.sample(10000)
        mean_k = [get_subject_meank(samples, i) for i in range(N)]
        with open('mean_k.pickle', 'wb') as outf:
            pickle.dump((ini, end), outf)
            pickle.dump(mean_k, outf)
    else:
        with open('mean_k.pickle', 'rb') as inpf:
            ini, end = pickle.load(inpf)
            mean_k = pickle.load(inpf)
    # ini, end = 200, 300
    mean_resp = [np.mean(y[ini:end]) for x, y in bdata]
    mod = sm.OLS(logit(mean_resp), [(1, k) for k in mean_k])
    res = mod.fit()
    print(res.summary())

if __name__ == '__main__':
    main()
