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

"""Functions for Bayesian data analysis."""

import math

def hpd(sampleVec, credMass=0.95):
    """
    Calculates the highest density interval.

    Computes highest density interval from a sample of representative values,
    estimated as shortest credible interval.
    Adapted from:
    Kruschke, J. K. (2015). Doing Bayesian Data Analysis, Second Edition:
    A Tutorial with R, JAGS, and Stan. Academic Press / Elsevier.
    https://sites.google.com/site/doingbayesiandataanalysis/software-installation

    Keyword arguments:
    sampleVec
        is a vector of representative values from a probability distribution.
    credMass
        is a scalar between 0 and 1, indicating the mass within the credible
        interval that is to be estimated (default: 0.95).
    Returns:
    HDIlim is a vector containing the limits of the HDI
    """
    sortedPts = list(sampleVec)
    sortedPts.sort()
    ciIdxInc = math.ceil(credMass * len(sortedPts))
    nCIs = len(sortedPts) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(nCIs):
        ciWidth[i] = sortedPts[i + ciIdxInc] - sortedPts[i]
    j = ciWidth.index(min(ciWidth))
    HDImin = sortedPts[j]
    HDImax = sortedPts[j + ciIdxInc]
    HDIlim = (HDImin, HDImax)
    return HDIlim
