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

"""Loads experimental data (x, y sequences) from data file."""

import os
import pickle
import pandas as pd

def load_behavioral_data():
    """Loads the behavioral data and saves it in a more convenient format."""
    if not os.path.exists('bdata.bin'):
        dtf = pd.read_csv('bdata.csv')
        bdata = []
        for part in dtf.participant.unique():
            x, y = [], []
            for _, row in dtf[dtf.participant == part].iterrows():
                x.append(row.x)
                y.append(row.y)
            bdata.append((x, y))
            del x
            del y
        del dtf
        with open('bdata.bin', 'wb') as outf:
            pickle.dump(bdata, outf)
    else:
        with open('bdata.bin', 'rb') as inf:
            bdata = pickle.load(inf)
    return bdata

bdata = load_behavioral_data()
N = len(bdata) # Number of participants
NTRIALS = len(bdata[0][0]) # Number of trials
