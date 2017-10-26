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

# Creates all the plots in the paper.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from bdata import bdata, N, NTRIALS
import pandas as pd
from mpl_stan import KMAX
from bfit_samples import *
import pickle
import sys
import statsmodels.api as sm
from outcome_probs import PROBS, OUTCOME_PROBS_FN
from bayesian import hpd
from mpl_meanresp_curve import KMAXCURVE, KSPEED_CURVES_FN, MEAN_RESP_CURVE_FN
from scipy.special import logit

if __name__ == '__main__':

    # Default style

    FONT_SIZE = 7
    LINEWIDTH = 1
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['lines.linewidth'] = LINEWIDTH
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    # Prevents bug that inserts space between point and decimal digits in TeX
    #mpl.rcParams['text.usetex'] = True

    # Plots of parameter recovery analysis

    from param_recover import PARAM_RECOVERY_FN
    params = pd.read_csv(PARAM_RECOVERY_FN)
    color = (0, 0, 0, 1)
    fig = plt.figure()
    fig.set_size_inches(19/2.54, 10/2.54)

    marker='.'
    markersize='0.1'
    
    plt.subplot(2, 3, 1)
    plt.title(r'$A$ parameter for $k>0$')
    plt.plot(
        params[params.k > 0].theta, params[params.k > 0].kld_A, marker,
        color=color, markersize=markersize)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Kullback-Leibler divergence')
    
    plt.subplot(2, 3, 2)
    plt.title(r'$\rho$ parameter for $k>0$')
    plt.plot(
        params[params.k > 0].theta, params[params.k > 0].kld_rho, marker,
        color=color, markersize=markersize)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Kullback-Leibler divergence')

    plt.subplot(2, 3, 3)
    plt.title(r'$A\rho$ parameter for $k=0$')
    plt.plot(
        params[params.k == 0].theta, params[params.k == 0].kld_Axrho, marker,
        color=color, markersize=markersize)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Kullback-Leibler divergence')

    plt.subplot(2, 3, 4)
    plt.title(r'$\theta$ parameter')
    plt.plot(
        params.theta, params.kld_theta, marker,
        color=color, markersize=markersize)
    plt.xlim(0, 5)
    plt.ylim(0, None)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Kullback-Leibler divergence')

    plt.subplot(2, 3, 5)
    plt.title(r'$k$ parameter')
    plt.plot(
        params.theta, params.kld_k, marker,
        color=color, markersize=markersize)
    plt.xlim(0, 5)
    plt.ylim(0, 1.8)
    plt.xlabel(r'$\theta$')
    plt.ylabel('Kullback-Leibler divergence')

    plt.tight_layout()
    plt.savefig('paramrecover.eps', dpi=300)
    plt.close()

    # Load predictive parameter samples
    mpl.rcParams['text.usetex'] = False
    with open('params_predictive.pickle', 'rb') as f:
        ps = pickle.load(f)
        As = pickle.load(f)
        rhos = pickle.load(f)
        thetas = pickle.load(f)

    # k predictive distribution
    fig = plt.figure()
    fig.set_size_inches(9/2.54, 6/2.54)
    axes = plt.gca()
    w, h = 0.6, 0.9
    axes.set_position((0.2, 0.05, w, h))
    ps = [ps[0], ps[1], ps[2], sum(ps[3:])]
    plt.pie(
        ps,
        labels=[r'$k = {}$ ({}%)'.format(k, int(round(p*100)))\
            if k <= 2 else '' for k, p in enumerate(ps)],
        labeldistance=0.2,
        colors=[(0.9, 0.9, 0.9), (0.75, 0.75, 0.75), (0.6, 0.6, 0.6), (0.45, 0.45, 0.45)],
    )
    plt.axis('equal')
    plt.savefig('k_dist.eps', dpi=300)
    plt.close()

    # A, rho, theta predictive distribution
    fig = plt.figure()
    fig.set_size_inches(19/2.54, 5/2.54)
    for i, (param, ys, xmin, xmax) in enumerate((('A', As, 0, 1),
        ('\\rho', rhos, 0, 1), ('\\theta', thetas, 0, 1.5))):
        axes = plt.subplot(1, 3, i + 1)
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        plt.hist(ys, bins=80, normed=True, range=(xmin, xmax), color='#000000')
        yl = r'$\Pr({}'.format(param) + '|\\mathscr{D})$'
        plt.ylabel(yl)
        plt.xlabel('${}$'.format(param))
        plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.savefig('A_rho_theta_dist.eps', dpi=300)
    plt.close()
    mpl.rcParams['text.usetex'] = True

    # Wavy effect with CAB-k and k=3
    from wavy import L
    fig = plt.figure()
    fig.set_size_inches(18/2.54, 6/2.54)
    with open('wavyk3.pickle', 'rb') as inpf:
        results = pickle.load(inpf)
    ff = [np.mean([cf[i] for y, cf, cl in results if not np.isnan(cf[i])]) for i in range(L)]
    fl = [np.mean([cl[i] for y, cf, cl in results if not np.isnan(cl[i])]) for i in range(L)]

    axes = plt.subplot(1, 2, 1)
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    plt.title(r'Wavy effect')
    markersize = 4
    plt.plot((0,), (ff[0],), 'o', color='k', markersize=markersize)
    plt.plot(np.arange(1, len(ff)), ff[1:], 'o-', color='k',
        markersize=markersize, label='Trials 1--100')
    plt.plot((0,), (fl[0],), 's', color=(0.7, 0.7, 0.7), markersize=markersize)
    plt.plot(np.arange(1, len(fl)), fl[1:], 's-', color=(0.7, 0.7, 0.7),
        markersize=markersize, label='Trials 201--300')
    plt.xticks(np.arange(len(ff)), (['Mean'] + [str(i) for i in range(1, len(ff) - 1)] + ['{}+'.format(len(ff) - 1)]))
    plt.ylim(0.5, 1)
    plt.xlim(-0.5, len(ff) - 0.5)
    plt.xlabel('Trials since most recent 0')
    plt.ylabel('Mean response')
    plt.legend(loc='lower right')

    axes = plt.subplot(1, 2, 2)
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    plt.title(r'Mean response curve')
    y = [np.mean([y[t] for y, cf, cl in results]) for t in range(NTRIALS)]
    plt.plot(np.arange(1, NTRIALS + 1), y, '-', color='k')
    plt.ylim(0, 1)
    plt.xlabel('Trial')
    plt.ylabel('Mean response')

    plt.tight_layout()
    plt.savefig('wavyk3.eps', dpi=300)
    plt.close()

    # Wavy effect plot
    fig = plt.figure()
    fig.set_size_inches(18/2.54, 6/2.54)
    with open('part_wavy.pickle', 'rb') as inpf:
        f1 = pickle.load(inpf)
        f2 = pickle.load(inpf)
    with open('sim_wavy.pickle', 'rb') as inpf:
        repf1 = pickle.load(inpf)
        repf2 = pickle.load(inpf)
    t1 = 'Trials 1--100'
    t2 = 'Trials 201--300'
    for i, (f, repf, t) in enumerate(((f1, repf1, t1), (f2, repf2, t2))):
        axes = plt.subplot(1, 2, i + 1)
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')

        plt.title(t)
        markersize = 4

        yerr = [[], []]
        for i, l in enumerate(f):
            m = l.mean()
            lb, ub = hpd(l)
            yerr[0].append(m - lb)
            yerr[1].append(ub - m)
        f = [l.mean() for l in f]
        plt.plot((0,), (f[0],), 'o', color='k', markersize=markersize)
        plt.errorbar(np.arange(len(f)), f, fmt='none',
            yerr=yerr, ecolor='k')
        plt.plot(np.arange(1, len(f)), f[1:], 'o-', color='k',
            label='Observed', markersize=markersize)

        yerr = [[], []]
        for i, l in enumerate(repf):
            m = np.mean(l)
            lb, ub = hpd(l)
            yerr[0].append(m - lb)
            yerr[1].append(ub - m)
        repf = [np.mean(l) for l in repf]
        plt.plot((0,), (repf[0],), 's', color='#999999', markersize=markersize,
            markeredgecolor='#999999')
        plt.errorbar(np.arange(len(f)), repf, fmt='none', yerr=yerr, ecolor='#999999')
        plt.plot(np.arange(1, len(f)), repf[1:], 's-', color='#999999', label='Predicted',
            markersize=markersize, markeredgecolor='#999999')

        plt.xticks(np.arange(len(f)), (['Mean'] + [str(i) for i in range(1, len(f) - 1)] + ['{}+'.format(len(f) - 1)]))
        plt.ylim(0.5, 1)
        plt.xlim(-0.5, len(f) - 0.5)
        plt.xlabel('Trials since most recent 0')
        plt.ylabel('Mean response')
        plt.legend()
    plt.tight_layout()
    plt.savefig('wavy.eps', dpi=300)
    plt.close()

    # MPL demonstration of pattern search
    import find_pattern
    fig = plt.figure()
    fig.set_size_inches((6*len(find_pattern.PATTERNS))/2.54, 6/2.54)
    with open(find_pattern.RESULTS_FN, 'rb') as f:
        for n, pat in enumerate(find_pattern.PATTERNS):
            plt.subplot(1, len(find_pattern.PATTERNS), n + 1)
            axes = plt.gca()
            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            axes.yaxis.set_ticks_position('left')
            x = np.arange(0, find_pattern.KMAXP1)
            plt.xlabel('$k$')
            plt.ylabel('Accuracy')
            plt.title('Pattern {}'.format(pat))
            plt.ylim(0, 1)
            h = [pickle.load(f)[1] for k in range(find_pattern.KMAXP1)]
            plt.bar(x, h, color='#cccccc')
            plt.xticks(x)
            plt.xlim(-0.5, 3.5)
            plt.grid(True, axis='y')
            plt.yticks(np.linspace(0, 1, 6))
    plt.tight_layout()
    plt.savefig('find_pattern.eps', dpi=300)
    plt.close()

    # Mean response distribution
    fig = plt.figure()
    fig.set_size_inches(9/2.54, 6/2.54)
    with open('mr_dist.pickle', 'rb') as f:
        mrs = pickle.load(f)
    axes = plt.gca()
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    plt.hist([np.mean(y[-100:]) for x, y in bdata], bins=15, normed=True, color=(0.3, 0.3, 0.3, 1), label='Observed', linewidth=0)
    plt.hist(mrs, bins=len(mrs.unique()), normed=True, color=(0, 0, 0, 0.3), label='Predicted', linewidth=0)
    plt.ylabel('Frequency')
    plt.xlabel('Mean response')
    plt.xlim(0, 1)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('mean_resp_dist.eps', dpi=300)
    plt.close()

    # Pattern search does not imply probability matching
    colors = [
        '#1b9e77',
        '#d95f02',
        '#7570b3',
        '#e7298a',
        '#66a61e',
        '#e6ab02',
    ]
    colors.reverse()
    fig = plt.figure()
    fig.set_size_inches(19/2.54, 12/2.54)
    from pattern_search_performance import PARAM_SETS, KMAX_PATSEARCH
    with open('mpl_pattern_search_curves.pickle', 'rb') as f:
        for i, (A, rho, theta) in enumerate(PARAM_SETS):
            axes = plt.subplot(2, 2, i + 1)
            plt.title(r'MPL $A={}, \rho={}, \theta={}$'.format(A, rho,
                theta if theta < 1 else r'\infty'))
            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            axes.yaxis.set_ticks_position('left')
            for k in range(KMAX_PATSEARCH + 1):
                mr = pickle.load(f)
                plt.plot(np.arange(1, len(mr)+1), mr, color=colors[k], label='$k={}$'.format(k))
            plt.ylim(0, 1)
            plt.xlabel('Trial')
            plt.ylabel('Mean response')
            plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('pattern_search.eps', dpi=300)
    plt.close()

    # How the outcome probability affects the predicted mean response
    fig = plt.figure()
    fig.set_size_inches(9/2.54, 6/2.54)
    axes = plt.gca()
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    with open(OUTCOME_PROBS_FN, 'rb') as f:
        for i, p in enumerate(PROBS):
            y = pickle.load(f)
            plt.plot(np.arange(1, len(y) + 1), y, color=colors[i], label=r'$p={:0.1f}$'.format(p))
    plt.xlabel('Trial')
    plt.ylabel('Mean response')
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.legend(loc='lower right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('outcome_probs.eps', dpi=300)
    plt.close()

    # Participants' learning curve vs model's
    fig = plt.figure()
    fig.set_size_inches(9/2.54, 6/2.54)
    axes = plt.gca()
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    xx = np.arange(1, NTRIALS + 1)
    yy = [np.mean([y[t] for x, y in bdata]) for t in range(NTRIALS)]
    plt.xlabel('Trial')
    plt.ylabel('Mean response')
    plt.ylim(0, 1)
    plt.plot((0, NTRIALS), (0.7, 0.7), color='#666666')
    plt.plot(xx, yy, color='#d7191c', label='Observed')
    with open(MEAN_RESP_CURVE_FN, 'rb') as f:
        yy = pickle.load(f)
    plt.plot(xx, yy, color='#2c7bb6', label='Predicted')
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig('meanresp_curve.eps', dpi=300)
    plt.close()

    # MPL curves for different ks
    with open(KSPEED_CURVES_FN, 'rb') as f:
        
        fig = plt.figure()
        fig.set_size_inches(9/2.54, 6/2.54)
        axes = plt.gca()
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        xs = np.arange(1, NTRIALS + 1)
        colors = [
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
        ]
        for k in range(KMAXCURVE + 1):
            ys = pickle.load(f)
            plt.plot(xs, ys, color=colors[k], label='$k={}$'.format(k))
        plt.ylim(0, 1)
        plt.ylabel('Mean response')
        plt.xlabel('Trial')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('mpl_speed_k.eps', dpi=300)
        plt.close()

    # Linear regression for mean response
    fig = plt.figure()
    fig.set_size_inches(19/2.54, 6/2.54)
    with open('mean_k.pickle', 'rb') as f:
        ini0, end0 = pickle.load(f)
        mean_k = pickle.load(f)
    for i, (ini, end) in enumerate(((ini0, end0), (200, 300))):
        axes = plt.subplot(1, 2, i+1)
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        mean_resp = [np.mean(y[ini:end]) for x, y in bdata]
        for i, mr in enumerate(mean_resp):
            if mr == 1:
                mean_resp[i] = 0.995
        plt.title('Trials {}--{}'.format(ini+1, end))
        lgtmr = logit(mean_resp)
        plt.plot(mean_k, lgtmr, '.', color='#000000')
        mod = sm.OLS(lgtmr, [(1, k) for k in mean_k])
        res = mod.fit()
        b, a = res.params
        linet = r'$\mathrm{logit}(y) = ' + '{:0.2f} - {:0.2f} x$'.format(b, -a)
        linet += '\n' + r'$R^2 = {:.02f}$'.format(res.rsquared)
        plt.annotate(linet, (1.5, logit(0.93)))
        x = np.array((0, 2.1))
        plt.plot(x, a*x + b, '-', color='#000000')
        plt.xlabel('Mean $k$')
        plt.ylabel('Mean response')
        plt.ylim(-0.5, 3.5)
        locs = logit(np.linspace(0.45, 0.95, 6))
        labels = ['{:0.2f}'.format(expit(i)) for i in locs]
        plt.yticks(locs, labels)
    plt.tight_layout()
    plt.savefig('mean_k_mean_resp.eps', dpi=300)
    plt.close()

    # How the learning curve changes with optimal parameters
    fig = plt.figure()
    fig.set_size_inches(9/2.54, 6/2.54)
    with open('params_performance.pickle', 'rb') as f:
        # plt.subplot(1, 2, 1)
        axes = plt.gca()
        axes.spines['right'].set_color('none')
        axes.spines['top'].set_color('none')
        axes.xaxis.set_ticks_position('bottom')
        axes.yaxis.set_ticks_position('left')
        x = np.arange(1, NTRIALS + 1)
        plt.xlabel('Trial')
        plt.ylabel('Mean response')
        plt.ylim(0, 1)
        labels = (
            'Predicted',
            'No pattern search',
            'No recency',
            'No exploration',
            # 'No pattern search, no recency',
            # 'No pattern search, no exploration',
            # 'No recency, no exploration',
            'Optimal',
        )
        colors = (
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
        )
        for i, (label, color) in enumerate(zip(labels, colors)):
            if i == 4:
                # Skip three results
                for j in range(3):
                    y = pickle.load(f)
            y = pickle.load(f)
            plt.plot(x, y, color=color, label=label)
        plt.grid(True, axis='y')
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('params_performance.eps', dpi=300)
    plt.close()
