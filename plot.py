#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os.path

from scripts.read_arma_mat import read_arma_mat_ascii

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def getargs():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('armaasciifilename', nargs='+')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    import seaborn as sns
    # ?
    sns.set()

    args = getargs()

    for filename in args.armaasciifilename:

        stub = os.path.splitext(filename)[0]

        res = read_arma_mat_ascii(filename)
        assert res.shape[0] == 1
        res = res[0]
        print(res)

        fig, ax = plt.subplots()

        sns.heatmap(res, linewidths=0.5, center=0, cmap='seismic', ax=ax)

        fig.savefig('{}.pdf'.format(stub), bbox_inches='tight')

        plt.close('all')
