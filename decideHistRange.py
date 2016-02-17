# -*- coding: utf-8 -*-

import sys, os
import numpy as np


def decideRange_CtH(filenames):
    minimum = np.inf
    maximum = -np.inf

    for f in filenames:
        tmp = np.load( f.strip() )
        tmp_max = np.amax(tmp)
        tmp_min = np.amin(tmp)
        if tmp_max > maximum:
            maximum = tmp_max
        if tmp_min < minimum:
            minimum = tmp_min

    return maximum, minimum


def contrastHistogram(filename, maximum, minimum, nbin=50):
    print filename
    x = np.load( filename )
    CtH = np.histogram(x,  bins=nbin, range=(minimum,maximum), normed=False)[0]
    CtH = CtH / np.linalg.norm(CtH, ord=1)
    basename, ext = os.path.splitext( filename)
    np.save(basename + "-CtH.npy", CtH)


def concatenateHist(SI_file, CtH_file, output_path):
    SI = np.load(SI_file)
    CtH = np.load(CtH_file)
    feature = np.r_[SI, CtH]
    np.save(output_path, feature)



if __name__ == '__main__':

    f = open("nl_file.txt")
    lines = f.readlines()
    f.close()

    maximum, minimum = decideRange_CtH(lines)

    print "max:", maximum
    print "min:", minimum

    for l in lines:
        contrastHistogram(l.strip(), maximum, minimum)
