# -*- coding: utf-8 -*-

import sys
import numpy as np
import csv


allSamples = "featurefile.txt"
# allLabels  = "labels.txt"
allLabels  = "labels-bin.txt"


def readLabels():
    labels = np.genfromtxt(allLabels, delimiter=',', dtype=int)
    return labels


def readFeatures():
    # read text files
    f = open(allSamples)
    lines = f.readlines()
    f.close()
    for i, line in enumerate(lines):
        if i == 0:
            allFeatures = np.load(line.strip())
            continue
        feature = np.load(line.strip())
        allFeatures = np.vstack((allFeatures, feature))
    return allFeatures


def colPrint(strings, col='r'):
    if col == 'r':
        print "\033[31m%s\033[39m" % strings
    elif col == 'g':
        print "\033[32m%s\033[39m" % strings
    elif col == 'y':
        print "\033[33m%s\033[39m" % strings
    elif col == 'b':
        print "\033[34m%s\033[39m" % strings
    elif col == 'm':
        print "\033[35m%s\033[39m" % strings
    elif col == 'c':
        print "\033[36m%s\033[39m" % strings
    else:
        print strings


def red(strings):
    return "\033[31m%s\033[39m" % strings
def green(strings):
    return "\033[32m%s\033[39m" % strings
def yellow(strings):
    return "\033[33m%s\033[39m" % strings
def blue(strings):
    return "\033[34m%s\033[39m" % strings
def magenta(strings):
    return "\033[35m%s\033[39m" % strings
def cyan(strings):
    return "\033[36m%s\033[39m" % strings



if __name__ == '__main__':
    features = readFeatures()
    labels = readLabels()
    print features
    print labels
