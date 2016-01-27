# -*- coding: utf-8 -*-

import sys, os
import math
import numpy as np
import cv2
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.ioff()

import TreeOfShapes as tos
from attributeGraph import *


# minimum and maximum value of histogram range of
# normalized gray level (CtH)
# You should change these values depending on texture images.
NL_max = 10.0
NL_min = 0.0


def traceChildrenNodesPix(g, n):
    if g.node[n]['leaf']:
        return g.node[n]['pixels']
    else:
        pix = g.node[n]['pixels']
        for i in g.node[n]['children']:
            pix = pix + traceChildrenNodesPix(g, i)
        return pix


def traceChildrenNodes(g, n):
    if g.node[n]['leaf']:
        return None
    else:
        ch = g.node[n]['children']
        for i in g.node[n]['children']
            if traceChildrenNodes(g, i) != None:
                ch = ch + self.traceChildrenNodes(g, i)
        return ch


def traceMParentNodes(g, n, M = 3):
    parents = []
    p = n
    for i in range(0, M):
        if g.node[p]['parent'] == -1:
            return None
        parents.append(g.node[p]['parent'][0])
        p = g.node[p]['parent'][0]
    return parents


def computeSITAFromTree(g, npmap, img, imgsize):
    darkElongation = []
    brightElongation = []
    darkCompactness = []
    brightCompactness = []
    darkScaleRatio = []
    brightScaleRatio = []
    normvalue = np.ones(imgsize, dtype='float')*-1

    # ============================================================
    # beginning of loop for each node
    # ============================================================
    for i in g.graph['orderNodes']:

        if g.node[i]['parent'] == -1:
            # pix = traceChildrenNodesPix(g, i)
            # g.node[i]['area'] = float(len(pix))
            continue

        if g.node[i]['grayLevel'] <= g.node[g.node[i]['parent'][0]]['grayLevel']:
            isDark = True
        else:
            isDark = False

        # moment ===============================================
        pix = traceChildrenNodesPix(g, i)
        tmp = np.zeros(imgsize, dtype='uint8')
        for p in pix:
            tmp[p[1], p[0]] = 255
        M = cv2.moments(tmp, binaryImage=1)

        # g.node[i]['area'] = M['m00']
        C = np.array( [ [M['nu20'], M['nu11']], [M['nu11'], M['nu02']] ] )
        eig, v = np.linalg.eig(C)

        lambda1 = max(eig)
        lambda2 = min(eig)
        if lambda2 <= 0.0:
            lambda2 = np.spacing(1)

        elongation = lambda2 / lambda1
        compact = 1.0 / (4.0 * math.pi * math.sqrt(lambda1*lambda2))
        if compact > 1:
            compact = 1.0

        # scale ratio =========================================
        parents = traceMParentNodes(g, i)
        if parents == None:
            continue
        ancestor_area = 0.0
        for parentIndex in parents:
            ancestor_area = ancestor_area + g.node[parentIndex]['area']
        alpha = g.node[i]['area'] / (ancestor_area / 3.0)

        # adding attribute to dark or bright ===================
        if isDark:
            darkElongation.append(elongation)
            darkCompactness.append(compact)
            darkScaleRatio.append(alpha)
        else:
            brightElongation.append(elongation)
            brightCompactness.append(compact)
            brightScaleRatio.append(alpha)

        # normalized gray level ================================
        if np.sum(npmap == i) != 0:
            num_pix = np.sum(npmap == i)
            grayvalues = []
            for p in pix:
                grayvalues.append(img[p[1], p[0]])
            meanval = float(sum(grayvalues)) / float(len(pix))
            g_x = np.array(grayvalues, dtype='float')
            sd = math.sqrt( sum((g_x - meanval) * (g_x - meanval)) / float(len(pix)) )
            if sd == 0.0:
                normvalue[npmap == i] = 0.0
            else:
                normvalue[npmap == i] = (img[npmap == i] - meanval) / sd
    # ============================================================
    # end of loop for each node
    # ============================================================

    # constructing histograms
    dark_EH    = np.histogram(darkElongation,    bins=25, range=(0,1), normed=True)[0]
    bright_EH  = np.histogram(brightElongation,  bins=25, range=(0,1), normed=True)[0]
    dark_CpH   = np.histogram(darkCompactness,   bins=25, range=(0,1), normed=True)[0]
    bright_CpH = np.histogram(brightCompactness, bins=25, range=(0,1), normed=True)[0]
    dark_SRH   = np.histogram(darkScaleRatio,    bins=25, range=(0,1), normed=True)[0]
    bright_SRH = np.histogram(brightScaleRatio,  bins=25, range=(0,1), normed=True)[0]

    EH  = np.r_[dark_EH,  bright_EH]
    CpH = np.r_[dark_CpH, bright_CpH]
    SRH = np.r_[dark_SRH, bright_SRH]
    CtH = np.histogram(normvalue.flatten(), bins=50,
                   range=(NL_min, NL_max), normed=True)[0]

    return EH, CpH, SRH, CtH


def SITA(filename, filterSize=10, fcomb='SI', isCtH=True):
    """
    fcomb: 'SI' or 'AI'
    SI = EH + CpH + SRH
    AI = EH + CpH

    isCtH: boolian
    If isCtH is True, CtH is concatenated to SI or AI.
    """
    src = cv2.imread(filename, 0).astype(np.uint32)

    # constructing tree of shapes
    AF_img = tos.areaFilter(src, size=filterSize)
    padding = tos.imagePadding(AF_img, 0)
    tree = tos.constructTreeOfShapes(padding, None)
    tos.addAttributeArea(tree)
    g, nodePixMap = createAttributeGraph(tree, padding)

    # compute each attribute histogram
    EH, CpH, SRH, CtH = computeSITAFromTree(g, nodePixMap, AF_img, AF_img.shape)

    # selecting output feature
    if fcomb == 'AI':
        feature = np.r_[EH, CpH]
    elif fcomb == 'SI':
        feature = np.r_[EH, CpH, SRH]
    else:
        print "Wrong string is specified. 'SI' was selected."
        feature = np.r_[EH, CpH, SRH]

    if isCtH:
        feature = np.r_[feature, CtH]

    # output
    basename, ext = os.path.splitext(filename)
    np.save(basename+".npy", feature)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        img_name = "brodatz_sample/D1-1.bmp"
    else:
        img_name = sys.argv[1]

    import time
    s = time.time()
    SITA(img_name, filterSize=10, fcomb='SI', isCtH=True)
    e = time.time()
    print e - s



