# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import cv2

import TreeOfShapes as tos
from attributeGraph import *


# minimum and maximum value of histogram range of
# normalized gray level (CtH)
# You should change these values depending on texture images.
# NL_max = 10.0
# NL_min = -10.0

# NL_max = 4.79583152331
# NL_min = -12.6491106407

NL_max = 9.98163707411
NL_min = -25.5404893122


def printNLrange():
    print "SITA: NL_max:", NL_max, "NL_min:", NL_min

    
def setNLrange(maxVal, minVal):
	global NL_max
	global NL_min
	NL_max = float(maxVal)
	NL_min = float(minVal)


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
        for i in g.node[n]['children']:
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


def removeEmptyRootNode(g, nodePixMap):
    for i in g.graph['orderNodes']:
        if len(g.node[i]['pixels']) == 0:
            if np.sum(nodePixMap[nodePixMap==i]) == 0:
                if i == g.graph['orderNodes'][0]:
                    ch = g.node[i]['children']
                    g.remove_node(i)
                    g.graph['orderNodes'].remove(i)
                    g.node[ch[0]]['parent'] = -1
                    break


def computeSitaElementsFromTree(g, nodePixMap, img):
    imgsize = img.shape

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

        # dark or bright blob ?
        if g.node[i]['parent'] == -1:
            isDark = True
        else:
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

        # appendiing attributes for lists
        if isDark:
            darkElongation.append(elongation)
            darkCompactness.append(compact)
        else:
            brightElongation.append(elongation)
            brightCompactness.append(compact)

        # scale ratio =========================================
        parents = traceMParentNodes(g, i)
        if parents != None:
            ancestor_area = 0.0
            for parentIndex in parents:
                ancestor_area = ancestor_area + g.node[parentIndex]['area']
            alpha = g.node[i]['area'] / (ancestor_area / 3.0)

            # appending attribute for list
            if isDark:
                darkScaleRatio.append(alpha)
            else:
                brightScaleRatio.append(alpha)

        # normalized gray level ================================
        if np.sum(nodePixMap == i) != 0:
            num_pix = np.sum(nodePixMap == i)
            grayvalues = []
            for p in pix:
                grayvalues.append(img[p[1], p[0]])
            meanval = float(sum(grayvalues)) / float(len(pix))
            g_x = np.array(grayvalues, dtype='float')
            sd = math.sqrt( sum((g_x - meanval) * (g_x - meanval)) / float(len(pix)) )
            if sd == 0.0:
                normvalue[nodePixMap == i] = 0.0
            else:
                normvalue[nodePixMap == i] = (img[nodePixMap == i] - meanval) / sd
    # ============================================================
    # end of loop for each node
    # ============================================================

    # constructing histograms
    dark_EH    = np.histogram(darkElongation,    bins=25, range=(0,1), normed=False)[0]
    bright_EH  = np.histogram(brightElongation,  bins=25, range=(0,1), normed=False)[0]
    dark_CpH   = np.histogram(darkCompactness,   bins=25, range=(0,1), normed=False)[0]
    bright_CpH = np.histogram(brightCompactness, bins=25, range=(0,1), normed=False)[0]
    dark_SRH   = np.histogram(darkScaleRatio,    bins=25, range=(0,1), normed=False)[0]
    bright_SRH = np.histogram(brightScaleRatio,  bins=25, range=(0,1), normed=False)[0]

    # normalize
    if np.sum(dark_EH) == 0:
        dark_EH = np.zeros(25, dtype='float')
    else:
        dark_EH = dark_EH / np.linalg.norm(dark_EH.astype('float'), ord=1)

    if np.sum(bright_EH) == 0:
        bright_EH = np.zeros(25, dtype='float')
    else:
        bright_EH = bright_EH / np.linalg.norm(bright_EH.astype('float'), ord=1)

    if np.sum(dark_CpH) == 0:
        dark_CpH = np.zeros(25, dtype='float')
    else:
        dark_CpH = dark_CpH / np.linalg.norm(dark_CpH.astype('float'), ord=1)

    if np.sum(bright_CpH) == 0:
        bright_CpH = np.zeros(25, dtype='float')
    else:
        bright_CpH = bright_CpH / np.linalg.norm(bright_CpH.astype('float'), ord=1)

    if np.sum(dark_SRH) == 0:
        dark_SRH = np.zeros(25, dtype='float')
    else:
        dark_SRH = dark_SRH / np.linalg.norm(dark_SRH.astype('float'), ord=1)

    if np.sum(bright_SRH) == 0:
        bright_SRH = np.zeros(25, dtype='float')
    else:
        bright_SRH = bright_SRH / np.linalg.norm(bright_SRH.astype('float'), ord=1)

    EH  = np.r_[dark_EH,  bright_EH]
    CpH = np.r_[dark_CpH, bright_CpH]
    SRH = np.r_[dark_SRH, bright_SRH]

    return EH, CpH, SRH, normvalue.flatten()


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

    # remove paddint (root) node
    removeEmptyRootNode(g, nodePixMap)

    # compute each attribute histogram
    EH, CpH, SRH, NL = computeSitaElementsFromTree(g, nodePixMap, AF_img)

    # selecting output feature
    if fcomb == 'AI':
        feature = np.r_[EH, CpH]
    elif fcomb == 'SI':
        feature = np.r_[EH, CpH, SRH]
    else:
        print "Wrong string is specified. 'SI' was selected."
        feature = np.r_[EH, CpH, SRH]

    if isCtH:
        CtH = np.histogram(NL, bins=50, range=(NL_min, NL_max), normed=False)[0]
        if np.sum(CtH) == 0:
            CtH = np.zeros(50, dtype='float')
        else:
            CtH = CtH / float( np.linalg.norm(CtH, ord=1) )
        feature = np.r_[feature, CtH]

    # output
    basename, ext = os.path.splitext( filename )
    np.save( basename + "-SITA.npy", feature )


def SitaElements(filename, filterSize=10, fcomb='SI'):
    """
    This function outputs two numpy array objects:
    1. SI or AI
    2. Normalized gray level values (without generating CtH)

    fcomb: 'SI' or 'AI'
    SI = EH + CpH + SRH
    AI = EH + CpH
    """
    src = cv2.imread(filename, 0).astype(np.uint32)

    # constructing tree of shapes
    AF_img = tos.areaFilter(src, size=filterSize)
    padding = tos.imagePadding(AF_img, 0)
    tree = tos.constructTreeOfShapes(padding, None)
    tos.addAttributeArea(tree)
    g, nodePixMap = createAttributeGraph(tree, padding)

    # remove padding (root) node
    removeEmptyRootNode(g, nodePixMap)

    # compute each attribute histogram
    EH, CpH, SRH, NL = computeSitaElementsFromTree(g, nodePixMap, AF_img)

    # selecting output feature
    if fcomb == 'AI':
        feature = np.r_[EH, CpH]
    elif fcomb == 'SI':
        feature = np.r_[EH, CpH, SRH]
    else:
        print "Wrong string is specified. 'SI' was selected."
        feature = np.r_[EH, CpH, SRH]

    # output
    basename, ext = os.path.splitext( filename )
    if fcomb == 'AI':
        np.save( basename + "-AI.npy", feature )
    elif fcomb == 'SI':
        np.save( basename + "-SI.npy", feature )
    else:
        print "Wrong string is specified. 'SI' was selected."
        np.save( basename + "-SI.npy", feature )
    np.save( basename + "-NL.npy", NL )


def SITA_fromArray(inputArr, filterSize=10, fcomb='SI', isCtH=True):
    """
    fcomb: 'SI' or 'AI'
    SI = EH + CpH + SRH
    AI = EH + CpH

    isCtH: boolian
    If isCtH is True, CtH is concatenated to SI or AI.
    """
    src = inputArr.astype(np.uint32)

    # constructing tree of shapes
    AF_img = tos.areaFilter(src, size=filterSize)
    padding = tos.imagePadding(AF_img, 0)
    tree = tos.constructTreeOfShapes(padding, None)
    tos.addAttributeArea(tree)
    g, nodePixMap = createAttributeGraph(tree, padding)

    # remove paddint (root) node
    removeEmptyRootNode(g, nodePixMap)

    # compute each attribute histogram
    EH, CpH, SRH, NL = computeSitaElementsFromTree(g, nodePixMap, AF_img)

    # selecting output feature
    if fcomb == 'AI':
        feature = np.r_[EH, CpH]
    elif fcomb == 'SI':
        feature = np.r_[EH, CpH, SRH]
    else:
        print "Wrong string is specified. 'SI' was selected."
        feature = np.r_[EH, CpH, SRH]

    if isCtH:
        CtH = np.histogram(NL, bins=50, range=(NL_min, NL_max), normed=False)[0]
        if np.sum(CtH) == 0:
            CtH = np.zeros(50, dtype='float')
        else:
            CtH = CtH / float( np.linalg.norm(CtH, ord=1) )
        feature = np.r_[feature, CtH]

    return feature



if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) < 2:
        img_name = "brodatz_sample/D1-1.bmp"
    else:
        img_name = sys.argv[1]

    # function: SITA
    s = time.time()
    SITA(img_name, filterSize=10, fcomb='SI', isCtH=True)
    e = time.time()
    print "computational time of SITA:", (e - s)

    # function: SitaElements
    s = time.time()
    SitaElements(img_name, filterSize=10, fcomb='SI')
    e = time.time()
    print "computational time of SitaElements:", (e - s)


