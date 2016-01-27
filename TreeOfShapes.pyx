# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython

import cv2, cv
from PriorityQueue import *

ctypedef np.uint32_t INT_t


# ==================================================================
# class: Tree Of Shapes
# ==================================================================
class TOS(object):
    def __init__(self, np.ndarray[INT_t, ndim=1] canonizedTreeImage, np.ndarray[INT_t, ndim=1] nodeLevels, sortedPixels, x, y):
        self.data = [None] * canonizedTreeImage.shape[0]
        self.size = [x, y]
        self.length = x*y
        self.levels = nodeLevels.astype(int).tolist()
        for j in xrange(canonizedTreeImage.shape[0]-1, -1, -1):
            i = sortedPixels[j]
            if (nodeLevels[i]!=nodeLevels[canonizedTreeImage[i]]):
                parent = i
            else:
                parent = canonizedTreeImage[i]
            if (self.data[parent] == None):
                self.data.append(-1)
                self.data[parent] = len(self.data)-1;
                self.levels.append(nodeLevels[parent])
            self.data[i] = self.data[parent]

        for j in xrange(canonizedTreeImage.shape[0]-1, -1, -1):
            i = sortedPixels[j]
            if (nodeLevels[i] != nodeLevels[canonizedTreeImage[i]]):
                parent = i
                pparent = canonizedTreeImage[parent]
                self.data[self.data[parent]] = self.data[pparent]
        self.addAttribute('reconstructedValue')

    def getParent(self, i):
        return self.data[i]

    def addAttribute(self,name,defaultValue=None):
        self.__dict__[name] = [defaultValue]*len(self.data)

    def reconstructImage(self,attributeName="levels",criterion=None):
        if criterion==None:
            criterion = (lambda _:True)
        root=len(self.data)
        for i in self.iterateFromRootToLeaves(True):
            if i>=self.length and (i==root or  criterion(i)) :
                self.reconstructedValue[i]=self.__dict__[attributeName][i]
            else:
                self.reconstructedValue[i]=self.reconstructedValue[self.data[i]]
        im = np.zeros([self.size[0]*self.size[1]], dtype='uint32')
        for i in range(im.shape[0]):
            im[i]=self.reconstructedValue[i]
        return im.reshape([self.size[1], self.size[0]])

    def iterateFromLeavesToRoot(self,includePixels=True):
        if includePixels:
            return range(len(self.data))
        else:
            return range(self.length,len(self.data),1)

    def iterateFromRootToLeaves(self,includePixels=True):
        if includePixels:
            return range(len(self.data)-1,-1,-1)
        else:
            return range(len(self.data)-1,self.length-1,-1)
# ==================================================================
# ==================================================================



def constructTreeOfShapes(image1, interpolation=max, verbose=False):
    #plain map
    if verbose:
        print("Plain Map Creation")
    low, up = interpolatePlainMapKhalimsky(image1)

    # sorting
    if verbose:
        print("Sorting")
    sortedPixels, enqueuedLevels = sort(low, up, np.amax(image1)+1)

    # generic tree construction
    if verbose:
        print("Union Find")
    parent = preTreeConstruction(low, sortedPixels)

    # canonize tree
    if verbose:
        print("Canonize")
    canonizeTree(parent,enqueuedLevels,sortedPixels)

    # encapsulated tree for ease of manipulation
    if verbose:
        print("Tree finalization")
    return TOS(parent, enqueuedLevels, sortedPixels, low.shape[1], low.shape[0])


# ==================================================================
# sorting
# ==================================================================
def sort(np.ndarray[dtype=INT_t, ndim=2] low_input,
         np.ndarray[dtype=INT_t, ndim=2] up_input,
         int levels):

    cdef unsigned int xorg = low_input.shape[1]
    cdef unsigned int yorg = low_input.shape[0]
    cdef np.ndarray[INT_t, ndim=1] low = low_input.flatten()
    cdef np.ndarray[INT_t, ndim=1] up  = up_input.flatten()
    cdef np.ndarray[np.uint8_t, ndim=1] dejaVu = np.zeros([yorg*xorg], dtype='uint8')
    cdef np.ndarray[INT_t, ndim=1] enqueuedLevels = np.zeros([yorg*xorg], dtype='uint32')
    sortedPixels = []

    # priority queue ===============
    cdef int startPoint = 0
    cdef int currentLevel, currentPoint, neighbor
    queue = PriorityQueue(levels)
    queue.push(low[startPoint], startPoint)
    currentLevel = low[startPoint]
    dejaVu[startPoint] = 1
    cdef int i = 0
    while not queue.isEmpty():
        (currentLevel, currentPoint) = queue.priorityPop(currentLevel)
        enqueuedLevels[currentPoint] = currentLevel
        i = i + 1
        sortedPixels.append(currentPoint)
        for neighbor in getNeighbourPoints(currentPoint, xorg, yorg):
            if dejaVu[neighbor] == 0:
                queue.priorityPush(neighbor, low, up, currentLevel)
                dejaVu[neighbor] = 1
    return sortedPixels, enqueuedLevels

def getNeighbourPoints(int currentPoint, int width, int height):
    cdef int x = currentPoint % width
    cdef int y = currentPoint // width
    nl = []
    if x-1>=0:
        nl.append(getCoordsLin(x-1, y, width))
    if y-1>=0:
        nl.append(getCoordsLin(x, y-1, width))
    if x+1<width:
        nl.append(getCoordsLin(x+1, y, width))
    if y+1<height:
        nl.append(getCoordsLin(x, y+1, width))
    return nl

def getCoordsLin(x, y, width):
    return y*width+x
# ==================================================================
# ==================================================================


def preTreeConstruction(np.ndarray[dtype=INT_t, ndim=2] img, sortedPixels):
    cdef int xorg = img.shape[1]
    cdef int yorg = img.shape[0]
    cdef int length = xorg * yorg
    cdef int currentPoint, cpReprez, cpNeigh

    cdef np.ndarray[INT_t, ndim=1] parent   = np.zeros([length], dtype='uint32')
    ufParent = [None] * length
    cdef np.ndarray[INT_t, ndim=1] ufRank   = np.zeros([length], dtype='uint32')
    cdef np.ndarray[INT_t, ndim=1] reprez   = np.zeros([length], dtype='uint32')

    for i in xrange(length-1, -1, -1):
        currentPoint           = sortedPixels[i]
        parent[currentPoint]   = currentPoint
        ufParent[currentPoint] = currentPoint
        reprez[currentPoint]   = currentPoint
        cpReprez               = currentPoint
        for neighbour in getNeighbourPoints(currentPoint, xorg, yorg):
            if ufParent[neighbour] != None:
                cpNeigh = findTarjan(neighbour, ufParent)
                if cpNeigh != cpReprez:
                    parent[reprez[cpNeigh]] = currentPoint
                    if ufRank[cpReprez]<ufRank[cpNeigh]:
                        cpNeigh,cpReprez=cpReprez,cpNeigh
                    ufParent[cpNeigh]=cpReprez
                    reprez[cpReprez]=currentPoint
                    if ufRank[cpReprez]==ufRank[cpNeigh]:
                        ufRank[cpReprez]=ufRank[cpReprez]+1
    return parent

def findTarjan(elem, Par):
    i = elem
    while Par[i] != i:
        i = Par[i]
    while Par[elem] != i:
        temp = elem
        elem = Par[elem]
        Par[temp] = i
    return i

# def unionTarjan(i, j, Par, Rank):
#     if Rank[i] > Rank[j]:
#         i, j = j, i
#     elif Rank[i] == Rank[j]:
#         Rank[j] = Rank[j] + 1
#     Par[i] = j
#     return j, i


def canonizeTree(np.ndarray[INT_t, ndim=1] parent,
                 np.ndarray[INT_t, ndim=1] enqueuedLevels,
                 sortedPixels):
    cdef int p, q
    for p in sortedPixels:
        q = parent[p]
        if enqueuedLevels[parent[q]] == enqueuedLevels[q]:
            parent[p] = parent[q]


# Attributes ==============================================
def addAttributeArea(tree):
    tree.addAttribute("area",0)
    for i in tree.iterateFromLeavesToRoot(True):
        if i<tree.length:
            tree.area[i]=1
        par=tree.data[i]
        if(par!=-1):
            tree.area[par]=tree.area[par]+tree.area[i]

def addAttributeDepth(tree):
    tree.addAttribute("depth",0)
    for i in tree.iterateFromRootToLeaves(False):
        par = tree.data[i]
        if (par!=-1):
            tree.depth[i] = tree.depth[par]+1
        else:
            tree.depth[i] = 1


# Filtering ==============================================
def areaFilter(im, size=100):
    im = imagePadding(im, 0)
    tree1= constructTreeOfShapes(im,None)
    addAttributeArea(tree1)
    reconstr = tree1.reconstructImage("levels",lambda x : tree1.area[x]>size)
    reconstr
    result = removePadding(reduceKhalimsky(reconstr.astype(np.uint32)))
    return result

def computeDepth(im):
    im = imagePadding(im, 0)
    tree1= constructTreeOfShapes(im,None)
    print 'tree1 = ', tree1.size
    print("Attribute depth")
    addAttributeDepth(tree1)
    print("Reconstruction")
    reconstr = tree1.reconstructImage("depth")
    result = removePadding(reduceKhalimsky(reconstr.astype(np.uint32)))
    return result



# ==================================================================
# Image padding, interpolation ...
# ==================================================================
def imagePadding(np.ndarray[dtype=INT_t, ndim=2] img, int grayValues=0):
    cdef unsigned int xorg = img.shape[1]
    cdef unsigned int yorg = img.shape[0]
    cdef unsigned int xdst = xorg + 2
    cdef unsigned int ydst = yorg + 2
    cdef np.ndarray[INT_t, ndim=2] dst = np.ones([ydst, xdst], dtype='uint32')
    dst = dst * grayValues
    dst[1:ydst-1:, 1:xdst-1] = img
    return dst

def removePadding(np.ndarray[dtype=INT_t, ndim=2] img):
    cdef unsigned int xorg = img.shape[1]
    cdef unsigned int yorg = img.shape[0]
    cdef unsigned int xdst = xorg - 2
    cdef unsigned int ydst = yorg - 2
    cdef np.ndarray[INT_t, ndim=2] dst = np.ones([ydst, xdst], dtype='uint32')
    dst = img[1:yorg-1:, 1:xorg-1]
    return dst

def getKhalimsky2FacesNeighbourValues(np.ndarray[dtype=INT_t, ndim=2] img, int x, int y):
    vals = []
    cdef unsigned int width  = img.shape[1]
    cdef unsigned int height = img.shape[0]
    cdef unsigned int minV, maxV

    if x%2==0 and y%2==0: # 0 face
        if x-1>=0:
            if y-1>=0:
                vals.append(img[y-1, x-1])
            if y+1<height:
                vals.append(img[y+1, x-1])
        if x+1<width:
            if y-1>=0:
                vals.append(img[y-1, x+1])
            if y+1<height:
                vals.append(img[y+1, x+1])
    elif y%2==1 and x%2==0: # vertical 1 face
        if x-1>=0:
            vals.append(img[y, x-1])
        if x+1<width:
            vals.append(img[y, x+1])
    elif y%2==0 and x%2==1: # horizontal 1 face
        if y-1>=0:
            vals.append(img[y-1,x])
        if y+1<height:
            vals.append(img[y+1,x])
    else : # 2 face !
        vals.append(img[y,x])

    minV = min(vals)
    maxV = max(vals)
    return minV, maxV

def interpolatePlainMapKhalimsky(np.ndarray[dtype=INT_t, ndim=2] img):
    cdef unsigned int xorg = img.shape[1]
    cdef unsigned int yorg = img.shape[0]
    cdef unsigned int xdst = xorg * 2 + 1
    cdef unsigned int ydst = yorg * 2 + 1
    cdef int x, y
    cdef unsigned int minV, maxV
    cdef np.ndarray[INT_t, ndim=2] low = np.zeros([ydst,xdst], dtype='uint32')
    cdef np.ndarray[INT_t, ndim=2] up  = np.zeros([ydst,xdst], dtype='uint32')
    # 2 faces copy
    for y in xrange(yorg):
        for x in xrange(xorg):
            low[y*2+1, x*2+1] = img[y, x]
            up[y*2+1, x*2+1]  = img[y, x]
    # <2 faces
    for y in xrange(ydst):
        for x in xrange(xdst):
            if y%2 != 1 or x%2 != 1:
                minV, maxV = getKhalimsky2FacesNeighbourValues(low, x, y)
                low[y, x] = minV
                up[y, x]  = maxV
    return low, up

def reduceKhalimsky(np.ndarray[INT_t, ndim=2] image, r=2):
    cdef int x, y, modx, mody, xori, yori
    cdef unsigned int xdst = image.shape[1]//r
    cdef unsigned int ydst = image.shape[0]//r
    cdef np.ndarray[INT_t, ndim=2] res = np.zeros([ydst, xdst], dtype='uint32')
    for y in xrange(1, image.shape[0], 2):
        for x in xrange(1, image.shape[1], 2):
            modx=x%2
            mody=y%2
            xori=x//2
            yori=y//2
            if modx==1 and mody==1:
                res[yori, xori] = image[y,x]
    return res
# ==================================================================
# ==================================================================
