# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
cimport cython

from collections import deque
ctypedef np.uint32_t INT_t


class PriorityQueue:
    def __init__(self, levels):
        self.data = []
        for _ in range(levels):
            self.data.append(deque())
        self.levels = levels
        self.size=0

    def push(self, int level, element):
        self.data[level].append(element)
        self.size = self.size+1

    def pop(self, int level):
        if len(self.data[level])>0:
            self.size=self.size-1
            return self.data[level].popleft()
        return None

    def priorityPop(self, int currentLevel):
        newLevel = self.findClosestNonEmpty(currentLevel)
        newPoint = self.pop(newLevel)
        return (newLevel, newPoint)

    def priorityPush(self, int point,
                     np.ndarray[INT_t, ndim=1] low_input,
                     np.ndarray[INT_t, ndim=1] up_input,
                     int currentLevel):
        cdef unsigned int low = low_input[point]
        cdef unsigned int up  = up_input[point]
        newLevel = min(up,max(low,currentLevel))
        self.push(newLevel, point)

    def isEmpty(self):
        return self.size==0

    def isLevelEmpty(self,level):
        return len(self.data[level])==0

    def findClosestNonEmpty(self,level):
        if not self.isLevelEmpty(level):
            return level
        lvlb=level-1
        lvlu=level+1
        while lvlb>=0 or lvlu<self.levels:
            if lvlb>=0:
                if not self.isLevelEmpty(lvlb):
                    return lvlb
                lvlb=lvlb-1
            if lvlu<self.levels:
                if not self.isLevelEmpty(lvlu):
                    return lvlu
                lvlu=lvlu+1
        return None
