# Shape-based Invariant Texture Analysis (SITA)

This repository implements a texture feature called *Shape-based Invariant Texture Analysis* (SITA) proposed by Xia et al. [1].

## Execution environment
* OS: Mac OS X (10.9 or 10.11)
* Language: Python 2.7.\* (Anaconda 2.4.\*)
* Modules: Numpy, Matplotlib, OpenCV, NetworkX, Cython


## Usage
    python setup.py build_ext --inplace
    python SITA.py [image file]


## Brief description for source files
### SegmentedShapeFeatures.py
This extracts SITA feature from an image and save as a binary file (.npy format).
To execute this scripts, it is necessary to import following modules．

### TreeOfShapes.pyx and PriorityQueue.pyx
This builds ToS.
The algorithm is based on [2].

### setup.py
This is used for compiling cython code (TreeOfShapes.pyx and PriorityQueue.pyx).

### attributeGraph.py
This converts ToS class of TreeOfShapes.pyx into NetworkX graph object.


## Sample images
Some of the Brodatz dataset images are stored in *brodatz_sample* directory.  
[Brodatz dataset](http://multibandtexture.recherche.usherbrooke.ca/original_brodatz.html)


## References
1. X.-S. Xia, J. Delon, and Y. Gousseau, "Shape-based Invariant Texture Indexing," International Journal of Computer Vision (IJCV), vol.88, no.3, pp.382–403, 2010.
2. T. Géraud, E. Carlinet, S. Crozet, and L. Najman, "A quasi-linear algorithm to compute the tree of shapes of nD images," International Symposium on Mathematical Morphology (ISMM), pp.98-110, 2013.
