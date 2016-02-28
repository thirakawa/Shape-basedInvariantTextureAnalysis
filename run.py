
import sys, os
import SITA as sita

if __name__ == '__main__':
    import time
    import itertools
    import multiprocessing
    from multiprocessing import Process

    def applyAll(filenames):
        for f in filenames:
            if f == None:
                break
            print f.strip()
            sita.SitaElements(f.strip(), filterSize=10, fcomb='SI')

    f = open('files.txt')
    lines = f.readlines()
    f.close()

    print "number of cpu:", multiprocessing.cpu_count()
    if multiprocessing.cpu_count() >= len(lines):
        N_seg = len(lines)
    else:
        N_seg = multiprocessing.cpu_count() - 1

    s = time.time()
    threads = []
    for files in itertools.izip_longest( *[iter(lines)] * (len(lines)/N_seg) ):
        threads.append( Process(target=applyAll, args=(files, )) )

    print "number of threads:", len(threads)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    e = time.time()
    print "extract SITA; done."
    print "Processing time", (e-s)

