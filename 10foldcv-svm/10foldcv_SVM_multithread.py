# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import threading

from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel

import option_10foldcv as op


def one_trial_SVM(counter, Features, Labels, train, test, C, Scores, Kernel):
    print "%sth trial in 10-fold CV with %s" % (counter, C)
    F_train, F_test = Features[train], Features[test]
    L_train, L_test = Labels[train], Labels[test]

    # selecting kernels ===============
    if Kernel == 'rbf':
        clf = svm.SVC(kernel='rbf', C=C, probability=False).fit(F_train, L_train)
    elif Kernel == 'chi2':
        clf = svm.SVC(kernel=chi2_kernel, C=C, probability=False).fit(F_train, L_train)
    elif Kernel == 'chi2ng':
        clf = svm.SVC(kernel=additive_chi2_kernel, C=C, probability=False).fit(F_train, L_train)
    else:
        clf = svm.SVC(kernel='linear', C=C, probability=False).fit(F_train, L_train)

    Scores[counter-1] = clf.score(F_test, L_test)
    print "%sth trial in 10-fold CV with %s; done. %s" % (counter, C, Scores[counter-1])


def nfoldCV_SVM(Features, Labels, KF, C, i, Scores, kernel):
    nIterate = 1
    score = np.zeros([10])
    jobs = []
    op.colPrint("C: %s" % C, col='c')
    for train, test in KF:
        jobs.append( threading.Thread(target=one_trial_SVM, args=(nIterate, Features, Labels, train, test, C, score, kernel)) )
        nIterate = nIterate + 1
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    Scores[i,:] = score
    op.colPrint("C: %s; done." % C, col='y')



# main ==================================================================
if __name__ == '__main__':
    import time

    pid = os.getpid()
    print "pid:", pid

    if len(sys.argv) < 2:
        kernel = 'linear'
    else:
        kernel = sys.argv[1]
    print "kernel:", kernel

    # read feature vector and corresponding labels ======================
    op.colPrint("Reading features & labels ================", col='c')
    Features = op.readFeatures()
    Labels   = op.readLabels()
    if Features.shape[0] != Labels.shape[0]:
        sys.exit("The number of samples and labels are different.")
    nSamples = Features.shape[0]
    op.colPrint("Reading features & labels; done. Number of samples: %s" % nSamples, col='y')

    # split samples (N = 10) ============================================
    kf = KFold(nSamples, n_folds=10, shuffle=True, random_state=None)

    # Parameter of SVM ===========================
    C = np.logspace(-5, 5, num=11)
    Scores = np.zeros([len(C), 10])

    start = time.time()
    for i, c in enumerate(C):
        nfoldCV_SVM(Features, Labels, kf, c, i, Scores, kernel)
    end = time.time()
    print "time:", (end - start)

    mean_accuracy = np.mean(Scores, axis=1)
    np.savetxt("result_%s.csv" % pid, mean_accuracy, delimiter=',')
    np.savetxt("result_eachFold_%s.csv" % pid, Scores, delimiter=',')

    # graph =========================
    plt.plot(C, mean_accuracy)
    plt.xscale("log")
    plt.savefig("result_%s.pdf" % pid)

    op.colPrint("End of 10-fold CV.", col='y')

