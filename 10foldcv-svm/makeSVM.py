

import sys
from sklearn import svm
from sklearn.externals import joblib
import option_10foldcv as op

if __name__ == '__main__':
    if len(sys.argv) < 2:
        modelName = "clf.model"
    else:
        modelName = sys.argv[1]

    # read feature vector and corresponding labels ======================
    op.colPrint("Reading features & labels ================", col='c')
    Features = op.readFeatures()
    Labels   = op.readLabels()
    if Features.shape[0] != Labels.shape[0]:
        sys.exit("The number of samples and labels are different.")
    nSamples = Features.shape[0]
    op.colPrint("Reading features & labels; done. Number of samples: %s" % nSamples, col='y')

    # create svm instance and training
    clf = svm.SVC(kernel='linear', C=10**1, probability=True).fit(Features, Labels)

    # export classifier
    print "Save classifier as", modelName
    joblib.dump(clf, modelName)
