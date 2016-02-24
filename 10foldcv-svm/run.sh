#!/bin/sh

for i in `seq 1 1`
do
    python 10foldcv_SVM_multithread.py linear
done
# `mv result* linear`


# for i in `seq 1 1`
# do
#     python 10foldcv_SVM_multithread.py rbf
# done
# `mv result* rbf`



# for i in `seq 1 1`
# do
#     python 10foldcv_SVM_multithread.py chi2
# done
# `mv result* chi2`


# for i in `seq 1 1`
# do
#     python 10foldcv_SVM_multithread.py chi2ng
# done
# `mv result* chi2ng`

# for i in `seq 1 1`
# do
#     python 10foldcv_SVM_multithread.py hi
# done
# `mv result* hi`