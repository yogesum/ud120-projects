#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def compute(C=1.0, kernel='linear'):
  clf = SVC(C=C, kernel=kernel)

  t0 = time()
  clf.fit(features_train, labels_train)
  print("training time: {}s".format(round(time() - t0, 3)))

  t1 = time()
  test_predict = clf.predict(features_test)
  accuracy = accuracy_score(labels_test, test_predict)
  print('The accuracy of SVC EmailClassifier is {}'.format(accuracy * 100))
  print("prediction time: {}s".format(round(time() - t1, 3)))

compute()

print()
print('Slice training data to 1% of original')
features_train = features_train[:len(features_train)//100]
labels_train = labels_train[:len(labels_train)//100]
compute()

print()
print('Change kernal to `rbf`')
compute(kernel='rbf')

print()
print('Change kernal to `rbf` & `C`=10.')
compute(C=10., kernel='rbf')

print()
print('Change kernal to `rbf` & `C`=100.')
compute(C=100., kernel='rbf')

print()
print('Change kernal to `rbf` & `C`=1000.')
compute(C=1000., kernel='rbf')

print()
print('Change kernal to `rbf` & `C`=10000.')
compute(C=10000., kernel='rbf')

#
#
#########################################################


