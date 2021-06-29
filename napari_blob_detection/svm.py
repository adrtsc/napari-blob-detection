# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:35:47 2020

@author: Adrian
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn import svm
import pandas as pd
import numpy as np


class SVM():

    def __init__(self, training_data, labels, split_ratio, weights, columns):
        self.weights = weights
        self.columns = columns
        self.training_data = training_data
        self.labels = labels
        self.split_ratio = split_ratio

    def train_svm(self):

        #######################################################################
        # PREPARE TRAINING SET AND TEST SET FOR CROSS VALIDATION
        #######################################################################

        training_data = self.training_data[self.columns]

        # split the annotatedtraining_data into training and test set

        (training_set,
         test_set,
         training_set_labels,
         test_set_labels) = train_test_split(training_data,
                                             self.labels,
                                             test_size=self.split_ratio,
                                             train_size=1-self.split_ratio,
                                             random_state=None)
        # normalize/scale thetraining_data
        # (important for PCA and SVM perfomance).
        # scale by Z-scoring (z = (x-mean)/std)

        scaler = StandardScaler()
        training_set_transformed = scaler.fit_transform(training_set)
        training_set_transformed = pd.DataFrame(training_set_transformed)

        #######################################################################
        # Randomized Parameter Optimization
        #######################################################################     

        # specify parameters and distributions to sample from
        clf = svm.SVC()
        param_dist = {'C': scipy.stats.expon(scale=100),
                      'gamma': scipy.stats.expon(scale=.1),
                      'kernel': ['rbf'], 'class_weight': [self.weights, None]}
        n_iter_search = 100
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search,
                                           scoring="f1",
                                           cv=5)

        random_search.fit(training_set_transformed, training_set_labels)

        self.clf = clf
        self.random_search = random_search
        self.training_set = training_set
        self.test_set = test_set
        self.scaler = scaler
        self.training_set_labels = training_set_labels
        self.test_set_labels = test_set_labels

    def report(self, n_top=3):

        results = self.random_search.cv_results_

        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def test(self):
        scaler = self.scaler
        test_set_transformed = scaler.transform(self.test_set)
        score = self.random_search.score(test_set_transformed,
                                         self.test_set_labels)

        return score

    def classify(self, data):

        # classify
        data2 = data[self.columns]
        data_transformed = self.scaler.transform(data2)
        data_transformed = pd.DataFrame(data_transformed)

        classification = self.random_search.predict(data_transformed)

        data['classification'] = classification

        return data
