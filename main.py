#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:59:44 2019

@author: Brian Ho and Adelina Voukadinova
"""

#################
#Loading the Data
#################
import pandas as pd
import numpy as np
data = pd.read_csv('risk_factors_cervical_cancer.csv', delimiter=',')

data = data.replace('?', np.nan)
data = data.drop(['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis', 'Schiller', 'Hinselmann', 'Citology'], axis=1)
data = data.dropna()

#################
#Splitting the Data
#################
#np.random.seed(2019)
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)

x_train = train.iloc[:,:-1].values
y_train = train['Biopsy'].values
x_test = test.iloc[:,:-1].values
y_test = test['Biopsy'].values

#################
#Standardizing the Data
#################
from sklearn.preprocessing import StandardScaler

ss = StandardScaler() 
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#convert from np arrays back to dataframes
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

#################
#Feature Selection
#################
#ANOVA
#import sklearn.feature_selection as fs
#features = fs.f_classif(x_train, y_train)
#print(features[1])

#backward selection
from sklearn.linear_model import LogisticRegression

#lr = LogisticRegression(solver='liblinear') #saga
#rfe_lr = fs.RFE(estimator = lr)
#results = rfe_lr.fit(x_train, y_train).ranking_
#print(results)

#Lasso i.e. l1 regularization
lr_fs = LogisticRegression(solver='liblinear', penalty = 'l1') #saga
lr_fs.fit(x_train, y_train)

#get coefficients 
all_features = lr_fs.coef_[0,:].tolist()

index = -1
for feature in all_features:

    if feature == 0: #removes predictor from training set if coefficient is 0
        index = all_features.index(feature, index+1)
        #print(index)
        x_train.drop([index], axis = 1, inplace = True)
        x_test.drop([index], axis = 1, inplace = True)
    else:
        next

x_train = x_train.values
        
        
#################
#Grid Search
#################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

CV_acc = 0
best_c = -1000
C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
kfolds = StratifiedKFold(n_splits=10) #shuffle and randomstate??

#testing different values for C
for c in C:
    predicted = []
    #scores = []
    lr_gs = LogisticRegression(solver='liblinear', C=c) #saga
    
    #10-fold cross validation
    for train, test in kfolds.split(x_train, y_train):
        lr_gs.fit(x_train[train], y_train[train])
        pred = lr_gs.predict(x_train[test])
        for p in pred:
            predicted.append(p)
        #score = lr_gs.score(x_train[test], y_train[test])
        #scores.append(score)
    
    score = accuracy_score(y_train, predicted)
    if score > CV_acc:
        CV_acc = score
        best_c = c     
    
#    if np.mean(scores) > CV_acc:
#        CV_acc = np.mean(scores)
#        best_c = c
    
    
print('The best C is ' + str(best_c))

#################
#building the final LR model
#################
final_model = LogisticRegression(solver='liblinear', C=best_c) #saga
final_model.fit(x_train, y_train)

#################
#performing on the test set
#################
predicted = final_model.predict(x_test)
#lr_acc = model.score(x_test, y_test)
lr_acc = accuracy_score(y_test, predicted)
print(lr_acc)

##building SVM
#from sklearn.svm import SVC
#svm = SVC()
#svm.fit(train.iloc[:,:-1 ], train['Biopsy'])
#svm_acc = svm.score(test.iloc[:,:-1 ], test['Biopsy'])
#print(svm_acc)
#
##building decision tree
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier()
#tree.fit(train.iloc[:,:-1 ], train['Biopsy'])
#tree_acc = tree.score(test.iloc[:,:-1 ], test['Biopsy'])
#print(tree_acc)
#
##building random forest
#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier()
#rf.fit(train.iloc[:,:-1 ], train['Biopsy'])
#rf_acc = rf.score(test.iloc[:,:-1 ], test['Biopsy'])
#print(rf_acc)
#
##building KNN
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier()
#knn.fit(train.iloc[:,:-1 ], train['Biopsy'])
#knn_acc = knn.score(test.iloc[:,:-1 ], test['Biopsy'])
#print(knn_acc)
#
##building LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#lda = LinearDiscriminantAnalysis()
#lda.fit(train.iloc[:,:-1 ], train['Biopsy'])
#lda_acc = lda.score(test.iloc[:,:-1 ], test['Biopsy'])
#print(lda_acc)

