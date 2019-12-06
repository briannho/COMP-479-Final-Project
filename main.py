#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:59:44 2019

@author: Brian Ho and Adelina Voukadinova
"""

#################
#PROCESSING THE DATA
#################
import pandas as pd
import numpy as np
data = pd.read_csv('risk_factors_cervical_cancer.csv', delimiter = ',')

#Remove Unwanted Data
data = data.replace('?', np.nan)
data = data.drop(['STDs: Time since last diagnosis', 'STDs: Time since first diagnosis', 'Schiller', 'Hinselmann', 'Citology'], axis = 1)
data = data.dropna()

#Splitting the Data
np.random.seed(479)
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.25)

x_train = train.iloc[:,:-1].values
y_train = train['Biopsy'].values
x_test = test.iloc[:,:-1].values
y_test = test['Biopsy'].values

#Standardizing the Data
from sklearn.preprocessing import StandardScaler

ss = StandardScaler() 
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#Convert From np Arrays Back to Dataframes
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

#################
#FEATURE SELECTION
#################
print('The data has been loaded and processed.\n')
from sklearn.linear_model import LogisticRegression

#Lasso i.e. l1 Regularization
lr_fs = LogisticRegression(solver='liblinear', penalty = 'l1')
lr_fs.fit(x_train, y_train)

#Get Coefficients 
all_features = lr_fs.coef_[0,:].tolist()

index = -1
for feature in all_features:

    if feature == 0: #removes predictor from training set if coefficient is 0
        index = all_features.index(feature, index+1)
        x_train.drop([index], axis = 1, inplace = True)
        x_test.drop([index], axis = 1, inplace = True)
    else:
        next

#Print Out the Best Features
selected_features = []
for col in x_train.columns:
    selected_features.append(train.columns[col])
    
print('The best features are: ' + ', '.join(selected_features) + '.')

#convert Dataframe to Array for Model Training
x_train = x_train.values
        
#################
#LOGISTIC REGRESSION
#################
print('\nLOGISTIC REGRESSION')

#Grid Search
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

CV_acc = 0
best_c = -1000
C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
kfolds = StratifiedKFold(n_splits = 10, shuffle = True)

#Testing Different Values for C
for c in C:
    predicted = []
    lr_gs = LogisticRegression(solver = 'liblinear', C = c)
    
    #10-Fold Cross Validation
    for training, testing in kfolds.split(x_train, y_train):
        lr_gs.fit(x_train[training], y_train[training])
        pred = lr_gs.predict(x_train[testing])
        
        for p in pred:
            predicted.append(p)
    
    score = accuracy_score(y_train, predicted)
    
    if score > CV_acc:
        CV_acc = score
        best_c = c     

print('The best C is ' + str(best_c) + '.')
print('The training accuracy is: ' + str(CV_acc) + '.')

#Test Set Performance
from sklearn.metrics import confusion_matrix

final_lr = LogisticRegression(solver = 'liblinear', C = best_c)
final_lr.fit(x_train, y_train)
lr_predicted = final_lr.predict(x_test)
lr_acc = accuracy_score(y_test, lr_predicted)

print('The test accuracy is: ' + str(lr_acc) + '.')
print(confusion_matrix(y_test, lr_predicted))

#################
#SUPPORT VECTOR MACHINE
#################
print('\nSUPPORT VECTOR MACHINE')
from sklearn.svm import SVC

#Grid Search
CV_acc = 0
best_c = -1000
for c in C:
    predicted = []
    svm_gs = SVC(gamma = 'auto', C = c)
    
    #10-Fold Cross Validation
    for training, testing in kfolds.split(x_train, y_train):
        svm_gs.fit(x_train[training], y_train[training])
        pred = svm_gs.predict(x_train[testing])
        
        for p in pred:
            predicted.append(p)
    
    score = accuracy_score(y_train, predicted)
    
    if score > CV_acc:
        CV_acc = score
        best_c = c     
    
print('The best C is ' + str(best_c) + '.')
print('The training accuracy is: ' + str(CV_acc) + '.')

#Test Set Performance
final_svm = SVC(gamma = 'auto', C = best_c)
final_svm.fit(x_train, y_train)
svm_predicted = final_svm.predict(x_test)
svm_acc = accuracy_score(y_test, svm_predicted)
print('The test accuracy is: ' + str(svm_acc) + '.')
print(confusion_matrix(y_test, svm_predicted))

#################
#DECISION TREE
#################
print('\nCLASSIFICATION DECISION TREE')
from sklearn.tree import DecisionTreeClassifier

#Train Classifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

#Test Set Performance
tree_predicted = tree.predict(x_test)
tree_acc = accuracy_score(y_test, tree_predicted)
print('The test accuracy is: ' + str(tree_acc) + '.')
print(confusion_matrix(y_test, tree_predicted))

#################
#BRANDOM FOREST
#################
print('\nRANDOM FOREST')
from sklearn.ensemble import RandomForestClassifier

#Train Classifer
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train, y_train)

#Test Set Performance
rf_predicted = rf.predict(x_test)
rf_acc = accuracy_score(y_test, rf_predicted)
print('The test accuracy is: ' + str(rf_acc) + '.')
print(confusion_matrix(y_test, rf_predicted))

#################
#Building KNN
#################
print('\nK NEAREST NEIGHBORS')
from sklearn.neighbors import KNeighborsClassifier

#Train Classifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

#Test Set Performance
predicted = knn.predict(x_test)
knn_acc = accuracy_score(y_test, predicted)
print('The test accuracy is: ' + str(knn_acc) + '.')
print(confusion_matrix(y_test, predicted))

#################
#Building LDA
#################
print('\nLINEAR DISCRIMINANT ANALYSIS')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Grid Search
CV_acc = 0
best_s = -1000
shrinkage = np.linspace(0.0, 1.0, 20)
for s in shrinkage:
    predicted = []
    lda_gs = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = s)
    
    #10-fold cross validation
    for training, testing in kfolds.split(x_train, y_train):
        lda_gs.fit(x_train[training], y_train[training])
        pred = lda_gs.predict(x_train[testing])
        
        for p in pred:
            predicted.append(p)
    
    score = accuracy_score(y_train, predicted)
    
    if score > CV_acc:
        CV_acc = score
        best_s = s    

print('The best shrinkage parameter is ' + str(best_s) + '.')
print('The training accuracy is: ' + str(CV_acc) + '.')

#Test Set Performance
final_lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = best_s)
final_lda.fit(x_train, y_train)
lda_predicted = final_lda.predict(x_test)
lda_acc = accuracy_score(y_test, lda_predicted)
print('The test accuracy is: ' + str(lda_acc) + '.')
print(confusion_matrix(y_test, lda_predicted))
