# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:32:28 2021

@author: Hp
"""

from matplotlib import pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt
#from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier

from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns
import pickle
# from boruta import BorutaPy
all_features=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilTeam','LandClass',
              'Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']
dataset = pd.read_csv("data/clean_data_labeled.csv", header=None, names=all_features)
dataset = dataset.drop(labels=0, axis=0)
dataset=dataset.reset_index()
dataset=dataset.drop(['index'], axis=1)
train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7']
                #,'Area','SoilTeam','LandClass']
test_feature=['Phosphorus','Potassium','Boron',
                       'Calcium','Magnesium','Manganese']


labelencoder = LabelEncoder()
for rx in test_feature:    
    dataset[rx]= labelencoder.fit_transform(dataset[rx])
    
for x in range(len(train_feature)):
    #df['Time'] = pd.to_numeric(df['Time'],errors='coerce')
    #df['Heartrate'] = pd.to_numeric(df['Heartrate'],errors='coerce')
    dataset[train_feature[x]] = pd.to_numeric(dataset[train_feature[x]],
           errors='coerce')


X = dataset[train_feature] # Features
y = dataset[test_feature]# Target variable


X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.3,random_state=42)

print(X_train.shape)
print(X_test.shape)

# #######Building The Model (Estimator + Multilabel Estimator)
# import skmultilearn
# #dir(skmultilearn)

# from skmultilearn.problem_transform import BinaryRelevance
# from skmultilearn.problem_transform import ClassifierChain
# from skmultilearn.problem_transform import LabelPowerset
# from skmultilearn.adapt import MLkNN

# binary_rel_clf = BinaryRelevance(MultinomialNB())
# binary_rel_clf.fit(X_train,y_train)
# br_predictions = binary_rel_clf.predict(X_test)
# br_predictions.toarray()


# ###KNN
# knn_clf=KNeighborsClassifier();
# knn_clf.fit(X_train,y_train)
# y_pred = knn_clf.predict(X_test)

#####RandonForest
model_RF = RandomForestClassifier(n_estimators = 100, random_state=30)
model_RF.fit(X_train,y_train)
y_pred=model_RF.predict(X_test)




def performance(a,b):
    compare=a.compare(b)
    compare.columns=['Phosphorus','Phosphorus-pred','Potassium','Phosphorus-pred','Boron','Boron-pred',
                       'Calcium','Calcium-pred', 'Magnesium','Magnesium-pred',
                       'Manganese','Manganese-pred']
    print(compare['Phosphorus'].isna().sum(),compare['Potassium'].isna().sum(),
          compare['Boron'].isna().sum(),compare['Calcium'].isna().sum(),
          compare['Magnesium'].isna().sum(),compare['Manganese'].isna().sum())
    acc_ph=(compare['Phosphorus'].isna().sum()/401)*100
    acc_po=(compare['Potassium'].isna().sum()/401)*100
    acc_bo=(compare['Boron'].isna().sum()/401)*100
    acc_ca=(compare['Calcium'].isna().sum()/401)*100
    acc_mg=(compare['Magnesium'].isna().sum()/401)*100
    acc_mn=(compare['Manganese'].isna().sum()/401)*100 
    print("{0:.3f}%, {1:.3f}%, {2:.3f}%, {3:.3f}%, {4:.3f}%, {5:.3f}% ".format(acc_ph,acc_po,acc_bo,
                                                                               acc_ca,acc_mg,acc_mn))

    return compare
    
y_pred = pd.DataFrame(y_pred, columns = test_feature )
y_test=y_test.reset_index()
y_test=y_test.drop(['index'], axis=1)
compare=performance(y_test,y_pred)
a=[[0.84,0.84,0.84,0.84,0.84,0.84]]
a=pd.DataFrame(a, columns = test_feature )
a_bar=model_RF.predict(a)
  
# compare['Phosphorus'].isna().sum()
# compare['Potassium'].isna().sum()
# compare['Boron'].isna().sum()
# compare['Calcium'].isna().sum()
# compare['Magnesium'].isna().sum()
# compare['Manganese'].isna().sum()
#,error_1,error_2,error_3,error_4,error_5,error_6


















