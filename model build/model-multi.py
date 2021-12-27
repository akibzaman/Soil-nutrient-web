# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:32:28 2021

@author: Akib Zaman

"""
from matplotlib import pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt
#from datetime import datetime, timedelta
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy import sparse

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import multioutput
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns
import pickle
from collections import Counter
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE

# from boruta import BorutaPy
all_features=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilType','LandClass',
              'Phosphorus','Potassium','Boron','Calcium','Magnesium','Manganese']
dataset = pd.read_csv("data/clean_data_labeled.csv", header=None, names=all_features)
dataset = dataset.drop(labels=0, axis=0)
dataset=dataset.reset_index()
dataset=dataset.drop(['index'], axis=1)
train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilType','LandClass']
train_string_feature=['Area','SoilType','LandClass']
test_feature=['Phosphorus','Potassium','Boron',
                       'Calcium','Magnesium','Manganese']

labelencoder = LabelEncoder()
for rx in train_string_feature:    
    dataset[rx]= labelencoder.fit_transform(dataset[rx])
    
# labelencoder = LabelEncoder()
for rx in test_feature:    
    dataset[rx]= labelencoder.fit_transform(dataset[rx])
    
for x in range(len(train_feature)):
    #df['Time'] = pd.to_numeric(df['Time'],errors='coerce')
    #df['Heartrate'] = pd.to_numeric(df['Heartrate'],errors='coerce')
    dataset[train_feature[x]] = pd.to_numeric(dataset[train_feature[x]],
           errors='coerce')


X1 = dataset[train_feature] # Features
X2 = dataset[train_feature] # Features
X3 = dataset[train_feature] # Features
X4 = dataset[train_feature] # Features
X5 = dataset[train_feature] # Features
X6 = dataset[train_feature] # Features
y = dataset[test_feature]# Target variable
y1= dataset['Phosphorus']
y2=dataset['Potassium']
y3=dataset['Boron']
y4= dataset['Calcium']
y5=dataset['Magnesium']
y6=dataset['Manganese']


counter = Counter(y1)
for k,v in counter.items():
	per = v / len(y1) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X1, y1 = oversample.fit_resample(X1, y1)
# summarize distribution
counter = Counter(y1)
for k,v in counter.items():
	per = v / len(y1) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


counter = Counter(y2)
for k,v in counter.items():
	per = v / len(y2) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X2, y2 = oversample.fit_resample(X2, y2)
# summarize distribution
counter = Counter(y2)
for k,v in counter.items():
	per = v / len(y2) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


counter = Counter(y3)
for k,v in counter.items():
	per = v / len(y3) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X3, y3 = oversample.fit_resample(X3, y3)
# summarize distribution
counter = Counter(y3)
for k,v in counter.items():
	per = v / len(y3) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


counter = Counter(y4)
for k,v in counter.items():
	per = v / len(y4) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X4, y4 = oversample.fit_resample(X4, y4)
# summarize distribution
counter = Counter(y4)
for k,v in counter.items():
	per = v / len(y4) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


counter = Counter(y5)
for k,v in counter.items():
	per = v / len(y5) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X5, y5 = oversample.fit_resample(X5, y5)
# summarize distribution
counter = Counter(y5)
for k,v in counter.items():
	per = v / len(y5) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


counter = Counter(y6)
for k,v in counter.items():
	per = v / len(y6) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
oversample = SMOTE()
X6, y6 = oversample.fit_resample(X6, y6)
# summarize distribution
counter = Counter(y6)
for k,v in counter.items():
	per = v / len(y6) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()



# X=X.to_numpy()
# y=y.to_numpy()
# LabelBinarizer().fit_transform(y)
# Split dataset into training set and test set
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0) 
print(X1_train.shape, y1_train.shape)
print(X1_test.shape, y1_test.shape)


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0) 
print(X2_train.shape, y2_train.shape)
print(X2_test.shape, y2_test.shape)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0) 
print(X3_train.shape, y3_train.shape)
print(X3_test.shape, y3_test.shape)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=0) 
print(X4_train.shape, y4_train.shape)
print(X4_test.shape, y4_test.shape)

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.3, random_state=0) 
print(X5_train.shape, y5_train.shape)
print(X5_test.shape, y5_test.shape)

X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.3, random_state=0) 
print(X6_train.shape, y6_train.shape)
print(X6_test.shape, y6_test.shape)


#####RandonForest
model_RF = RandomForestClassifier(n_estimators = 100, random_state=30)
model_RF.fit(X1_train,y1_train)
y1_pred_train=model_RF.predict(X1_train)
y1_pred_test = model_RF.predict(X1_test)
print("===============> Phosphorus")
print(metrics.classification_report(y1_train, y1_pred_train))
print(metrics.classification_report(y1_test, y1_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_Ph.pkl','wb'))

model_RF.fit(X2_train,y2_train)
y2_pred_train=model_RF.predict(X2_train)
y2_pred_test = model_RF.predict(X2_test)
print("===============> Potassium")
print(metrics.classification_report(y2_train, y2_pred_train))
print(metrics.classification_report(y2_test, y2_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_K.pkl','wb'))

model_RF.fit(X3_train,y3_train)
y3_pred_train=model_RF.predict(X3_train)
y3_pred_test = model_RF.predict(X3_test)
print("===============> Boron")
print(metrics.classification_report(y3_train, y3_pred_train))
print(metrics.classification_report(y3_test, y3_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_B.pkl','wb'))

model_RF.fit(X4_train,y4_train)
y4_pred_train=model_RF.predict(X4_train)
y4_pred_test = model_RF.predict(X4_test)
print("===============> Calcium")
print(metrics.classification_report(y4_train, y4_pred_train))
print(metrics.classification_report(y4_test, y4_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_Ca.pkl','wb'))

model_RF.fit(X5_train,y5_train)
y5_pred_train=model_RF.predict(X5_train)
y5_pred_test = model_RF.predict(X5_test)
print("===============> Magnesium")
print(metrics.classification_report(y5_train, y5_pred_train))
print(metrics.classification_report(y5_test, y5_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_Mg.pkl','wb'))


model_RF.fit(X6_train,y6_train)
y6_pred_train=model_RF.predict(X6_train)
y6_pred_test = model_RF.predict(X6_test)
print("===============> Manganese")
print(metrics.classification_report(y6_train, y6_pred_train))
print(metrics.classification_report(y6_test, y6_pred_test))
pickle.dump(model_RF, open('trained_model\model_rf_Mn.pkl','wb'))


####Evaluation with Random Sample
model1= pickle.load(open('trained_model\model_rf_Ph.pkl','rb'))
model2= pickle.load(open('trained_model\model_rf_K.pkl','rb'))
model3= pickle.load(open('trained_model\model_rf_B.pkl','rb'))
model4= pickle.load(open('trained_model\model_rf_Ca.pkl','rb'))
model5= pickle.load(open('trained_model\model_rf_Mg.pkl','rb'))
model6= pickle.load(open('trained_model\model_rf_Mn.pkl','rb'))
test_case=[[0.14,0.14,0.14,0.14,0.14,0.14,0,2,3]]
test_case=pd.DataFrame(test_case, columns = train_feature )
Ph_pred = model1.predict(test_case)
K_pred = model2.predict(test_case)
B_pred = model3.predict(test_case)
Ca_pred = model4.predict(test_case)
Mg_pred = model5.predict(test_case)
Mn_pred = model6.predict(test_case)
print("{0}, {1}, {2}, {3}, {4}, {5} ".format(Ph_pred,K_pred,B_pred,Ca_pred,Mg_pred,Mn_pred))


