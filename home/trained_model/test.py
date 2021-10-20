import pandas as pd
import pickle

train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilType','LandClass']

base_path = "trained_model" #add your drive model folder path here
####Evaluation with Random Sample
model1= pickle.load(open(base_path+'/model_rf_Ph.pkl','rb'))
model2= pickle.load(open(base_path+'/model_rf_K.pkl','rb'))
model3= pickle.load(open(base_path+'/model_rf_B.pkl','rb'))
model4= pickle.load(open(base_path+'/model_rf_Ca.pkl','rb'))
model5= pickle.load(open(base_path+'/model_rf_Mg.pkl','rb'))
model6= pickle.load(open(base_path+'/model_rf_Mn.pkl','rb'))

test_case=[[0.14,0.14,0.14,0.14,0.14,0.14,0,2,2]]
test_case=pd.DataFrame(test_case, columns = train_feature )
# print(test_case)
Ph_pred = model1.predict(test_case)
K_pred = model2.predict(test_case)
B_pred = model3.predict(test_case)
Ca_pred = model4.predict(test_case)
Mg_pred = model5.predict(test_case)
Mn_pred = model6.predict(test_case)
print("{0}, {1}, {2}, {3}, {4}, {5} ".format(Ph_pred,K_pred,B_pred,Ca_pred,Mg_pred,Mn_pred))