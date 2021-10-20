# ======================================================================
# model prediction start here

import pandas as pd  
import numpy as np
import os

import random

# file_path = 'dataset\Dataset_Update.csv'
# data = pd.read_csv(file_path)

# # data.head()

# # set all the group data globaly
# group_soil = data.groupby(['SoilTeam']).size().reset_index(name='count')
# group_soil = group_soil['SoilTeam'][:].values.tolist()

# group_land = data.groupby(['LandClass']).size().reset_index(name='count')
# group_land = group_land['LandClass'][:].values.tolist()

group_soil =["Ethel","EthelLoam","Loam","Sandy","SandyLoam"]
group_land =["DMEL","HL","LL","MEL","MLL","SMEL"]

# ============================================================================
# views start here

from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import render

# Create your views here.

# index page views
def indexPage(request):
    page_info = 1
    return render(request, 'index.html', {'page_info':page_info})


# prediction model page views
def predictionPage(request):
    page_info = 2
    return render(request, 'prediction.html', {'page_info':page_info, 'group_soil':group_soil, 'group_land':group_land})


# team page views
def teamPage(request):
    page_info = 3
    return render(request, 'team.html', {'page_info':page_info})


# test page views
def reflectancePage(request):
    page_info = 4
    return render(request, 'reflectance.html', {'page_info':page_info})


# =============================================================================================
# reflectance
#os.path.join(BASE_DIR, 'reflectance')
base_path_rf = 'home/reflectance'

data_rf2 = pd.read_csv(base_path_rf+'/band2.csv')
data_rf3 = pd.read_csv(base_path_rf+'/band3.csv')
data_rf4 = pd.read_csv(base_path_rf+'/band4.csv')
data_rf5 = pd.read_csv(base_path_rf+'/band5.csv')
data_rf6 = pd.read_csv(base_path_rf+'/band6.csv')
data_rf7 = pd.read_csv(base_path_rf+'/band7.csv')

data_list2 = data_rf2.values.tolist()
data_list3 = data_rf3.values.tolist()
data_list4 = data_rf4.values.tolist()
data_list5 = data_rf5.values.tolist()
data_list6 = data_rf6.values.tolist()
data_list7 = data_rf7.values.tolist()

def submitReflectancePage(request):
    if request.method == 'POST':
        From = request.POST['From']
        To = request.POST['To']

        row = len(data_list2)
        # print(row)
        col = len(data_list2[0])
        # print(col)

        #from input a = lower, b = higer range value.
        a = float(From)
        b = float(To)

        # print(a,"\t" , b)

        send_list2 = []
        send_list3 = []
        send_list4 = []
        send_list5 = []
        send_list6 = []
        send_list7 = []

        random_list = [a, b]
        if a < b:
            count = int(b*100) - int(a*100) - 1
            max_count = 18
            if max_count < count:
                count = max_count
            # print(count)
            while count != 0:
                r_val = round(random.uniform(a, b), 2)
                if r_val not in random_list:
                    random_list.append(r_val)
                    count -= 1
            
            random_list.sort()
            # print(random_list)
            
            for x in random_list:
                # for band2
                if x in data_list2[0]:
                    index_col2 = data_list2[0].index(x)
                    temp_list2 = [x]
                    for i in range(1, row):
                        temp_list2.append(round(data_list2[i][index_col2], 7))
                    send_list2.append(temp_list2)

                # for band3
                if x in data_list3[0]:
                    index_col3 = data_list3[0].index(x)
                    temp_list3 = [x]
                    for i in range(1, row):
                        temp_list3.append(round(data_list3[i][index_col3], 7))
                    send_list3.append(temp_list3)

                # for band4
                if x in data_list4[0]:
                    index_col4 = data_list4[0].index(x)
                    temp_list4 = [x]
                    for i in range(1, row):
                        temp_list4.append(round(data_list4[i][index_col4], 7))
                    send_list4.append(temp_list4)

                # for band5
                if x in data_list5[0]:
                    index_col5 = data_list5[0].index(x)
                    temp_list5 = [x]
                    for i in range(1, row):
                        temp_list5.append(round(data_list5[i][index_col5], 7))
                    send_list5.append(temp_list5)

                # for band6
                if x in data_list6[0]:
                    index_col6 = data_list6[0].index(x)
                    temp_list6 = [x]
                    for i in range(1, row):
                        temp_list6.append(round(data_list6[i][index_col6], 7))
                    send_list6.append(temp_list6)

                # for band7
                if x in data_list7[0]:
                    index_col7 = data_list7[0].index(x)
                    temp_list7 = [x]
                    for i in range(1, row):
                        temp_list7.append(round(data_list7[i][index_col7], 7))
                    send_list7.append(temp_list7)
                
            # print(random_list)
            # print(send_list)
            # return JsonResponse({"error": ""}, status=400)

            return JsonResponse({"band2":send_list2, "band3":send_list3, "band4":send_list4, "band4":send_list4, "band5":send_list5, "band6":send_list6, "band7":send_list7}, status=200)
        else:
            return JsonResponse({"error": ""}, status=400)
    return JsonResponse({"error": ""}, status=400)



# ============================================================================
# save request data globaly
pre_request_list = []
pre_send_list = []

# submit test views
def submitTestPage(request):
    if request.method == 'POST':
        B2Reft2 = float(request.POST['B2Reft2'])
        B2Reft3 = float(request.POST['B2Reft3'])
        B2Reft4 = float(request.POST['B2Reft4'])
        B2Reft5 = float(request.POST['B2Reft5'])
        B2Reft6 = float(request.POST['B2Reft6'])
        B2Reft7 = float(request.POST['B2Reft7'])

        SoilTeam = int(request.POST['SoilTeam'])
        LandClass = int(request.POST['LandClass'])


        test_list = [[B2Reft2, B2Reft3, B2Reft4, B2Reft5, B2Reft6, B2Reft7, SoilTeam, LandClass ]]
        request_list = [B2Reft2, B2Reft3, B2Reft4, B2Reft5, B2Reft6, B2Reft7, SoilTeam, LandClass]

        global pre_request_list
        global pre_send_list

        # print(pre_request_list, "\n", pre_send_list)
        if pre_request_list == request_list:
            return JsonResponse({"data":pre_send_list}, status=200)
        else:
            pre_request_list = request_list.copy()

            # test data
            train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','SoilType','LandClass']  
            x_test = pd.DataFrame(test_list, columns =train_feature)

            # print(x_test)

            y_pred_list = update_model_hybrid(x_test)
    
            # # print(send_data)
            feature_name_lsit = ['Phosphorus', 'Potassium', 'Boron', 'Calcium', 'Magnesium', 'Manganese']
            length_f = len(y_pred_list) 
            send_data = []

            # random.seed(15)  # use seed for same random number
            for i in range(0, length_f):
                feature_name = feature_name_lsit[i]
                x = y_pred_list[i]
                percntage = 0
                bar_type = ""
                if x == 2:
                    percntage = random.uniform(75, 100)
                    bar_type = 'cssProgress-success'
                elif x == 1 :
                    percntage = random.uniform(45, 60)
                    bar_type = 'cssProgress-warning'
                else:
                    percntage = random.uniform(15, 25)
                    bar_type = 'cssProgress-danger'
                percntage = round(percntage, 2)
                send_data.append([feature_name, percntage, bar_type, percntage])
            
            pre_send_list = send_data.copy()
            return JsonResponse({"data":send_data}, status=200)

    return JsonResponse({"error": ""}, status=400)


##=========================================================================
import pickle
def update_model(test_case):

    # train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilTeam','LandClass']

    base_path = "trained_model" #add your drive model folder path here
    ####Evaluation with Random Sample
    model1= pickle.load(open(base_path+'/model_rf_Ph.pkl','rb'))
    model2= pickle.load(open(base_path+'/model_rf_K.pkl','rb'))
    model3= pickle.load(open(base_path+'/model_rf_B.pkl','rb'))
    model4= pickle.load(open(base_path+'/model_rf_Ca.pkl','rb'))
    model5= pickle.load(open(base_path+'/model_rf_Mg.pkl','rb'))
    model6= pickle.load(open(base_path+'/model_rf_Mn.pkl','rb'))

    # test_case=[[0.14,0.14,0.14,0.14,0.14,0.14,0,2,2]]
    # test_case=pd.DataFrame(test_case, columns = train_feature )
    # print(test_case)
    Ph_pred = model1.predict(test_case)
    K_pred = model2.predict(test_case)
    B_pred = model3.predict(test_case)
    Ca_pred = model4.predict(test_case)
    Mg_pred = model5.predict(test_case)
    Mn_pred = model6.predict(test_case)
    print([Ph_pred[0],K_pred[0],B_pred[0],Ca_pred[0],Mg_pred[0],Mn_pred[0]])
    return [Ph_pred[0],K_pred[0],B_pred[0],Ca_pred[0],Mg_pred[0],Mn_pred[0]]

def update_model_hybrid(test_case):
    # train_feature=['B2Reft2','B3Reft3','B4Reft4','B5Reft5','B6Reft6','B7Reft7','Area','SoilTeam','LandClass']

    base_path = "hybrid_model" #add your drive model folder path here
    ####Evaluation with Random Sample
    model1= pickle.load(open(base_path+'/Phosphorus-hybrid.pkl','rb'))
    model2= pickle.load(open(base_path+'/Potassium-hybrid.pkl','rb'))
    model3= pickle.load(open(base_path+'/Boron-hybrid.pkl','rb'))
    model4= pickle.load(open(base_path+'/Calcium-hybrid.pkl','rb'))
    model5= pickle.load(open(base_path+'/Manesium-hybrid.pkl','rb'))
    model6= pickle.load(open(base_path+'/Manganese-hybrid.pkl','rb'))

    # test_case=[[0.14,0.14,0.14,0.14,0.14,0.14,0,2,2]]
    # test_case=pd.DataFrame(test_case, columns = train_feature )
    # print(test_case)
    Ph_pred = model1.predict(test_case)
    K_pred = model2.predict(test_case)
    B_pred = model3.predict(test_case)
    Ca_pred = model4.predict(test_case)
    Mg_pred = model5.predict(test_case)
    Mn_pred = model6.predict(test_case)
    return [Ph_pred[0],K_pred[0],B_pred[0],Ca_pred[0],Mg_pred[0],Mn_pred[0]]



