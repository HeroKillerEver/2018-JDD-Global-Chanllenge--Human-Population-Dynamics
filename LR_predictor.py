
# coding: utf-8

# In[1]:

import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
import pickle
import copy
def RMSLE(Y_TRUE,Y_PREDICT):
    Y_PREDICT=Y_PREDICT.flatten()
    Y_TRUE=Y_TRUE.flatten()
    sum=0.0
    for x in range(len(Y_PREDICT)):
        p = np.log(Y_PREDICT[x]+1)
        r = np.log(Y_TRUE[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(Y_PREDICT))**0.5
def Find_district(district):
    for i in range(len(DISTRICTS)):
        if district==DISTRICTS[i]:
            return i
df_flow = pd.read_csv('./DATA/flow_train.csv')


# In[ ]:




# In[2]:

##########################################################################################################################################
##########################################################################################################################################
# This cell search for 
# 1. best length of history and 
# 2. best length of future (hyper parameters) 
# for prediction of the 5 missing days in the local validation set
##########################################################################################################################################
##########################################################################################################################################
print("Constructing LR1...")
print("Grid Search for [length of history, length of future]...")
DISTRICTS = pickle.load( open( "DISTRICTS.pkl", "rb" ) )
CITY_OF_DISTRICTS = pickle.load( open( "CITY_OF_DISTRICTS.pkl", "rb" ) )

avaliable_dict = []
for LENGTH_OF_HISTORY in [7,8,9,10]:
    for LENGTH_OF_FUTURE in [6,7,8,9,10]:
        avaliable_dict.append([LENGTH_OF_HISTORY,LENGTH_OF_FUTURE])
MIN = 100
BEST_HYPER = [5,5]

LENGTH_OF_P = 5
LENGTH_OF_VAL = 2
START_PERIOD = 0
for search in range(len(avaliable_dict)):
    LENGTH_OF_HISTORY = avaliable_dict[search][0]
    LENGTH_OF_FUTURE = avaliable_dict[search][1]
    TRAIN_RANGE1 = 88 - LENGTH_OF_HISTORY - LENGTH_OF_P - START_PERIOD # - LENGTH_OF_FUTURE
    TRAIN_RANGE2 = 73 - LENGTH_OF_HISTORY - LENGTH_OF_P                         - LENGTH_OF_FUTURE - LENGTH_OF_P - (LENGTH_OF_VAL-1) # validation mode. For submission mode, 73 - 26
    LENGTH_OF_TRAINING = TRAIN_RANGE1 + TRAIN_RANGE2
    
    X_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
    X_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_P] )
    X_predict = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
    for i in range(204):
        for k in range(3):
            my_array = np.array(np.array(df_flow[df_flow["district_code"]==DISTRICTS[i]])[:,3+k].tolist())
            for j in range(TRAIN_RANGE1):
                X_data[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[START_PERIOD+j:                             START_PERIOD+j+LENGTH_OF_HISTORY]            
                X_data[i,j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                             START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
                Y_data[i,j,k,:] =                     my_array[START_PERIOD+j+LENGTH_OF_HISTORY:                             START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P]            
            ####################################################################################
            for j in range(TRAIN_RANGE2):
                X_data[i,TRAIN_RANGE1+j,k,0:LENGTH_OF_HISTORY] =                     my_array[88+j:                             88+j+LENGTH_OF_HISTORY]            
                X_data[i,TRAIN_RANGE1+j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[88+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                             88+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
                Y_data[i,TRAIN_RANGE1+j,k,:] =                     my_array[88+j+LENGTH_OF_HISTORY:                             88+j+LENGTH_OF_HISTORY+LENGTH_OF_P]
            ####################################################################################
            for j in range(LENGTH_OF_VAL):
                X_test[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[161-LENGTH_OF_FUTURE - LENGTH_OF_P -LENGTH_OF_HISTORY -(LENGTH_OF_VAL-j-1):                             161-LENGTH_OF_FUTURE - LENGTH_OF_P -(LENGTH_OF_VAL-j-1)]            
                X_test[i,j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[161-LENGTH_OF_FUTURE -(LENGTH_OF_VAL-j-1):                             161 -(LENGTH_OF_VAL-j-1)]            
                Y_test[i,j,k,:] =                     my_array[161-LENGTH_OF_FUTURE-LENGTH_OF_P -(LENGTH_OF_VAL-j-1):                             161-LENGTH_OF_FUTURE -(LENGTH_OF_VAL-j-1)]  
            X_predict[i,0,k,0:LENGTH_OF_HISTORY] =                 my_array[88 - LENGTH_OF_HISTORY:88]            
            X_predict[i,0,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                 my_array[88:88 + LENGTH_OF_FUTURE]            
    ################################################################################################
    reg = []
    for i in range(204):
        __=[]
        for k in range(3):
            _=[]
            for date in range(LENGTH_OF_P):
                _.append([])
            __.append(_)
        reg.append(__)
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                #reg[i][k][date]=LinearRegression().fit(X_data[i,:,:].reshape(LENGTH_OF_TRAINING,int(X_data[i,:,:].size/LENGTH_OF_TRAINING)), Y_data[i,:,k,date])
                reg[i][k][date]=LinearRegression().fit(X_data[i,:,k], Y_data[i,:,k,date])
    ################################################################################################
    Y_validate = Y_test*0.0
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                Y_validate[i,:,k,date]=reg[i][k][date].predict(X_test[i,:,k])
                #Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,:].reshape(1,int(X_test[i,:,:].size)))
    ################################################################################################
    #     print(Y_validate.shape)
    #     print(np.min(Y_validate))
    #     print(np.max(Y_validate))

    #     print(np.min(Y_test))
    #     print(np.max(Y_test))
    Y_validate=Y_validate.clip(0.0,1000000)


    print( [LENGTH_OF_HISTORY,LENGTH_OF_FUTURE],"RMSLE",RMSLE(Y_validate,Y_test))
    if RMSLE(Y_validate,Y_test)<MIN:
        MIN = RMSLE(Y_validate,Y_test)
        BEST_HYPER = [LENGTH_OF_HISTORY,LENGTH_OF_FUTURE]
print("Best [length of history, length of future]:",BEST_HYPER)
LENGTH_OF_HISTORY = BEST_HYPER[0]
LENGTH_OF_FUTURE = BEST_HYPER[1]


# In[3]:

##########################################################################################################################################
##########################################################################################################################################
# This cell use the former best length of history and best length of future (hyper parameters)to search for
# 1. the best starting point of training and
# 2. the best end point of training
# for prediction of the 5 missing days in the local validation set.
##########################################################################################################################################
##########################################################################################################################################
print("Grid Search for [start, end]...")
avaliable_dict = []
for START_PERIOD in [0,11,12,13,14]:
    for END in [0,5,10,12,14]:
        avaliable_dict.append([START_PERIOD,END])
MIN = 100
BEST_HYPER = [0,0]
LENGTH_OF_P = 5
LENGTH_OF_VAL = 5
for search in range(len(avaliable_dict)):    
    START_PERIOD = avaliable_dict[search][0] #-50
    END = avaliable_dict[search][1]
    TRAIN_RANGE1 = 88 - LENGTH_OF_HISTORY - LENGTH_OF_P - START_PERIOD
    TRAIN_RANGE2 = 73 - END - LENGTH_OF_HISTORY - LENGTH_OF_P                         - LENGTH_OF_FUTURE - LENGTH_OF_P - (LENGTH_OF_VAL-1) # validation mode. For submission mode, 73 - 26
    LENGTH_OF_TRAINING = TRAIN_RANGE1 + TRAIN_RANGE2
    
    X_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
    X_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_P] )
    X_predict = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
    for i in range(204):
        for k in range(3):
            my_array = np.array(np.array(df_flow[df_flow["district_code"]==DISTRICTS[i]])[:,3+k].tolist())
            for j in range(TRAIN_RANGE1):
                X_data[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[START_PERIOD+j:                             START_PERIOD+j+LENGTH_OF_HISTORY]            
                X_data[i,j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                             START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
                Y_data[i,j,k,:] =                     my_array[START_PERIOD+j+LENGTH_OF_HISTORY:                             START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P]            

            ####################################################################################
            for j in range(TRAIN_RANGE2):
                X_data[i,TRAIN_RANGE1+j,k,0:LENGTH_OF_HISTORY] =                     my_array[88+j:                             88+j+LENGTH_OF_HISTORY]            
                X_data[i,TRAIN_RANGE1+j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[88+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                             88+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
                Y_data[i,TRAIN_RANGE1+j,k,:] =                     my_array[88+j+LENGTH_OF_HISTORY:                             88+j+LENGTH_OF_HISTORY+LENGTH_OF_P]
            ####################################################################################
            for j in range(LENGTH_OF_VAL):
                X_test[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[161-LENGTH_OF_FUTURE - LENGTH_OF_P -LENGTH_OF_HISTORY -(LENGTH_OF_VAL-j-1):                             161-LENGTH_OF_FUTURE - LENGTH_OF_P -(LENGTH_OF_VAL-j-1)]            
                X_test[i,j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                     my_array[161-LENGTH_OF_FUTURE -(LENGTH_OF_VAL-j-1):                             161 -(LENGTH_OF_VAL-j-1)]            
                Y_test[i,j,k,:] =                     my_array[161-LENGTH_OF_FUTURE-LENGTH_OF_P -(LENGTH_OF_VAL-j-1):                             161-LENGTH_OF_FUTURE -(LENGTH_OF_VAL-j-1)]  
            X_predict[i,0,k,0:LENGTH_OF_HISTORY] =                 my_array[88 - LENGTH_OF_HISTORY:88]            
            X_predict[i,0,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                 my_array[88:88 + LENGTH_OF_FUTURE]            

    #     print("Shape of Xdata:",X_data.shape,
    #           "Shape of Ydata:",Y_data.shape,
    #           "Shape of Xtest:",X_test.shape,
    #           "Shape of Ytest:",Y_test.shape)
    ################################################################################################
    reg = []
    for i in range(204):
        __=[]
        for k in range(3):
            _=[]
            for date in range(LENGTH_OF_P):
                _.append([])
            __.append(_)
        reg.append(__)
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                #reg[i][k][date]=LinearRegression().fit(X_data[i,:,:].reshape(LENGTH_OF_TRAINING,int(X_data[i,:,:].size/LENGTH_OF_TRAINING)), Y_data[i,:,k,date])
                reg[i][k][date]=LinearRegression().fit(X_data[i,:,k], Y_data[i,:,k,date])
    ################################################################################################
    Y_validate = Y_test*0.0
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                Y_validate[i,:,k,date]=reg[i][k][date].predict(X_test[i,:,k])
                #Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,:].reshape(1,int(X_test[i,:,:].size)))
    ################################################################################################
    #     print(Y_validate.shape)
    #     print(np.min(Y_validate))
    #     print(np.max(Y_validate))

    #     print(np.min(Y_test))
    #     print(np.max(Y_test))
    Y_validate=Y_validate.clip(0.0,1000000)


    print( [START_PERIOD,END],"RMSLE",RMSLE(Y_validate,Y_test))
    if RMSLE(Y_validate,Y_test)<MIN:
        MIN = RMSLE(Y_validate,Y_test)
        BEST_HYPER = [START_PERIOD,END]
print("Best [start, end]:",BEST_HYPER)
START_PERIOD = BEST_HYPER[0]
END = BEST_HYPER[1]


# In[4]:

print("Summary of the grid search for LR1:")
print("Length of history:",LENGTH_OF_HISTORY)
print("Length of future:",LENGTH_OF_FUTURE)
print("Start:",START_PERIOD)
print("End:",END)
# 7，8，0，14


# In[5]:

##########################################################################################################################################
##########################################################################################################################################
# This cell use the former best hyper parameters for prediction of the 5 missing days.
##########################################################################################################################################
##########################################################################################################################################
print("LR1 Prediction...")
TRAIN_RANGE1 = 88 - LENGTH_OF_HISTORY - LENGTH_OF_P - START_PERIOD # - LENGTH_OF_FUTURE
TRAIN_RANGE2 = 73 - END - LENGTH_OF_HISTORY - LENGTH_OF_P - LENGTH_OF_FUTURE  # submission mode
LENGTH_OF_TRAINING = TRAIN_RANGE1 + TRAIN_RANGE2
LENGTH_OF_P = 5
X_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
X_test = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_test = np.zeros( [204,1,3,LENGTH_OF_P] )
X_predict = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
for i in range(204):
    for k in range(3):
        my_array = np.array(np.array(df_flow[df_flow["district_code"]==DISTRICTS[i]])[:,3+k].tolist())
        for j in range(TRAIN_RANGE1):
            X_data[i,j,k,0:LENGTH_OF_HISTORY] =                 my_array[START_PERIOD+j:                         START_PERIOD+j+LENGTH_OF_HISTORY]            
            X_data[i,j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                 my_array[START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                         START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
            Y_data[i,j,k,:] =                 my_array[START_PERIOD+j+LENGTH_OF_HISTORY:                         START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P]            

        ####################################################################################
        for j in range(TRAIN_RANGE2):
            X_data[i,TRAIN_RANGE1+j,k,0:LENGTH_OF_HISTORY] =                 my_array[88+j:                         88+j+LENGTH_OF_HISTORY]            
            X_data[i,TRAIN_RANGE1+j,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =                 my_array[88+j+LENGTH_OF_HISTORY+LENGTH_OF_P:                         88+j+LENGTH_OF_HISTORY+LENGTH_OF_P+LENGTH_OF_FUTURE]
            Y_data[i,TRAIN_RANGE1+j,k,:] =                 my_array[88+j+LENGTH_OF_HISTORY:                         88+j+LENGTH_OF_HISTORY+LENGTH_OF_P]
        ####################################################################################
        X_test[i,0,k,0:LENGTH_OF_HISTORY] =             my_array[161-LENGTH_OF_FUTURE - LENGTH_OF_P -LENGTH_OF_HISTORY:                     161-LENGTH_OF_FUTURE - LENGTH_OF_P]            
        X_test[i,0,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =             my_array[161-LENGTH_OF_FUTURE:                     161]            
        Y_test[i,0,k,:] =             my_array[161-LENGTH_OF_FUTURE-LENGTH_OF_P:                     161-LENGTH_OF_FUTURE]  
        X_predict[i,0,k,0:LENGTH_OF_HISTORY] =             my_array[88 - LENGTH_OF_HISTORY:88]            
        X_predict[i,0,k,LENGTH_OF_HISTORY:LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] =             my_array[88:88 + LENGTH_OF_FUTURE]            
################################################################################################
reg = []
for i in range(204):
    __=[]
    for k in range(3):
        _=[]
        for date in range(LENGTH_OF_P):
            _.append([])
        __.append(_)
    reg.append(__)
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            reg[i][k][date]=LinearRegression().fit(X_data[i,:,k], Y_data[i,:,k,date])
################################################################################################
# Y_validate = Y_test*0.0
# for i in range(204):
#     for k in range(3):
#         for date in range(LENGTH_OF_P):
#             Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,k])
# ################################################################################################
# print(Y_validate.shape)
# print(np.min(Y_validate))
# print(np.max(Y_validate))

# print(np.min(Y_test))
# print(np.max(Y_test))
# Y_validate=Y_validate.clip(0.0,1000000)
# print( [START_PERIOD,END],"RMSLE",RMSLE(Y_validate,Y_test))
################################################################################################
Y_predict = Y_test*0.0
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            Y_predict[i,0,k,date]=reg[i][k][date].predict(X_predict[i,:,k])
Y_predict = Y_predict.clip(0.0,1000000)
print("Summary of prediction of LR1, Min/Max/Mean:")
print(np.min(Y_predict))
print(np.max(Y_predict))
print(np.mean(Y_predict))
np.save("P5_FINAL",Y_predict)
################################################################################################
def amend_P5_to_flow(Y_predict,FileName="flow_FINAL"):
    flow = np.zeros([204,3,161+5])
    df_flow = pd.read_csv('./DATA/flow_train.csv')
    flow[:,:,88:88+5]=Y_predict[:,0,:,:]
    for i in range(204):
        for k in range(3):
            my_array = np.array(np.array(df_flow[df_flow["district_code"]==DISTRICTS[i]])[:,3+k].tolist())
            flow[i,k,0:88]=my_array[0:88]
            flow[i,k,88+5:161+5]=my_array[88:161]
    #print(flow)
    np.save(FileName,flow)
    return flow
flow = amend_P5_to_flow(Y_predict)
# print(flow.shape)
# for i in range(204):
#     plt.plot(flow[i,2])


# In[ ]:




# In[6]:

##########################################################################################################################################
##########################################################################################################################################
# This cell search for the best length of history for prediction of the last 10 days in the local validation set
##########################################################################################################################################
##########################################################################################################################################
print("Constructing LR2...")
print("Grid search for [length of history]:")
my_info = pickle.load( open( "SQAURE.pkl", "rb" ) )
avaliable_dict = []
for LENGTH_OF_HISTORY in [3,4,5,12,29]:
    avaliable_dict.append(LENGTH_OF_HISTORY)
MIN = 100
BEST_HYPER = [7]
LENGTH_OF_P = 10
LENGTH_OF_VAL = 5
for search in range(len(avaliable_dict)):
    LENGTH_OF_HISTORY = avaliable_dict[search]
    reg = []
    for i in range(204):
        __=[]
        for k in range(3):
            _=[]
            for date in range(LENGTH_OF_P):
                _.append([])
            __.append(_)
        reg.append(__)
    
    X_data = []
    Y_data = []
    X_test = []
    Y_test = []
    X_predict = []
    Y_predict = []
    
    for i in range(204):
        X_data.append([])
        Y_data.append([])
        X_test.append([])
        Y_test.append([])
        X_predict.append([])
        Y_predict.append([])
    
    for i in range(204):
        if len(my_info[i][0]["START"])==0:
            my_array = flow[i][:,:]
        elif len(my_info[i][0]["START"])==1:
            my_array = flow[i][:,0:my_info[i][0]["START"][0]]
            my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][0]:166]), axis=1)
        else:
            my_array = np.zeros([3,0])
            my_array = np.concatenate((my_array, flow[i][:,0:my_info[i][0]["START"][0]]), axis=1)
            for j in range(len(my_info[i][0]["START"])-1):
                my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][j]:my_info[i][0]["START"][j+1]]), axis=1)
            my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][j+1]:166]), axis=1)
        L = my_array.shape[1]
        TRAIN_RANGE1 = my_array.shape[1] - LENGTH_OF_HISTORY - LENGTH_OF_P - (LENGTH_OF_VAL-1)
        
        if TRAIN_RANGE1<10:
            #print(i,TRAIN_RANGE1,end=";")
            my_array = flow[i][:,:]
            L = my_array.shape[1]
            TRAIN_RANGE1 = my_array.shape[1] - LENGTH_OF_HISTORY - LENGTH_OF_P - (LENGTH_OF_VAL-1)
        
        
        LENGTH_OF_TRAINING = TRAIN_RANGE1
        
        X_data[i] = np.zeros( [LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY] )
        Y_data[i] = np.zeros( [LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
        X_test[i] = np.zeros( [LENGTH_OF_VAL,3,LENGTH_OF_HISTORY] )
        Y_test[i] = np.zeros( [LENGTH_OF_VAL,3,LENGTH_OF_P] )
        X_predict[i] = np.zeros( [1,3,LENGTH_OF_HISTORY] )
        Y_predict[i] = np.zeros( [1,3,LENGTH_OF_P] )
        
        for j in range(TRAIN_RANGE1):
            X_data[i][j,:,0:LENGTH_OF_HISTORY] = my_array[:,j:j+LENGTH_OF_HISTORY]      
            Y_data[i][j,:,:] = my_array[:,j+LENGTH_OF_HISTORY:j+LENGTH_OF_HISTORY+LENGTH_OF_P]  
        
        for j in range(LENGTH_OF_VAL):
            X_test[i][j,:,0:LENGTH_OF_HISTORY] = my_array[:,L  - LENGTH_OF_P - LENGTH_OF_HISTORY -(LENGTH_OF_VAL-j-1):                         L  - LENGTH_OF_P -(LENGTH_OF_VAL-j-1)]                      
            Y_test[i][j,:,:] = my_array[:,L-LENGTH_OF_P -(LENGTH_OF_VAL-j-1):                         L  -(LENGTH_OF_VAL-j-1)]  
        X_predict[i][0,:,0:LENGTH_OF_HISTORY] = my_array[:,L - LENGTH_OF_HISTORY:L]                      
        
        ################################################################################################

    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                reg[i][k][date]=LinearRegression().fit(X_data[i][:,k], Y_data[i][:,k,date])
    ################################################################################################
    Y_validate = copy.deepcopy(Y_test)
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                Y_validate[i][:,k,date]=reg[i][k][date].predict(X_test[i][:,k])
                #Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,:].reshape(1,int(X_test[i,:,:].size)))
    ################################################################################################
    Y_validate = np.array(Y_validate)
    Y_test = np.array(Y_test)
    #     print(Y_validate.shape)
    #     print(np.min(Y_validate))
    #     print(np.max(Y_validate))

    #     print(np.min(Y_test))
    #     print(np.max(Y_test))
    Y_validate=Y_validate.clip(0.0,1000000)


    print( [LENGTH_OF_HISTORY],"RMSLE",RMSLE(Y_validate,Y_test))
    if RMSLE(Y_validate,Y_test)<MIN:
        MIN = RMSLE(Y_validate,Y_test)
        BEST_HYPER = [LENGTH_OF_HISTORY]
print("Best [length of history]:",BEST_HYPER)


# In[ ]:




# In[7]:

##########################################################################################################################################
##########################################################################################################################################
# This cell use the best length of history for prediction of the last 10 days.
# The square wave data is discarded here (as outliers).
##########################################################################################################################################
##########################################################################################################################################
print("LR2 Prediction...")
LENGTH_OF_HISTORY = BEST_HYPER[0]
LENGTH_OF_P = 10
LENGTH_OF_VAL = 1

reg = []
for i in range(204):
    __=[]
    for k in range(3):
        _=[]
        for date in range(LENGTH_OF_P):
            _.append([])
        __.append(_)
    reg.append(__)

X_data = []
Y_data = []
X_test = []
Y_test = []
X_predict = []
Y_predict = []

for i in range(204):
    X_data.append([])
    Y_data.append([])
    X_test.append([])
    Y_test.append([])
    X_predict.append([])
    Y_predict.append([])

for i in range(204):
    if len(my_info[i][0]["START"])==0:
        my_array = flow[i][:,:]
    elif len(my_info[i][0]["START"])==1:
        my_array = flow[i][:,0:my_info[i][0]["START"][0]]
        my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][0]:166]), axis=1)
    else:
        my_array = np.zeros([3,0])
        my_array = np.concatenate((my_array, flow[i][:,0:my_info[i][0]["START"][0]]), axis=1)
        for j in range(len(my_info[i][0]["START"])-1):
            my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][j]:my_info[i][0]["START"][j+1]]), axis=1)
        my_array = np.concatenate((my_array, flow[i][:,my_info[i][0]["END"][j+1]:166]), axis=1)
    L = my_array.shape[1]
    TRAIN_RANGE1 = L - LENGTH_OF_HISTORY - LENGTH_OF_P

    if TRAIN_RANGE1<10:
        #print(i,TRAIN_RANGE1,end=";")
        my_array = flow[i][:,:]
        L = my_array.shape[1]
        TRAIN_RANGE1 = L - LENGTH_OF_HISTORY - LENGTH_OF_P


    LENGTH_OF_TRAINING = TRAIN_RANGE1

    X_data[i] = np.zeros( [LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY] )
    Y_data[i] = np.zeros( [LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
    X_test[i] = np.zeros( [LENGTH_OF_VAL,3,LENGTH_OF_HISTORY] )
    Y_test[i] = np.zeros( [LENGTH_OF_VAL,3,LENGTH_OF_P] )
    X_predict[i] = np.zeros( [1,3,LENGTH_OF_HISTORY] )
    Y_predict[i] = np.zeros( [1,3,LENGTH_OF_P] )

    for j in range(TRAIN_RANGE1):
        X_data[i][j,:,0:LENGTH_OF_HISTORY] = my_array[:,j:j+LENGTH_OF_HISTORY]      
        Y_data[i][j,:,:] = my_array[:,j+LENGTH_OF_HISTORY:j+LENGTH_OF_HISTORY+LENGTH_OF_P]  

    for j in range(LENGTH_OF_VAL):
        X_test[i][j,:,0:LENGTH_OF_HISTORY] = my_array[:,L  - LENGTH_OF_P - LENGTH_OF_HISTORY -(LENGTH_OF_VAL-j-1):                     L  - LENGTH_OF_P -(LENGTH_OF_VAL-j-1)]                      
        Y_test[i][j,:,:] = my_array[:,L-LENGTH_OF_P -(LENGTH_OF_VAL-j-1):                     L  -(LENGTH_OF_VAL-j-1)]  
    X_predict[i][0,:,0:LENGTH_OF_HISTORY] = my_array[:,L - LENGTH_OF_HISTORY:L]                      

    ################################################################################################

for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            reg[i][k][date]=LinearRegression().fit(X_data[i][:,k], Y_data[i][:,k,date])
################################################################################################
Y_validate = copy.deepcopy(Y_test)
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            Y_validate[i][:,k,date]=reg[i][k][date].predict(X_test[i][:,k])
################################################################################################
Y_validate = np.array(Y_validate)
Y_test = np.array(Y_test)
# print(Y_validate.shape)
# print(np.min(Y_validate))
# print(np.max(Y_validate))

# print(np.min(Y_test))
# print(np.max(Y_test))
Y_validate=Y_validate.clip(0.0,1000000)

print("RMSLE",RMSLE(Y_validate,Y_test))
################################################################################################
Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            Y_predict[i,0,k,date]=reg[i][k][date].predict(X_predict[i][:,k])
print("Summary of prediction of LR2, Min/Max/Mean:")
print(np.min(Y_predict))
print(np.max(Y_predict))
print(np.mean(Y_predict))
Y_predict = Y_predict.clip(0.0,10000000)
np.save("P10_FINAL",Y_predict)
flow_all = np.concatenate((flow,Y_predict[:,0,:,:]),axis=2)


# In[ ]:




# In[8]:

##########################################################################################################################################
##########################################################################################################################################
# This cell runs another linear model to predict the sudden rise (of the square waves) of some districts of the last 10 days.
# use the most future prediction minus the most recent prediction as the rise
##########################################################################################################################################
##########################################################################################################################################
print("Constructing LR3...")
print("Grid search for [length of history, start]:")
avaliable_dict = []
for LENGTH_OF_HISTORY in [3,4,9,15]:
    for START_PERIOD in [12,13,31,61]:
        avaliable_dict.append([LENGTH_OF_HISTORY,START_PERIOD])
MIN = 100
BEST_HYPER = [5,5]
for search in range(len(avaliable_dict)):
    LENGTH_OF_HISTORY = avaliable_dict[search][0]
    START_PERIOD = avaliable_dict[search][1]
    
    LENGTH_OF_FUTURE = 0
    LENGTH_OF_P = 10
    
    LENGTH_OF_VAL = 4
    
    TRAIN_RANGE1 = 161+5 - LENGTH_OF_HISTORY - LENGTH_OF_P - START_PERIOD - (LENGTH_OF_VAL-1)  # validation mode. For submission mode, 73 - 26
    TRAIN_RANGE2 = 0
    LENGTH_OF_TRAINING = TRAIN_RANGE1 + TRAIN_RANGE2
    
    X_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
    X_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_test = np.zeros( [204,LENGTH_OF_VAL,3,LENGTH_OF_P] )
    X_predict = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
    Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
    for i in range(204):
        for k in range(3):
            my_array = flow[i,k,:]
            for j in range(TRAIN_RANGE1):
                X_data[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[START_PERIOD+j:                             START_PERIOD+j+LENGTH_OF_HISTORY]            
                Y_data[i,j,k,:] =                     my_array[START_PERIOD+j+LENGTH_OF_HISTORY:                             START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P]            
            ####################################################################################
            for j in range(LENGTH_OF_VAL):
                X_test[i,j,k,0:LENGTH_OF_HISTORY] =                     my_array[161+5-LENGTH_OF_FUTURE - LENGTH_OF_P -LENGTH_OF_HISTORY -(LENGTH_OF_VAL-j-1):                             161+5-LENGTH_OF_FUTURE - LENGTH_OF_P -(LENGTH_OF_VAL-j-1)]                      
                Y_test[i,j,k,:] =                     my_array[161+5-LENGTH_OF_FUTURE-LENGTH_OF_P -(LENGTH_OF_VAL-j-1):                             161+5-LENGTH_OF_FUTURE -(LENGTH_OF_VAL-j-1)]  
            X_predict[i,0,k,0:LENGTH_OF_HISTORY] =                 my_array[161+5 - LENGTH_OF_HISTORY:161+5]                      
    ################################################################################################
    reg = []
    for i in range(204):
        __=[]
        for k in range(3):
            _=[]
            for date in range(LENGTH_OF_P):
                _.append([])
            __.append(_)
        reg.append(__)
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                #reg[i][k][date]=LinearRegression().fit(X_data[i,:,:].reshape(LENGTH_OF_TRAINING,int(X_data[i,:,:].size/LENGTH_OF_TRAINING)), Y_data[i,:,k,date])
                reg[i][k][date]=LinearRegression().fit(X_data[i,:,k], Y_data[i,:,k,date])
    ################################################################################################
    Y_validate = Y_test*0.0
    for i in range(204):
        for k in range(3):
            for date in range(LENGTH_OF_P):
                Y_validate[i,:,k,date]=reg[i][k][date].predict(X_test[i,:,k])
                #Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,:].reshape(1,int(X_test[i,:,:].size)))
    ################################################################################################
    #     print(Y_validate.shape)
    #     print(np.min(Y_validate))
    #     print(np.max(Y_validate))

    #     print(np.min(Y_test))
    #     print(np.max(Y_test))
    Y_validate=Y_validate.clip(0.0,1000000)


    print( [LENGTH_OF_HISTORY,START_PERIOD],"RMSLE",RMSLE(Y_validate,Y_test))
    if RMSLE(Y_validate,Y_test)<MIN:
        MIN = RMSLE(Y_validate,Y_test)
        BEST_HYPER = [LENGTH_OF_HISTORY,START_PERIOD]
print("Best [length of history, start]:",BEST_HYPER)
###########################################################################################################################################
###########################################################################################################################################
print("LR3 prediction...")
LENGTH_OF_HISTORY = BEST_HYPER[0]
START_PERIOD = BEST_HYPER[1]
LENGTH_OF_P = 10
LENGTH_OF_FUTURE = 0
TRAIN_RANGE1 = 161+5 - LENGTH_OF_HISTORY - LENGTH_OF_P - START_PERIOD # submission mode
TRAIN_RANGE2 = 0  
LENGTH_OF_TRAINING = TRAIN_RANGE1 + TRAIN_RANGE2
X_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_data = np.zeros( [204,LENGTH_OF_TRAINING,3,LENGTH_OF_P] )
X_test = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_test = np.zeros( [204,1,3,LENGTH_OF_P] )
X_predict = np.zeros( [204,1,3,LENGTH_OF_HISTORY+LENGTH_OF_FUTURE] )
Y_predict = np.zeros( [204,1,3,LENGTH_OF_P] )
for i in range(204):
    for k in range(3):
        my_array = flow[i,k,:]
        for j in range(TRAIN_RANGE1):
            X_data[i,j,k,0:LENGTH_OF_HISTORY] =                 my_array[START_PERIOD+j:                         START_PERIOD+j+LENGTH_OF_HISTORY]            
            Y_data[i,j,k,:] =                 my_array[START_PERIOD+j+LENGTH_OF_HISTORY:                         START_PERIOD+j+LENGTH_OF_HISTORY+LENGTH_OF_P]            
        ####################################################################################
        X_test[i,0,k,0:LENGTH_OF_HISTORY] =             my_array[161+5-LENGTH_OF_FUTURE - LENGTH_OF_P -LENGTH_OF_HISTORY:                     161+5-LENGTH_OF_FUTURE - LENGTH_OF_P]            
        Y_test[i,0,k,:] =             my_array[161+5-LENGTH_OF_FUTURE-LENGTH_OF_P:                     161+5-LENGTH_OF_FUTURE]  
        X_predict[i,0,k,0:LENGTH_OF_HISTORY] =             my_array[161+5 - LENGTH_OF_HISTORY:161+5]            
################################################################################################
reg = []
for i in range(204):
    __=[]
    for k in range(3):
        _=[]
        for date in range(LENGTH_OF_P):
            _.append([])
        __.append(_)
    reg.append(__)
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            #reg[i][k][date]=LinearRegression().fit(X_data[i,:,:].reshape(LENGTH_OF_TRAINING,int(X_data[i,:,:].size/LENGTH_OF_TRAINING)), Y_data[i,:,k,date])
            reg[i][k][date]=LinearRegression().fit(X_data[i,:,k], Y_data[i,:,k,date])
################################################################################################
Y_validate = Y_test*0.0
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,k])
            #Y_validate[i,0,k,date]=reg[i][k][date].predict(X_test[i,:,:].reshape(1,int(X_test[i,:,:].size)))
################################################################################################
print(Y_validate.shape)
print(np.min(Y_validate))
print(np.max(Y_validate))

print(np.min(Y_test))
print(np.max(Y_test))

Y_validate=Y_validate.clip(0.0,1000000)
print( [LENGTH_OF_HISTORY,START_PERIOD],"RMSLE",RMSLE(Y_validate,Y_test))
################################################################################################
Y_predict = Y_test*0.0
for i in range(204):
    for k in range(3):
        for date in range(LENGTH_OF_P):
            Y_predict[i,0,k,date]=reg[i][k][date].predict(X_predict[i,:,k])
print("Summary of prediction of LR3, Min/Max/Mean:")
print(np.min(Y_predict))
print(np.max(Y_predict))
print(np.mean(Y_predict))
Y_predict.clip(0.0,10000000)
np.save("P10_RISE_FINAL",Y_predict)


# In[ ]:




# In[9]:

##########################################################################################################################################
##########################################################################################################################################
# This cell predicts the sudden drop in the last 10 days if a square wave is observed before prediction.
# A tunable hyper parameter is the day of sudden drop
##########################################################################################################################################
##########################################################################################################################################
print("Post-processing phase I: Sudden drops.")
flow_all = np.concatenate((flow,np.load("P10_FINAL.npy")[:,0,:,:]),axis=2)
def plot_amend(a,b):
    plt.plot(flow_all[a,b,:])
    plt.plot([87.0,87.0001],[0,np.max(flow_all[a,b,:])])
    plt.plot([92.0,92.0001],[0,np.max(flow_all[a,b,:])])
    plt.plot([160.0+5,160.0001+5],[0,np.max(flow_all[a,b,:])])
    plt.show()
HYPER_DROP_START = 166
Y_predict_refined = np.load("P10_FINAL.npy")
for i in range(204):
    if 166 in my_info[i][0]["END"]:
#         print(i)
#         plot_amend(i,0)
        for k in range(3):
            Y_predict_refined[i,0,k,166-166:HYPER_DROP_START-166]+=flow_all[i,k,165]-flow_all[i,k,166]
np.save("P10_FINAL",Y_predict_refined)


# In[ ]:




# In[15]:

##########################################################################################################################################
##########################################################################################################################################
# This cell predicts the sudden rise (square wave) in the lasat 10 days if square wave patterns exist before prediction.
##########################################################################################################################################
##########################################################################################################################################
print("Post-processing phase II: Sudden rises.")
DISTRICT_RISE_BYLC=[ '1ee792b0e8a4692d66e60f8d7beee5fe',
                     '61662c2542a64a7a3abe5ca2ea9beb4c',
                     'b0233a782ed2ba05234081a25670ff5e',
                     'ce43671e2f14642ae700179f4974900f',
                     'd1a1265a24dbe06b8d7e95026a6f684b',
                     'eca573563bea4cdf078344745748ca1f',
                     '32b373c56a06dbee11da154380176f88',
                     '596321022e054f1eca5fcbe5059db05f',
                     'f14086632a735e50a4995b0933792892',
                     'a29daab64f058b3320ea5c87c678d3e9',
                     '2d7c97f86cbbfd9eb67da7cc989d8325',
                     'a800377324c381d866ad1ed0295d773d',
                     'cc475efb2b1965338158d419c6e5926a',
                     '0c43816e5bb9aebe39c75eaea3d0fc21',
                     'e253d54d713b71b5ac02eef62d52a6a1',
                     '7fd683ff9a9d7bb73bac7f86b0a81f15',
                     'b27bbc5718b26ff066190c23f35c9ac1',
                     'e3ec54d322bb97795df4920d3a2d0ceb',
                     '7d0f648c41bee96ab1a7f2df7d84904c',
                     '7f280b39e16f76ff3447e31ac9d957a8',
                     'cac3ac32bd3aabd2285460035f62bdfd',
                     '36a4c7f5bd890819bd6d2e3bc772ae92',
                     '81cce7cc849b099f9d2986d0d457f1db',
                     '99c45c93c535f54c9579b597297dabef',
                     'e24d4e84146e672f89ce72ab7dc1dbf7',
                     '5f5e648d22de4bc322a5d68b4c76091e',
                     '777d079b7a42595e6564c916c5a27074']
with open('DISTRICT_RISE_BYLC.pkl', 'wb') as f:
    pickle.dump(DISTRICT_RISE_BYLC, f)
DISTRICT_RISE_BYLC_index=[]
for i in DISTRICT_RISE_BYLC:
    DISTRICT_RISE_BYLC_index.append(Find_district(i))
print(DISTRICT_RISE_BYLC_index)

Y_5 = np.load("P5_FINAL.npy")
###############################################################################
Y_10_SQUAREC = np.load("P10_FINAL.npy")
Y_10_PLUS = np.load("P10_RISE_FINAL.npy")
###############################################################################
Y_10_SQUARECA = Y_10_SQUAREC*1.0
for i in DISTRICT_RISE_BYLC_index:
    for k in range(3):
        for d in range(10):
            Y_10_SQUARECA[i,0,k,d]=Y_10_PLUS[i,0,k,9]
flow_all = np.concatenate((flow,Y_10_SQUARECA[:,0,:,:]),axis=2)
###############################################################################
HYPER_RISE_FRACTION = 0.9027
for i in DISTRICT_RISE_BYLC_index:
    for k in range(3):
        for d in range(10):
            Y_10_SQUARECA[i,0,k,d] = Y_10_SQUARECA[i,0,k,d] +                                                 -(1-HYPER_RISE_FRACTION)*(Y_10_SQUARECA[i,0,k,d]-flow_all[i,k,165])
###############################################################################
Y_all_SQUAREC = np.concatenate((Y_5[:,0,:,:],Y_10_SQUAREC[:,0,:,:]),axis=2)     
Y_all_PLUS = np.concatenate((Y_5[:,0,:,:],Y_10_PLUS[:,0,:,:]),axis=2)     
Y_all_SQUARECA = np.concatenate((Y_5[:,0,:,:],Y_10_SQUARECA[:,0,:,:]),axis=2)   
print("RMSLE diff",end=":")
print( RMSLE(Y_all_SQUAREC,Y_all_SQUARECA)**2 - RMSLE(Y_all_PLUS,Y_all_SQUARECA)**2 )
flow_all = np.concatenate((flow,Y_10_SQUARECA[:,0,:,:]),axis=2)
###############################################################################
# for i in DISTRICT_RISE_BYLC_index:
#     if i in DISTRICT_RISE_BYLC_index:
#         #/plt.figure(dpi=300)
#         print(i,DISTRICTS[i],end=":")
#         print(my_info[i][0])
#         plt.plot(flow_all[i][2])
#         plt.plot(flow_all_1947P_LRRISE_END[i][2])
#         plt.show()     
###############################################################################
# for i in range(204):
#    print(i)
#    plot_amend(i,0)


# In[16]:

######################################################################################################
######################################################################################################
######################################################################################################
Y_5_merged = Y_5
print("Summary of prediction: (missing 5/last 10) Min/Max/Mean")
print(np.min(Y_5_merged))
print(np.max(Y_5_merged))
print(np.mean(Y_5_merged))
Y_5_merged = Y_5_merged.clip(0.0,10000000)
Y_10_merged = Y_10_SQUARECA
print(np.min(Y_10_merged))
print(np.max(Y_10_merged))
print(np.mean(Y_10_merged))
Y_10_merged = Y_10_merged.clip(0.0,10000000)


def amend_P15_to_csv(FileName="predictionLR.csv"):#output csv and full data
    import csv
    with open(FileName, 'w') as csvfile:
        fieldnames = ['date_dt', 'city_code','district_code', 'dwell','flow_in','flow_out']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        start=20170819
        for j in range(5):
            for i in range(204):
                writer.writerow({'date_dt':start, 
                                 'city_code':CITY_OF_DISTRICTS[i],
                                 'district_code':DISTRICTS[i], 
                                 'dwell':Y_5_merged[i,0][0][j],
                                 'flow_in':Y_5_merged[i,0][1][j],
                                 'flow_out':Y_5_merged[i,0][2][j]})
            start+=1
        ###############################################################
        start=20171105
        for j in range(10):
            for i in range(204):
                writer.writerow({'date_dt':start, 
                                 'city_code':CITY_OF_DISTRICTS[i],
                                 'district_code':DISTRICTS[i], 
                                 'dwell':Y_10_merged[i,0][0][j],
                                 'flow_in':Y_10_merged[i,0][1][j],
                                 'flow_out':Y_10_merged[i,0][2][j]})
            start+=1
    _ = pd.read_csv(FileName)
    #print(_)
    _ = _.to_csv(FileName,index=False,header=None)
amend_P15_to_csv()
print("File saved!")


# In[ ]:



