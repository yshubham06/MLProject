import os
import sys

import pandas as pd
import numpy as np
import dill

from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)   #get the directory path
        os.makedirs(dir_path,exist_ok=True) #create the directory if it does not exist
        with open(file_path,"wb") as file_obj: #open the file in write mode and binary mode
            dill.dump(obj,file_obj) #dump the object into the file
    except Exception as e:
        raise CustomException(e,sys)#raise an exception if any error occurs
    
def evaluate_models(x_train, y_train, x_test, y_test, models,params):
        try:
            
            report = {} #create an empty dictionary to store the r2 score of each model

            for i in range(len(list(models))):
                model = list(models.values())[i] #get each and every model
                para = list(params.values())[i] #get the parameters of the model
                #para = list(models.keys())[i] #get the parameters of the model

                gs = GridSearchCV(model,para,cv=3)
                gs.fit(x_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(x_train,y_train)

                #model.fit(x_train,y_train) #fit the model

                y_train_pred = model.predict(x_train) #predict the model on train data

                y_test_pred = model.predict(x_test) #predict the model on test data

                train_model_score = r2_score(y_train,y_train_pred)#calculate the r2 score on train data

                test_model_score = r2_score(y_test,y_test_pred)#calculate the r2 score on test data

                report[list(models.keys())[i]] = test_model_score #append the r2 score to the report dictionary

            return report #return the report dictionary


        except Exception as e:
            raise CustomException(e,sys)

def load_object(file_path):#function to load the object from the file
    try:
        with open(file_path,"rb") as file_obj:#open the file in read mode and binary mode 
            obj = dill.load(file_obj)#load the object from the file
        return obj #return the object
    except Exception as e:
        raise CustomException(e,sys)